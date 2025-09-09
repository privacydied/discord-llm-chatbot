from __future__ import annotations

import asyncio
import json
import logging
import contextlib
import wave
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
import discord

from bot.config import load_config
from bot.retry_utils import API_RETRY_CONFIG, retry_async
from utils.opus import transcode_to_ogg_opus
from utils.waveform import compute_waveform_b64


IS_VOICE_MESSAGE_FLAG = 8192  # Discord message flags: IS_VOICE_MESSAGE
USER_AGENT = "DiscordVoicePublisher/1.0"
VOICE_MSG_FORBIDDEN_CODE = (
    50173  # Discord API error code when voice messages are disallowed in channel
)
BLOCK_TTL_SECONDS = 15 * 60  # 15 minutes [PA]


@dataclass
class VoicePublishResult:
    message: Optional[discord.Message]
    ogg_path: Optional[Path]
    ok: bool


class VoiceMessagePublisher:
    """
    Publish Discord-native voice messages via attachments.create + upload + message create.
    Follows official-ish flow documented by community references. [CA][REH][IV]
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        # Channels that recently returned 50173 (Cannot send voice messages in this channel) [REH]
        self._blocked_channels: dict[int, float] = {}
        # HTTP timeouts (seconds); defaults preserve legacy behavior
        self._attachments_timeout_s: float = 30.0
        self._upload_timeout_s: float = 60.0
        self._message_post_timeout_s: float = 30.0
        # Preflight tool availability cache
        self._tools_checked: bool = False
        self._tools_ok: bool = False

    def _check_tools(self) -> bool:
        """Preflight check for ffmpeg and ffprobe. Cache the result and log once. [REH]"""
        if self._tools_checked:
            return self._tools_ok
        self._tools_checked = True
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        self._tools_ok = bool(ffmpeg_path) and bool(ffprobe_path)
        if not self._tools_ok:
            self.logger.warning(
                "voice.native.tools_missing | ffmpeg/ffprobe not found; disabling native voice for this run"
            )
        else:
            self.logger.debug(
                f"voice.native.tools_ok | ffmpeg={ffmpeg_path} ffprobe={ffprobe_path}"
            )
        return self._tools_ok

    def _is_blocked(self, channel_id: int) -> bool:
        now = time.monotonic()
        exp = self._blocked_channels.get(channel_id)
        if exp is None:
            return False
        if exp <= now:
            # expire entry lazily
            self._blocked_channels.pop(channel_id, None)
            return False
        return True

    def _block(self, channel_id: int) -> None:
        self._blocked_channels[channel_id] = time.monotonic() + BLOCK_TTL_SECONDS

    async def _attachments_create(
        self,
        session: aiohttp.ClientSession,
        channel_id: int,
        token: str,
        filename: str,
        file_size: int,
    ) -> dict:
        url = f"https://discord.com/api/v10/channels/{channel_id}/attachments"
        payload = {"files": [{"filename": filename, "file_size": file_size, "id": "0"}]}
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        async def _do():
            async with session.post(
                url, headers=headers, json=payload, timeout=self._attachments_timeout_s
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    err = aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message=text,
                    )
                    # Respect Retry-After if provided by Discord [REH]
                    try:
                        ra = resp.headers.get("Retry-After")
                        if ra is not None:
                            err.retry_after_seconds = float(ra)
                    except Exception:
                        pass
                    raise err
                return await resp.json()

        return await retry_async(_do, API_RETRY_CONFIG)

    async def _upload_file(
        self,
        session: aiohttp.ClientSession,
        upload_url: str,
        token: str,
        ogg_bytes: bytes,
    ) -> None:
        # Do NOT send bot Authorization to the upload_url (usually a signed CDN/S3 URL). [REH]
        headers = {"Content-Type": "audio/ogg"}

        async def _do():
            async with session.put(
                upload_url,
                headers=headers,
                data=ogg_bytes,
                timeout=self._upload_timeout_s,
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    err = aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message=text,
                    )
                    try:
                        ra = resp.headers.get("Retry-After")
                        if ra is not None:
                            err.retry_after_seconds = float(ra)
                    except Exception:
                        pass
                    raise err
                return None

        return await retry_async(_do, API_RETRY_CONFIG)

    async def _post_voice_message(
        self,
        session: aiohttp.ClientSession,
        channel_id: int,
        token: str,
        uploaded_filename: str,
        duration_secs: float,
        waveform_b64: str,
        reply_to_id: Optional[int],
    ) -> dict:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        attachments = [
            {
                "id": "0",
                "filename": "voice-message.ogg",
                "uploaded_filename": uploaded_filename,
                "duration_secs": float(round(duration_secs, 3)),
                "waveform": waveform_b64,
            }
        ]
        payload: dict = {"flags": IS_VOICE_MESSAGE_FLAG, "attachments": attachments}
        if reply_to_id:
            payload["message_reference"] = {
                "message_id": str(reply_to_id),
                "fail_if_not_exists": False,
            }
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        async def _do():
            async with session.post(
                url, headers=headers, json=payload, timeout=self._message_post_timeout_s
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    err = aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message=text,
                    )
                    try:
                        ra = resp.headers.get("Retry-After")
                        if ra is not None:
                            err.retry_after_seconds = float(ra)
                    except Exception:
                        pass
                    raise err
                return await resp.json()

        return await retry_async(_do, API_RETRY_CONFIG)

    async def publish(
        self,
        *,
        message: discord.Message,
        wav_path: str | Path,
        include_transcript: bool = False,
        transcript_text: Optional[str] = None,
    ) -> VoicePublishResult:
        """
        Publish a native voice message to the message's channel replying to the original.
        Returns VoicePublishResult with the created discord.Message (if fetched) and ogg path.
        On failure, returns (None, ogg_path or None) and logs errors for fallback.
        """
        cfg = load_config()
        # Refresh HTTP timeouts from configuration/env [IV]
        try:
            # Global fallback (0.0 or missing -> ignore)
            global_to = float(cfg.get("VOICE_PUBLISHER_TIMEOUT_S", 0.0) or 0.0)
            att_to = float(
                cfg.get("VOICE_PUBLISHER_ATTACHMENTS_CREATE_TIMEOUT_S", 30.0)
            )
            upl_to = float(cfg.get("VOICE_PUBLISHER_UPLOAD_TIMEOUT_S", 60.0))
            msg_to = float(cfg.get("VOICE_PUBLISHER_MESSAGE_POST_TIMEOUT_S", 30.0))
            if global_to > 0:
                self._attachments_timeout_s = global_to
                self._upload_timeout_s = global_to
                self._message_post_timeout_s = global_to
            else:
                self._attachments_timeout_s = att_to
                self._upload_timeout_s = upl_to
                self._message_post_timeout_s = msg_to
        except Exception:
            # Keep defaults on parse errors
            pass
        if not cfg.get("VOICE_ENABLE_NATIVE", False):
            self.logger.debug("voice.native.disabled")
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        token = cfg.get("DISCORD_TOKEN")
        if not token:
            self.logger.error("voice.native.missing_token")
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        channel_id = getattr(message.channel, "id", None)
        if not channel_id:
            self.logger.error("voice.native.missing_channel_id")
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        # Skip early if we recently observed 50173 for this channel [REH]
        if isinstance(channel_id, int) and self._is_blocked(channel_id):
            self.logger.info(
                f"voice.native.skipping_blocked | channel_id={channel_id}",
            )
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        # 1) Prepare audio: transcode WAV -> OGG Opus, compute duration + waveform
        try:
            if not self._check_tools():
                return VoicePublishResult(message=None, ogg_path=None, ok=False)
            wav_p = Path(wav_path)
            # Compute duration directly from WAV header [REH]
            duration = 0.0
            if wav_p.exists():
                with contextlib.closing(wave.open(str(wav_p), "rb")) as wf:
                    fr = wf.getframerate() or 0
                    nf = wf.getnframes() or 0
                    duration = (nf / float(fr)) if fr else 0.0
            # Transcode close to reference script: 48kHz, libopus @ 64k
            bitrate = str(cfg.get("VOICE_PUBLISHER_OPUS_BITRATE", "64k") or "64k")
            vbr = str(cfg.get("VOICE_PUBLISHER_OPUS_VBR", "on") or "on")
            comp = int(cfg.get("VOICE_PUBLISHER_OPUS_COMP_LEVEL", 10) or 10)
            ogg_p = await transcode_to_ogg_opus(
                wav_p, bitrate=bitrate, vbr=vbr, compression_level=comp
            )
            ogg_bytes = ogg_p.read_bytes()
            waveform_b64 = compute_waveform_b64(wav_p)
            # Prefer probing duration from OGG (matches reference behavior)
            probed = await self._probe_duration(ogg_p)
            if probed is not None:
                duration = probed
        except Exception as e:
            self.logger.error(f"voice.native.audio_prep_failed | {e}", exc_info=True)
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        # 2) Upload flow
        try:
            async with aiohttp.ClientSession(raise_for_status=False) as session:
                # Use a standard display filename per reference script
                filename_display = "voice-message.ogg"
                meta = await self._attachments_create(
                    session, channel_id, token, filename_display, len(ogg_bytes)
                )
                attach = (meta or {}).get("attachments", [{}])[0]
                upload_url = attach.get("upload_url")
                # API typically returns 'upload_filename'; be tolerant if 'uploaded_filename' appears.
                upload_filename = attach.get("upload_filename") or attach.get(
                    "uploaded_filename"
                )
                if not upload_url or not upload_filename:
                    raise RuntimeError(f"attachments.create missing fields: {attach}")

                await self._upload_file(session, upload_url, token, ogg_bytes)
                msg_json = await self._post_voice_message(
                    session,
                    channel_id,
                    token,
                    upload_filename,
                    duration,
                    waveform_b64,
                    getattr(message, "id", None),
                )

            # 3) Fetch created message (optional) and return success
            try:
                created_id = (
                    int(msg_json.get("id"))
                    if isinstance(msg_json.get("id"), (str, int))
                    else None
                )
                created_msg: Optional[discord.Message] = None
                if created_id:
                    created_msg = await message.channel.fetch_message(created_id)
                self.logger.info(
                    "voice.native.ok",
                    extra={
                        "subsys": "voice",
                        "event": "native_voice_ok",
                        "channel_id": channel_id,
                    },
                )
                return VoicePublishResult(message=created_msg, ogg_path=ogg_p, ok=True)
            except Exception:
                # If we cannot fetch, still consider it success if we got a valid JSON back
                self.logger.info(
                    "voice.native.ok.fetch_failed",
                    extra={
                        "subsys": "voice",
                        "event": "native_voice_ok_nofetch",
                        "channel_id": channel_id,
                    },
                )
                return VoicePublishResult(message=None, ogg_path=ogg_p, ok=True)

        except Exception as e:
            # Detect Discord code 50173 and block channel for a while [REH]
            blocked = False
            # Attempt to parse known structure from aiohttp.ClientResponseError
            if hasattr(e, "status") and getattr(e, "status", None) == 400:
                msg_text = getattr(e, "message", "") or ""
                try:
                    data = json.loads(msg_text)
                    if int(
                        data.get("code", 0)
                    ) == VOICE_MSG_FORBIDDEN_CODE and isinstance(channel_id, int):
                        self._block(channel_id)
                        blocked = True
                except Exception:
                    # Fallback string check
                    if str(VOICE_MSG_FORBIDDEN_CODE) in msg_text and isinstance(
                        channel_id, int
                    ):
                        self._block(channel_id)
                        blocked = True

            if blocked:
                self.logger.info(
                    f"voice.native.blocked_channel | channel_id={channel_id} ttl={BLOCK_TTL_SECONDS}s",
                )
            self.logger.error(f"voice.native.upload_failed | {e}", exc_info=True)
            return VoicePublishResult(message=None, ogg_path=ogg_p, ok=False)

    async def _probe_duration(self, path: Path) -> Optional[float]:
        """Probe audio duration using ffprobe; return seconds (rounded to 0 decimals per reference) or None."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "error",
                "-hide_banner",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None
            data = json.loads(stdout.decode("utf-8", errors="ignore"))
            dur = float(data.get("format", {}).get("duration", 0.0))
            return float(int(round(dur)))
        except Exception:
            return None
