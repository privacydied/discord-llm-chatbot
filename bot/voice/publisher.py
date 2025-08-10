from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
import discord

from bot.config import load_config
from bot.retry_utils import API_RETRY_CONFIG, retry_async
from utils.opus import transcode_to_ogg_opus
from utils.waveform import compute_waveform_b64
from utils.wav_stats import wav_stats


IS_VOICE_MESSAGE_FLAG = 8192  # Discord message flags: IS_VOICE_MESSAGE


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

    async def _attachments_create(self, session: aiohttp.ClientSession, channel_id: int, token: str, filename: str, file_size: int) -> dict:
        url = f"https://discord.com/api/v10/channels/{channel_id}/attachments"
        payload = {"files": [{"filename": filename, "file_size": file_size, "id": "0"}]}
        headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}

        async def _do():
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise aiohttp.ClientResponseError(request_info=resp.request_info, history=resp.history, status=resp.status, message=text)
                return await resp.json()

        return await retry_async(_do, API_RETRY_CONFIG)

    async def _upload_file(self, session: aiohttp.ClientSession, upload_url: str, token: str, ogg_bytes: bytes) -> None:
        headers = {"Authorization": f"Bot {token}", "Content-Type": "audio/ogg"}

        async def _do():
            async with session.put(upload_url, headers=headers, data=ogg_bytes, timeout=60) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise aiohttp.ClientResponseError(request_info=resp.request_info, history=resp.history, status=resp.status, message=text)
                return None

        return await retry_async(_do, API_RETRY_CONFIG)

    async def _post_voice_message(self, session: aiohttp.ClientSession, channel_id: int, token: str, uploaded_filename: str, duration_secs: float, waveform_b64: str, reply_to_id: Optional[int]) -> dict:
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
            payload["message_reference"] = {"message_id": str(reply_to_id), "fail_if_not_exists": False}
        headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}

        async def _do():
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise aiohttp.ClientResponseError(request_info=resp.request_info, history=resp.history, status=resp.status, message=text)
                return await resp.json()

        return await retry_async(_do, API_RETRY_CONFIG)

    async def publish(self, *,
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
        if not cfg.get("VOICE_ENABLE_NATIVE", False):
            self.logger.debug("voice.native.disabled")
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        token = cfg.get("DISCORD_TOKEN")
        if not token:
            self.logger.error("voice.native.missing_token")
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        channel_id = getattr(message.channel, 'id', None)
        if not channel_id:
            self.logger.error("voice.native.missing_channel_id")
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        # 1) Prepare audio: transcode WAV -> OGG Opus, compute duration + waveform
        try:
            wav_p = Path(wav_path)
            duration = wav_stats(wav_p).duration if wav_p.exists() else 0.0
            ogg_p = await transcode_to_ogg_opus(wav_p)
            ogg_bytes = ogg_p.read_bytes()
            waveform_b64 = compute_waveform_b64(wav_p)
        except Exception as e:
            self.logger.error(f"voice.native.audio_prep_failed | {e}", exc_info=True)
            return VoicePublishResult(message=None, ogg_path=None, ok=False)

        # 2) Upload flow
        try:
            async with aiohttp.ClientSession(raise_for_status=False) as session:
                meta = await self._attachments_create(session, channel_id, token, ogg_p.name, len(ogg_bytes))
                attach = (meta or {}).get("attachments", [{}])[0]
                upload_url = attach.get("upload_url")
                # API typically returns 'upload_filename'; be tolerant if 'uploaded_filename' appears.
                upload_filename = attach.get("upload_filename") or attach.get("uploaded_filename")
                if not upload_url or not upload_filename:
                    raise RuntimeError(f"attachments.create missing fields: {attach}")

                await self._upload_file(session, upload_url, token, ogg_bytes)
                msg_json = await self._post_voice_message(session, channel_id, token, upload_filename, duration, waveform_b64, getattr(message, 'id', None))
        except Exception as e:
            self.logger.error(f"voice.native.upload_failed | {e}", exc_info=True)
            return VoicePublishResult(message=None, ogg_path=ogg_p, ok=False)

        # 3) Fetch created message to return a discord.Message instance for context tracking
        try:
            created_id = int(msg_json.get('id')) if isinstance(msg_json.get('id'), (str, int)) else None
            created_msg: Optional[discord.Message] = None
            if created_id:
                created_msg = await message.channel.fetch_message(created_id)
            return VoicePublishResult(message=created_msg, ogg_path=ogg_p, ok=True)
        except Exception:
            # If we cannot fetch, still consider it success if we got a valid JSON back
            return VoicePublishResult(message=None, ogg_path=ogg_p, ok=True)
