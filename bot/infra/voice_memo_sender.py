"""Discord Voice-Memo Sender using 3-step REST API.

Now fully async and non-blocking to avoid event loop stalls that can cause
Discord heartbeat delays. Provides async APIs and guarded sync wrappers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import aiohttp

from utils.opus import transcode_to_ogg_opus
from utils.waveform import compute_waveform_b64

logger = logging.getLogger(__name__)


class VoiceMemoError(Exception):
    """Raised when voice memo creation or sending fails."""


async def _probe_duration_async(path: Path) -> float | None:
    """Probe audio duration using ffprobe asynchronously. Returns seconds or None."""
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
        # Match reference rounding behavior
        return float(round(dur))
    except (json.JSONDecodeError, OSError, ValueError, KeyError, TypeError):
        return None


async def wav_bytes_to_voice_memo_async(
    channel_id: int,
    wav_bytes: bytes,
    bot_token: str,
    *,
    bitrate: str = "64k",
    vbr: str = "on",
    compression_level: int = 10,
    attachments_timeout_s: float = 30.0,
    upload_timeout_s: float = 60.0,
    message_post_timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Convert WAV bytes to Discord voice memo and send (fully async, non-blocking).

    Returns Discord message response JSON.
    """
    if not wav_bytes:
        msg = "Empty WAV bytes provided"
        raise VoiceMemoError(msg)
    if not bot_token:
        msg = "Bot token required"
        raise VoiceMemoError(msg)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            ogg_path = Path(temp_dir) / "voice-message.ogg"

            # Write WAV data
            wav_path.write_bytes(wav_bytes)

            # Convert WAV to OGG (Opus @ 48kHz mono) asynchronously
            ogg_path = await transcode_to_ogg_opus(
                wav_path,
                out_path=ogg_path,
                bitrate=bitrate,
                vbr=vbr,
                compression_level=compression_level,
            )

            # Compute duration (prefer probing from OGG), file size, and waveform (from WAV header)
            duration_secs = await _probe_duration_async(ogg_path)
            if duration_secs is None:
                duration_secs = 1.0
            file_size = ogg_path.stat().st_size
            waveform_b64 = compute_waveform_b64(wav_path)

            # Step 1: Request upload URL
            attachments_url = f"https://discord.com/api/v10/channels/{channel_id}/attachments"
            headers_json = {
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
                "User-Agent": "DiscordVoiceMemoAsync/1.0",
            }
            attachment_payload = {"files": [{"filename": "voice-message.ogg", "file_size": file_size, "id": "0"}]}

            async with aiohttp.ClientSession(raise_for_status=False) as session:
                async with session.post(
                    attachments_url,
                    headers=headers_json,
                    json=attachment_payload,
                    timeout=attachments_timeout_s,
                ) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        msg = f"Failed to request upload URL: HTTP {resp.status} {text[:200]}"
                        raise VoiceMemoError(msg)
                    upload_data = await resp.json()
                try:
                    upload_info = upload_data["attachments"][0]
                    upload_url = upload_info["upload_url"]
                    uploaded_filename = upload_info.get("upload_filename") or upload_info.get("uploaded_filename")
                except (KeyError, IndexError, TypeError) as e:
                    msg = f"Invalid upload response format: {e}"
                    raise VoiceMemoError(msg)

                # Step 2: Upload file to provided URL (no bot Authorization header)
                ogg_bytes = ogg_path.read_bytes()
                headers_upload = {"Content-Type": "audio/ogg"}
                async with session.put(
                    upload_url,
                    headers=headers_upload,
                    data=ogg_bytes,
                    timeout=upload_timeout_s,
                ) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        msg = f"Failed to upload file: HTTP {resp.status} {text[:200]}"
                        raise VoiceMemoError(msg)

                # Step 3: Send message with voice memo flags
                message_url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
                message_payload = {
                    "flags": 8192,  # Voice message flag
                    "attachments": [
                        {
                            "id": "0",
                            "filename": "voice-message.ogg",
                            "uploaded_filename": uploaded_filename,
                            "duration_secs": float(duration_secs),
                            "waveform": waveform_b64,
                        },
                    ],
                }
                async with session.post(
                    message_url,
                    headers=headers_json,
                    json=message_payload,
                    timeout=message_post_timeout_s,
                ) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        msg = f"Failed to send message: HTTP {resp.status} {text[:200]}"
                        raise VoiceMemoError(msg)
                    result = await resp.json()

            logger.info(
                f"voice_memo.sent | duration={duration_secs}s size={file_size}B",
                extra={"subsys": "discord", "event": "voice_memo.sent"},
            )
            return result
    except Exception as e:
        if isinstance(e, VoiceMemoError):
            raise
        msg = f"Unexpected error: {e}"
        raise VoiceMemoError(msg)


def wav_bytes_to_voice_memo(
    channel_id: int,
    wav_bytes: bytes,
    bot_token: str,
    bitrate: str = "64k",
    vbr: str = "on",
    compression_level: int = 10,
) -> dict[str, Any]:
    """Synchronous wrapper kept for compatibility. Do NOT call from within an async context.
    Prefer `wav_bytes_to_voice_memo_async` to avoid blocking the event loop.
    """
    # Disallow running this in an active event loop to prevent blocking
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            msg = "wav_bytes_to_voice_memo must not be called in an async context; use wav_bytes_to_voice_memo_async"
            raise VoiceMemoError(msg)
    except RuntimeError:
        # No running loop, safe to proceed
        pass

    return asyncio.run(
        wav_bytes_to_voice_memo_async(
            channel_id,
            wav_bytes,
            bot_token,
            bitrate=bitrate,
            vbr=vbr,
            compression_level=compression_level,
        ),
    )


async def send_tts_voice_memo_async(channel_id: int, wav_bytes: bytes, bot_token: str | None = None) -> dict[str, Any]:
    """Async convenience to send TTS output as a Discord voice memo."""
    if bot_token is None:
        bot_token = os.getenv("DISCORD_TOKEN")
        if not bot_token:
            msg = "Bot token required. Provide via bot_token parameter or DISCORD_TOKEN environment variable."
            raise VoiceMemoError(msg)
    return await wav_bytes_to_voice_memo_async(channel_id, wav_bytes, bot_token)


def send_tts_voice_memo(channel_id: int, wav_bytes: bytes, bot_token: str | None = None) -> dict[str, Any]:
    """Synchronous convenience kept for compatibility. Avoid in async contexts."""
    if bot_token is None:
        bot_token = os.getenv("DISCORD_TOKEN")
        if not bot_token:
            msg = "Bot token required. Provide via bot_token parameter or DISCORD_TOKEN environment variable."
            raise VoiceMemoError(msg)
    # Guard against blocking the event loop
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            msg = "send_tts_voice_memo must not be called in an async context; use send_tts_voice_memo_async"
            raise VoiceMemoError(msg)
    except RuntimeError:
        pass
    return asyncio.run(wav_bytes_to_voice_memo_async(channel_id, wav_bytes, bot_token))
