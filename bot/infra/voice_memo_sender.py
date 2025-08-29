"""
Discord Voice-Memo Sender using 3-step REST API.

Converts WAV bytes to OGG (Opus 48kHz mono) and sends as Discord voice message
with proper flags and metadata for voice bubble UI.
"""

import os
import json
import base64
import subprocess
import tempfile
import logging
from typing import Dict, Any

import requests

logger = logging.getLogger(__name__)


class VoiceMemoError(Exception):
    """Raised when voice memo creation or sending fails."""
    pass


def _ffprobe_duration(audio_path: str) -> int:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-hide_banner", "-select_streams", "a:0",
        "-show_entries", "format=duration", "-of", "json", audio_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        return round(duration)
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to get audio duration: {e}")
        return 1  # Default to 1 second


def wav_bytes_to_voice_memo(
    channel_id: int,
    wav_bytes: bytes,
    bot_token: str,
    bitrate: str = "64k",
    vbr: str = "on",
    compression_level: int = 10
) -> Dict[str, Any]:
    """
    Convert WAV bytes to Discord voice memo and send.

    Args:
        channel_id: Discord channel ID to send to
        wav_bytes: WAV audio data
        bot_token: Discord bot token
        bitrate: Opus bitrate (default "64k")
        vbr: Variable bitrate setting (default "on")
        compression_level: Opus compression level 0-10 (default 10)

    Returns:
        Discord message response JSON

    Raises:
        VoiceMemoError: If conversion or sending fails
    """
    if not wav_bytes:
        raise VoiceMemoError("Empty WAV bytes provided")

    if not bot_token:
        raise VoiceMemoError("Bot token required")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = os.path.join(temp_dir, "audio.wav")
            ogg_path = os.path.join(temp_dir, "voice-message.ogg")

            # Write WAV data
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)

            # Convert WAV to OGG (Opus @ 48kHz mono)
            ffmpeg_cmd = [
                "ffmpeg", "-hide_banner", "-y", "-i", wav_path,
                "-ac", "1",  # Mono
                "-ar", "48000",  # 48kHz sample rate
                "-c:a", "libopus",  # Opus codec
                "-b:a", bitrate,  # Bitrate
                "-vbr", vbr,  # Variable bitrate
                "-compression_level", str(compression_level),
                ogg_path
            ]

            try:
                subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30
                )
            except subprocess.CalledProcessError as e:
                raise VoiceMemoError(f"FFmpeg conversion failed: {e}")
            except subprocess.TimeoutExpired:
                raise VoiceMemoError("FFmpeg conversion timed out")

            # Get duration and file size
            duration_secs = _ffprobe_duration(ogg_path)
            file_size = os.path.getsize(ogg_path)

            # Generate random waveform data (256 bytes base64)
            waveform = base64.b64encode(os.urandom(256)).decode("utf-8")

            # Step 1: Request upload URL
            attachments_url = f"https://discord.com/api/v10/channels/{channel_id}/attachments"
            headers = {
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json"
            }
            
            attachment_payload = {
                "files": [{
                    "filename": "voice-message.ogg",
                    "file_size": file_size,
                    "id": 0
                }]
            }

            try:
                upload_response = requests.post(
                    attachments_url,
                    headers=headers,
                    json=attachment_payload,
                    timeout=10
                )
                upload_response.raise_for_status()
                upload_data = upload_response.json()
                upload_info = upload_data["attachments"][0]
            except requests.RequestException as e:
                raise VoiceMemoError(f"Failed to request upload URL: {e}")
            except (KeyError, IndexError) as e:
                raise VoiceMemoError(f"Invalid upload response format: {e}")

            # Step 2: Upload file to provided URL
            try:
                with open(ogg_path, "rb") as audio_file:
                    upload_file_response = requests.put(
                        upload_info["upload_url"],
                        headers={"Content-Type": "audio/ogg"},
                        data=audio_file,
                        timeout=30
                    )
                    upload_file_response.raise_for_status()
            except requests.RequestException as e:
                raise VoiceMemoError(f"Failed to upload file: {e}")

            # Step 3: Send message with voice memo flags
            message_url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
            
            message_payload = {
                "flags": 8192,  # Voice message flag
                "attachments": [{
                    "id": "0",
                    "filename": "voice-message.ogg",
                    "uploaded_filename": upload_info["upload_filename"],
                    "duration_secs": duration_secs,
                    "waveform": waveform
                }]
            }

            try:
                message_response = requests.post(
                    message_url,
                    headers=headers,
                    json=message_payload,
                    timeout=10
                )
                message_response.raise_for_status()
                result = message_response.json()
                
                logger.info(
                    f"Voice memo sent: {duration_secs}s, {file_size} bytes",
                    extra={'subsys': 'discord', 'event': 'voice_memo.sent'}
                )
                
                return result
                
            except requests.RequestException as e:
                raise VoiceMemoError(f"Failed to send message: {e}")

    except Exception as e:
        if isinstance(e, VoiceMemoError):
            raise
        raise VoiceMemoError(f"Unexpected error: {e}")


def send_tts_voice_memo(channel_id: int, wav_bytes: bytes, bot_token: str = None) -> Dict[str, Any]:
    """
    Convenience function to send TTS output as Discord voice memo.
    
    Args:
        channel_id: Discord channel ID
        wav_bytes: WAV audio from TTS engine
        bot_token: Discord bot token (uses DISCORD_BOT_TOKEN env if not provided)
        
    Returns:
        Discord message response
        
    Raises:
        VoiceMemoError: If sending fails
    """
    if bot_token is None:
        bot_token = os.getenv("DISCORD_BOT_TOKEN")
        if not bot_token:
            raise VoiceMemoError(
                "Bot token required. Provide via bot_token parameter or DISCORD_BOT_TOKEN environment variable."
            )
    
    return wav_bytes_to_voice_memo(channel_id, wav_bytes, bot_token)
