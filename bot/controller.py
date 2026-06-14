"""Hybrid multimodal pipeline controller."""

import contextlib
import logging
import os
import tempfile
from pathlib import Path

import discord

from .brain import brain_infer
from .exceptions import InferenceError
from .hear import hear_infer
from .see import see_infer
from .speak import speak_infer

logger = logging.getLogger(__name__)


async def hybrid_pipeline(ctx, content: str, mode: str = "both"):
    """Orchestrate multimodal processing pipeline."""
    try:
        logger.info(f"🚀 Starting hybrid pipeline in {mode} mode")

        # STT mode requires audio processing first
        if mode == "stt":
            if not ctx.message.attachments:
                await ctx.send("❌ Please provide an audio file for STT processing")
                return None

            audio_path = await download_attachment(ctx.message.attachments[0])
            try:
                content = await hear_infer(audio_path)
                logger.info(f"👂 Transcribed audio: {content}")
            finally:
                _safe_unlink(audio_path)

            # After STT, continue with text processing
            mode = "text"

        # Vision-Language processing
        if mode == "vl":
            if not ctx.message.attachments:
                await ctx.send("❌ Please provide an image for vision processing")
                return None

            image_path = await download_attachment(ctx.message.attachments[0])
            try:
                result = await see_infer(image_path, content)
                await ctx.send(result)
            finally:
                _safe_unlink(image_path)
            return None

        # Text processing core
        text_out = await brain_infer(content)
        replies = []

        # Text output modes
        if mode in ("text", "both"):
            replies.append(await ctx.send(text_out))

        # TTS output modes
        if mode in ("tts", "both"):
            try:
                audio_path = await speak_infer(text_out)
                replies.append(await ctx.send(file=discord.File(str(audio_path))))
            except InferenceError:
                logger.error("⚠️ TTS inference failed", exc_info=True)
                await ctx.send("⚠️ TTS failed. Please try again later.")

        return replies
    except Exception as e:
        logger.exception(f"🚨 Pipeline error: {e!s}")
        with contextlib.suppress(Exception):
            await ctx.send("⚠️ An error occurred while processing your request")


def _safe_unlink(path: Path) -> None:
    """Safely remove a temp file, ignoring errors."""
    try:
        if path and path.exists():
            os.unlink(path)
    except (OSError, PermissionError) as e:
        logger.debug(f"Failed to unlink temp file {path}: {e}")


async def download_attachment(attachment) -> Path:
    """Download Discord attachment to a secure temporary file.

    Uses tempfile.NamedTemporaryFile to avoid path-traversal attacks
    (attachment.filename is attacker-controlled) and filename collisions.
    """
    # Sanitize suffix from the original filename, fall back to .bin
    suffix = Path(attachment.filename).suffix or ".bin"
    # Only allow known safe suffixes to prevent writing executable content
    safe_suffixes = {
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".m4a",
        ".mp4",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bin",
    }
    if suffix.lower() not in safe_suffixes:
        suffix = ".bin"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    temp_file.close()  # Close the fd; attachment.save() will open it by path
    await attachment.save(temp_path)
    return temp_path
