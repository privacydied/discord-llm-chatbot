"""
Centralized speech-to-text inference module (hear)
Supports both Discord voice messages and URL-based video audio ingestion.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
from .stt_orchestrator import stt_orchestrator
from .exceptions import InferenceError
from .util.logging import get_logger

logger = get_logger(__name__)


def _get_file_extension(filename: str) -> str:
    """Get the lowercase file extension with leading dot."""
    return os.path.splitext(filename)[1].lower()


async def hear_infer(audio_path: Path) -> str:
    """Transcribe audio to text with automatic format normalization and 1.5x speedup"""
    temp_wav = None
    temp_sped_up = None

    try:
        logger.info(f"👂 STT inference started for {audio_path}")
        # Let orchestrator handle availability/fallback [REH]

        # Create temporary files
        with (
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file1,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file2,
        ):
            temp_normalized = Path(temp_file1.name)
            temp_sped_up = Path(temp_file2.name)

        try:
            # First normalize the audio to 16kHz mono WAV
            normalize_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-af",
                "aresample=async=1:first_pts=0",  # Handle timestamp gaps
                str(temp_normalized),
            ]

            proc = await asyncio.create_subprocess_exec(
                *normalize_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = f"ffmpeg normalization failed with code {proc.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode()}"
                raise RuntimeError(error_msg)

            # Apply 1.5x speedup
            speedup_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_normalized),
                "-filter:a",
                "atempo=1.5",  # 1.5x speedup
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                str(temp_sped_up),
            ]

            proc = await asyncio.create_subprocess_exec(
                *speedup_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = f"ffmpeg speedup failed with code {proc.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode()}"
                raise RuntimeError(error_msg)

            # Transcribe the processed audio via orchestrator (falls back if disabled)
            logger.debug(f"Transcribing processed audio (orchestrated): {temp_sped_up}")
            return await stt_orchestrator.transcribe(temp_sped_up)

        finally:
            # Clean up temporary files
            for temp_file in [temp_normalized, temp_sped_up]:
                if temp_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning(
                            f"Error deleting temporary file {temp_file}: {e}"
                        )

    except Exception as e:
        # Check if this is an expected VideoIngestError or InferenceError
        from .video_ingest import VideoIngestError

        if isinstance(e, (VideoIngestError, InferenceError)):
            logger.info(f"👂🎵 File STT: {str(e)}")
        else:
            logger.error(f"👂🎵 File STT inference failed: {str(e)}", exc_info=True)
        raise InferenceError(f"Audio transcription failed: {str(e)}")

        # Provide more user-friendly error messages
        if "no such file or directory" in str(e).lower():
            raise InferenceError("Could not access the audio file")
        elif "invalid data" in str(e).lower() or "invalid argument" in str(e).lower():
            raise InferenceError(
                "The audio file is corrupted or in an unsupported format"
            )
        elif "operation not permitted" in str(e).lower():
            raise InferenceError("Permission denied when processing the audio file")
        else:
            raise InferenceError(f"Speech recognition failed: {str(e)}")

    finally:
        # Final cleanup in case anything was missed
        for temp_file in [temp_wav, temp_sped_up]:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Error in final cleanup of {temp_file}: {e}")


async def hear_infer_from_url(
    url: str, speedup: float = 1.5, force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Transcribe audio from YouTube/TikTok URL with metadata preservation.

    Args:
        url: YouTube or TikTok URL
        speedup: Audio speedup factor (default 1.5x)
        force_refresh: Force re-download even if cached

    Returns:
        Dict containing transcription and metadata
    """
    try:
        logger.info(f"👂🎥 STT inference started for URL: {url}")
        # Let orchestrator handle availability/fallback [REH]

        # Import here to avoid circular imports
        from .video_ingest import fetch_and_prepare_url_audio

        # Fetch and prepare audio from URL
        processed_audio = await fetch_and_prepare_url_audio(url, speedup, force_refresh)

        # Transcribe the processed audio using orchestrator (uses providers/modes)
        logger.debug(
            f"Transcribing processed video audio (orchestrated): {processed_audio.audio_path}"
        )
        transcription = await stt_orchestrator.transcribe(processed_audio.audio_path)

        # Return transcription with rich metadata
        result = {
            "transcription": transcription,
            "metadata": {
                "source": processed_audio.metadata.source_type,
                "url": processed_audio.metadata.url,
                "title": processed_audio.metadata.title,
                "uploader": processed_audio.metadata.uploader,
                "upload_date": processed_audio.metadata.upload_date,
                "original_duration_s": processed_audio.metadata.duration_seconds,
                "processed_duration_s": processed_audio.processed_duration_seconds,
                "speedup_factor": processed_audio.speedup_factor,
                "cache_hit": processed_audio.cache_hit,
                "timestamp": processed_audio.timestamp.isoformat(),
            },
        }

        logger.info(
            f"✅ URL transcription completed: {processed_audio.metadata.title[:50]}..."
        )
        return result

    except Exception as e:
        error_msg = str(e).lower()

        # Check if this is an expected VideoIngestError (already logged appropriately upstream)
        from .video_ingest import VideoIngestError

        if isinstance(e, VideoIngestError) and (
            "no video or audio content found" in error_msg
            or "no video could be found" in error_msg
            or "failed to download video" in error_msg
        ):
            # Don't log scary tracebacks for expected "no video content" errors
            logger.info(f"👂🎥 URL STT: {str(e)}")
        else:
            # Log unexpected errors with full tracebacks
            logger.error(f"👂🎥 URL STT inference failed: {str(e)}", exc_info=True)

        # Provide user-friendly error messages
        if "unsupported url" in error_msg:
            raise InferenceError(
                "This URL is not supported. Please use YouTube or TikTok links."
            )
        elif "video too long" in error_msg:
            raise InferenceError(
                "This video is too long to process. Please try a shorter video."
            )
        elif "no video or audio content found" in error_msg:
            raise InferenceError(
                "🔍 No video or audio content found in this URL. This appears to be a text-only post or the content is not accessible."
            )
        elif "download failed" in error_msg:
            raise InferenceError(
                "Could not download the video. It may be private or unavailable."
            )
        elif "audio processing failed" in error_msg:
            raise InferenceError("Could not process the audio from this video.")
        else:
            raise InferenceError(f"Video transcription failed: {str(e)}")
