"""Centralized TTS inference module (speak)."""

import contextlib
import logging
from pathlib import Path

from .exceptions import TTSAudioError
from .tts.errors import SynthesisError
from .tts.interface import TTSManager

logger = logging.getLogger(__name__)


async def speak_infer(text: str) -> Path:
    """Synthesize speech from text using the async TTSManager.

    Returns a Path to a WAV file. Falls back to the stub engine if the
    primary engine is unavailable, without blocking the event loop.
    """
    manager = TTSManager()
    try:
        logger.info("🔊 TTS inference started")

        status = manager.get_status()
        if status.get("degraded") and not status.get("explicit_stub"):
            logger.warning(
                "TTS engine degraded: %s",
                status.get("degraded_reason") or "unknown",
            )

        # Ask the manager to generate a WAV file (non-blocking APIs internally)
        gen_res = await manager.generate_tts(text, out_path=None, output_format="wav")
        if isinstance(gen_res, tuple):
            out_path, content_type = gen_res
        else:
            out_path = gen_res
            content_type = "audio/wav"

        # Validate output file
        if not isinstance(out_path, Path):
            out_path = Path(out_path)

        if not out_path.exists():
            logger.error(f"TTS output file does not exist: {out_path}")
            msg = "Speech synthesis failed: Output file does not exist"
            raise TTSAudioError(msg)

        if out_path.stat().st_size == 0:
            logger.error(f"TTS output file is empty: {out_path}")
            msg = "Speech synthesis failed: Output file is empty"
            raise TTSAudioError(msg)

        logger.debug(f"TTS synthesis successful: {out_path}, size: {out_path.stat().st_size} bytes, type: {content_type}")
        return out_path
    except SynthesisError as exc:
        status = manager.get_status()
        reason = status.get("degraded_reason") or str(exc)
        logger.exception(f"🔊 TTS inference failed: {reason}")
        msg = f"Speech synthesis failed: {reason}"
        raise TTSAudioError(msg)
    except Exception as e:
        logger.exception(f"🔊 TTS inference failed: {e!s}")
        msg = f"Speech synthesis failed: {e!s}"
        raise TTSAudioError(msg)
    finally:
        with contextlib.suppress(Exception):
            await manager.close()
