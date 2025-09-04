"""
Centralized TTS inference module (speak)
"""
import logging
from pathlib import Path
from .tts.interface import TTSManager
from .exceptions import TTSAudioError

logger = logging.getLogger(__name__)


async def speak_infer(text: str) -> Path:
    """Synthesize speech from text using the async TTSManager.

    Returns a Path to a WAV file. Falls back to the stub engine if the
    primary engine is unavailable, without blocking the event loop.
    """
    manager = TTSManager()
    try:
        logger.info("🔊 TTS inference started")

        if not manager.is_available():
            # Proceed anyway; manager will fall back to StubEngine internally
            logger.warning("TTS primary engine unavailable; using stub fallback where applicable")

        # Ask the manager to generate a WAV file (non-blocking APIs internally)
        out_path, content_type = await manager.generate_tts(text, out_path=None, output_format="wav")

        # Validate output file
        if not isinstance(out_path, Path):
            out_path = Path(out_path)

        if not out_path.exists():
            logger.error(f"TTS output file does not exist: {out_path}")
            raise TTSAudioError("Speech synthesis failed: Output file does not exist")

        if out_path.stat().st_size == 0:
            logger.error(f"TTS output file is empty: {out_path}")
            raise TTSAudioError("Speech synthesis failed: Output file is empty")

        logger.debug(f"TTS synthesis successful: {out_path}, size: {out_path.stat().st_size} bytes, type: {content_type}")
        return out_path
    except Exception as e:
        logger.error(f"🔊 TTS inference failed: {str(e)}")
        raise TTSAudioError(f"Speech synthesis failed: {str(e)}")
    finally:
        try:
            await manager.close()
        except Exception:
            pass