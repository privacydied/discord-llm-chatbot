"""
Centralized speech-to-text inference module (hear)
"""
import logging
from pathlib import Path
from .stt import stt_manager
from .exceptions import InferenceError

logger = logging.getLogger(__name__)

async def hear_infer(audio_path: Path) -> str:
    """Transcribe audio to text"""
    try:
        logger.info("👂 STT inference started")
        if not stt_manager.is_available():
            raise InferenceError("STT engine not available")
        
        return await stt_manager.transcribe_async(audio_path)
    except Exception as e:
        logger.error(f"👂 STT inference failed: {str(e)}")
        raise InferenceError(f"Speech recognition failed: {str(e)}")