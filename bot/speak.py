"""
Centralized TTS inference module (speak)
"""
import logging
from pathlib import Path
from .tts_manager import tts_manager
from .exceptions import InferenceError, TTSAudioError

logger = logging.getLogger(__name__)

async def speak_infer(text: str) -> Path:
    """Synthesize speech from text"""
    try:
        logger.info("🔊 TTS inference started")
        if not tts_manager.is_available():
            raise TTSAudioError("TTS engine not available")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            audio_path = Path(tmpfile.name)
        
        audio_path = await tts_manager.synthesize_async(text, audio_path)
        return audio_path
    except Exception as e:
        logger.error(f"🔊 TTS inference failed: {str(e)}")
        raise TTSAudioError(f"Speech synthesis failed: {str(e)}")