"""
Centralized TTS inference module (speak)
"""
import logging
from pathlib import Path

import tempfile
from .exceptions import TTSAudioError

logger = logging.getLogger(__name__)

async def speak_infer(bot, text: str, audio_path: Path) -> Path:
    """Synthesize speech from text"""
    try:
        logger.info("🔊 TTS inference started")
        if not bot.tts_manager.is_available():
            raise TTSAudioError("TTS engine not available")
        
        audio_path = await bot.tts_manager.synthesize_async(text, audio_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            audio_path = Path(tmpfile.name)
        
        audio_path = await tts_manager.synthesize_async(text, audio_path)
        return audio_path
    except Exception as e:
        logger.error(f"🔊 TTS inference failed: {str(e)}")
        raise TTSAudioError(f"Speech synthesis failed: {str(e)}")