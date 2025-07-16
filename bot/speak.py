"""
Centralized TTS inference module (speak)
"""
import logging
from pathlib import Path
from .tts_manager import tts_manager
import tempfile
from .exceptions import TTSAudioError

logger = logging.getLogger(__name__)

async def speak_infer(text: str) -> Path:
    """Synthesize speech from text"""
    try:
        logger.info("🔊 TTS inference started")
        if not tts_manager.is_available():
            raise TTSAudioError("TTS engine not available")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            audio_path = Path(tmpfile.name)
        
        result = await tts_manager.synthesize_async(text, audio_path)
        
        # Ensure we always return a valid Path object
        if result is None:
            logger.error("TTS synthesis returned None")
            raise TTSAudioError("Speech synthesis failed: No output path returned")
            
        # Convert to Path if it's a string
        if isinstance(result, str):
            logger.debug(f"Converting string path to Path object: {result}")
            result = Path(result)
            
        # Ensure result is a Path object
        if not isinstance(result, Path):
            logger.error(f"TTS synthesis returned non-Path object: {type(result)}")
            raise TTSAudioError(f"Speech synthesis failed: Invalid return type {type(result)}")
            
        # Verify the file exists and has content
        if not result.exists():
            logger.error(f"TTS output file does not exist: {result}")
            raise TTSAudioError("Speech synthesis failed: Output file does not exist")
            
        if result.stat().st_size == 0:
            logger.error(f"TTS output file is empty: {result}")
            raise TTSAudioError("Speech synthesis failed: Output file is empty")
            
        logger.debug(f"TTS synthesis successful: {result}, size: {result.stat().st_size} bytes")
        return result
    except Exception as e:
        logger.error(f"🔊 TTS inference failed: {str(e)}")
        raise TTSAudioError(f"Speech synthesis failed: {str(e)}")