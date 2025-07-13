from .tts import TTSManager
import logging

logger = logging.getLogger(__name__)

tts_manager = TTSManager()

async def initialize_tts():
    """Initialize TTS manager asynchronously"""
    try:
        logger.info("🔊 Initializing TTS model...")
        await tts_manager.load_model()
        logger.info("✅ TTS initialization completed")
    except Exception as e:
        logger.error(f"❌ TTS initialization failed: {str(e)}")