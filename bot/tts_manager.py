from .tts import TTSManager
import logging
from .config import load_config  # Import config loader

logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Initialize TTSManager with config
tts_manager = TTSManager(config)

async def initialize_tts():
    """Initialize TTS manager asynchronously [CA]"""
    try:
        logger.info("🔊 Checking TTS availability...")
        # TTS initialization now happens in __init__, just verify it's available [CA]
        if tts_manager.is_available():
            logger.info("✅ TTS initialization completed successfully")
        else:
            logger.error("❌ TTS is not available - check configuration and model files")
    except Exception as e:
        logger.error(f"❌ TTS initialization check failed: {str(e)}")