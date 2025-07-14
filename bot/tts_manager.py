from .tts import TTSManager
import logging
from .config import load_config  # Import config loader

logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Initialize TTSManager with config
tts_manager = TTSManager(config)

async def initialize_tts():
    """Initialize TTS manager asynchronously"""
    try:
        logger.info("🔊 Initializing TTS model...")
        await tts_manager.load_model()
        logger.info("✅ TTS initialization completed")
    except Exception as e:
        logger.error(f"❌ TTS initialization failed: {str(e)}")