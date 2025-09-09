#!/usr/bin/env python
"""
Debug script to diagnose TTS initialization issues.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("debug_tts")

def check_file_exists(file_path):
    """Check if a file exists and log details about it."""
    path = Path(file_path)
    logger.info(f"Checking file: {path}")
    
    if path.exists():
        logger.info(f"✅ File exists: {path}")
        logger.info(f"  - Size: {path.stat().st_size} bytes")
        logger.info(f"  - Absolute path: {path.absolute()}")
        return True
    else:
        logger.error(f"❌ File does not exist: {path}")
        # Check parent directory
        parent = path.parent
        if parent.exists():
            logger.info(f"  - Parent directory exists: {parent}")
            # List files in parent directory
            logger.info("  - Files in parent directory:")
            for item in parent.iterdir():
                logger.info(f"    - {item.name} ({'dir' if item.is_dir() else 'file'})")
        else:
            logger.error(f"  - Parent directory does not exist: {parent}")
        return False

def main():
    """Main debug function."""
    logger.info("Starting TTS debug script")
    
    # Check environment variables
    logger.info("Checking environment variables:")
    tts_model_path = os.environ.get("TTS_MODEL_PATH")
    tts_voices_path = os.environ.get("TTS_VOICES_PATH")
    tts_model_file = os.environ.get("TTS_MODEL_FILE")
    tts_voice_file = os.environ.get("TTS_VOICE_FILE")
    
    logger.info(f"TTS_MODEL_PATH: {tts_model_path}")
    logger.info(f"TTS_VOICES_PATH: {tts_voices_path}")
    logger.info(f"TTS_MODEL_FILE: {tts_model_file}")
    logger.info(f"TTS_VOICE_FILE: {tts_voice_file}")
    
    # Check model file
    model_path = tts_model_path or tts_model_file or "tts/onnx/kokoro-v1.0.onnx"
    voices_path = tts_voices_path or tts_voice_file or "tts/voices/voices-v1.0.bin"
    
    model_exists = check_file_exists(model_path)
    voices_exist = check_file_exists(voices_path)
    
    if not model_exists or not voices_exist:
        logger.error("❌ Required files are missing!")
        return 1
    
    # Try to import and initialize KokoroDirect
    try:
        logger.info("Attempting to import KokoroDirect...")
        sys.path.insert(0, os.path.abspath('.'))
        from bot.kokoro_direct_fixed import KokoroDirect
        
        logger.info("Creating KokoroDirect instance...")
        kokoro = KokoroDirect(model_path, voices_path)
        
        logger.info("✅ KokoroDirect initialized successfully!")
        
        # Check available voices
        voices = kokoro.get_voice_names()
        logger.info(f"Available voices: {voices}")
        
        # Try to generate a test audio
        if voices:
            logger.info(f"Testing TTS generation with voice: {voices[0]}")
            try:
                output_path = kokoro.create("This is a test of the TTS system.", voices[0])
                logger.info(f"✅ TTS generation successful! Output saved to: {output_path}")
                if output_path.exists():
                    logger.info(f"  - Output file size: {output_path.stat().st_size} bytes")
                else:
                    logger.error("  - Output file does not exist!")
            except Exception as e:
                logger.error(f"❌ TTS generation failed: {e}", exc_info=True)
        
    except ImportError as e:
        logger.error(f"❌ Failed to import KokoroDirect: {e}", exc_info=True)
        return 1
    except Exception as e:
        logger.error(f"❌ Failed to initialize KokoroDirect: {e}", exc_info=True)
        return 1
    
    logger.info("Debug script completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
