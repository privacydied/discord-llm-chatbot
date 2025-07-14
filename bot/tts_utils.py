"""
Utility functions for TTS initialization and voice model management.
Handles automatic downloading of voice models when required.
"""
import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import requests

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Voice model constants
VOICE_BIN_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
MODEL_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx?download=true"
# DEPRECATED
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
CONFIG_URL = "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json"

# Model paths
ONNX_DIR = Path("tts/onnx")
VOICES_DIR = Path("tts/voices")
MODEL_PATH = ONNX_DIR / "model.onnx"
VOICE_BIN_PATH = VOICES_DIR / "voices-v1.0.bin"

# The voices.json and config.json are no longer used by KokoroDirect, which uses .bin files
# We will remove the download logic for them but keep the paths for cleanup.
VOICES_PATH = Path("tts/voices.json")
CONFIG_PATH = Path("tts/config.json")

def check_tts_files_exist():
    """Check if all required TTS files exist."""
    # KokoroDirect requires both the model and the voice .bin file.
    return MODEL_PATH.exists() and VOICE_BIN_PATH.exists()

def create_dirs():
    """Create necessary directories for TTS files."""
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

def download_voice_bin():
    """Download the voices-v1.0.bin file."""
    logger.info(f"📥 Downloading TTS voice package from {VOICE_BIN_URL}")
    try:
        response = requests.get(VOICE_BIN_URL, stream=True)
        response.raise_for_status()
        with open(VOICE_BIN_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"✅ Voice package downloaded to {VOICE_BIN_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download voice package: {e}")
        return False

def download_model():
    """Download the ONNX model file."""
    logger.info(f"📥 Downloading TTS model from {MODEL_URL}")
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        
        # Verify file size
        file_size = MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Model downloaded, size: {file_size:.2f} MB")
        
        logger.info(f"✅ Model downloaded to {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return False

def download_voices():
    """DEPRECATED: Download the voices.json file. KokoroDirect uses .bin files."""
    logger.info(f"📥 Downloading TTS voices from {VOICES_URL}")
    try:
        response = requests.get(VOICES_URL)
        response.raise_for_status()
        
        with open(VOICES_PATH, 'w') as f:
            f.write(response.text)
        
        # Verify the voices file is valid JSON
        try:
            with open(VOICES_PATH, 'r') as f:
                voices_data = json.load(f)
                if isinstance(voices_data, dict):
                    # Get first voice and verify dimensions
                    first_voice_key = next(iter(voices_data))
                    first_voice = np.array(voices_data[first_voice_key])
                    if hasattr(first_voice, 'shape'):
                        dimensions = first_voice.shape[0] if first_voice.ndim == 1 else first_voice.shape[1]
                        logger.info(f"✅ Voices file contains {len(voices_data)} voices with {dimensions} dimensions")
                else:
                    logger.warning("⚠️ Voices file is not a dictionary")
        except Exception as ve:
            logger.error(f"❌ Downloaded voices file is not valid JSON: {ve}")
        
        logger.info(f"✅ Voices downloaded to {VOICES_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download voices: {e}")
        return False

def download_config():
    """DEPRECATED: Download the config.json file."""
    logger.info(f"📥 Downloading TTS config from {CONFIG_URL}")
    try:
        response = requests.get(CONFIG_URL)
        response.raise_for_status()
        
        with open(CONFIG_PATH, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"✅ Config downloaded to {CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download config: {e}")
        return False

def ensure_tts_files():
    """Ensure all required TTS files exist, downloading them if necessary."""
    create_dirs()
    
    all_files_exist = check_tts_files_exist()
    if all_files_exist:
        logger.info("✅ All TTS files already exist")
        return True
    
    logger.info("🔍 Some TTS files are missing, downloading...")
    
    success = True
    
    if not MODEL_PATH.exists():
        success = success and download_model()

    if not VOICE_BIN_PATH.exists():
        success = success and download_voice_bin()
    
    # The voices.json and config.json files are no longer needed for KokoroDirect.
    # The logic for downloading individual voice .bin files should be added if not present.
    # For now, we assume they are manually placed or handled elsewhere.

    # Clean up old files if they exist
    if VOICES_PATH.exists():
        try:
            VOICES_PATH.unlink()
            logger.info(f"🗑️ Removed obsolete voices.json file")
        except OSError as e:
            logger.error(f"Error removing obsolete voices.json: {e}")

    if CONFIG_PATH.exists():
        try:
            CONFIG_PATH.unlink()
            logger.info(f"🗑️ Removed obsolete config.json file")
        except OSError as e:
            logger.error(f"Error removing obsolete config.json: {e}")

    # Also remove the old model path if it exists
    old_model_path = Path("tts/kokoro-v1.0.onnx")
    if old_model_path.exists():
        try:
            old_model_path.unlink()
            logger.info(f"🗑️ Removed obsolete kokoro-v1.0.onnx model file")
        except OSError as e:
            logger.error(f"Error removing obsolete model file: {e}")
    
    if success:
        logger.info("✅ All TTS files downloaded successfully")
    else:
        logger.error("❌ Failed to download some TTS files")
    
    return success

if __name__ == "__main__":
    # When run as a script, download all files
    ensure_tts_files()
