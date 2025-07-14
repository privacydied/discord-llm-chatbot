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
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
MODEL_URL = "https://huggingface.co/NeuML/kokoro-base-onnx/resolve/main/model.onnx"
CONFIG_URL = "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json"

# Model paths
MODEL_PATH = Path("tts/kokoro-v1.0.onnx")
VOICES_PATH = Path("tts/voices.json")
CONFIG_PATH = Path("tts/config.json")

def check_tts_files_exist():
    """Check if all required TTS files exist."""
    return MODEL_PATH.exists() and VOICES_PATH.exists() and CONFIG_PATH.exists()

def create_dirs():
    """Create necessary directories for TTS files."""
    os.makedirs("tts", exist_ok=True)

def download_model():
    """Download the ONNX model file."""
    logger.info(f"📥 Downloading TTS model from {MODEL_URL}")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = int(50 * downloaded / total_size) if total_size > 0 else 0
                    sys.stdout.write(f"\r[{'=' * progress}{' ' * (50 - progress)}] {downloaded // (1024 * 1024)}MB/{total_size // (1024 * 1024)}MB")
                    sys.stdout.flush()
        
        logger.info(f"✅ Model downloaded to {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return False

def download_voices():
    """Download the voices.json file."""
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
    """Download the config.json file."""
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
    
    if not VOICES_PATH.exists():
        success = success and download_voices()
    
    if not CONFIG_PATH.exists():
        success = success and download_config()
    
    if success:
        logger.info("✅ All TTS files downloaded successfully")
    else:
        logger.error("❌ Failed to download some TTS files")
    
    return success

if __name__ == "__main__":
    # When run as a script, download all files
    ensure_tts_files()
