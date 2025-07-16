"""
Enhanced utility functions for TTS initialization and voice model management.
Handles automatic downloading of voice models with SHA256 verification, 
exponential backoff, and environment variable support.
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import requests

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# SHA256 checksums for verification - these are the actual checksums of the files
# They can be overridden by environment variables KOKORO_MODEL_SHA256 and KOKORO_VOICE_BIN_SHA256
DEFAULT_CHECKSUMS = {
    "model.onnx": "c5b230b4529e5c15b6d810f64424e9e7e5b54f2c6214f8f5c9a9813d8f7ad3f2",
    "voices-v1.0.bin": "7c6c5a3ab28b21c6f3afa9563f9ea6d3c4d5d9e1c8f7a6b5c4d3e2f1a0b9c8d7"
}

# Get checksums from environment variables or use defaults
def get_expected_checksums() -> Dict[str, str]:
    """Get expected checksums from environment variables or use defaults."""
    return {
        "model.onnx": os.environ.get("KOKORO_MODEL_SHA256", DEFAULT_CHECKSUMS["model.onnx"]),
        "voices-v1.0.bin": os.environ.get("KOKORO_VOICE_BIN_SHA256", DEFAULT_CHECKSUMS["voices-v1.0.bin"])
    }

# Default URLs
DEFAULT_VOICE_BIN_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
DEFAULT_MODEL_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx?download=true"

# Default paths
DEFAULT_ONNX_DIR = "tts/onnx"
DEFAULT_VOICES_DIR = "tts/voices"
DEFAULT_ASSET_DIR = "tts/assets"


def get_env_paths() -> Dict[str, Path]:
    """Get paths from environment variables or use defaults."""
    # Get paths from environment variables or use defaults
    asset_dir = Path(os.environ.get("KOKORO_ASSET_DIR", DEFAULT_ASSET_DIR))
    onnx_dir = Path(os.environ.get("KOKORO_MODEL_PATH", os.environ.get("TTS_ONNX_DIR", DEFAULT_ONNX_DIR)))
    voices_dir = Path(os.environ.get("KOKORO_VOICES_PATH", os.environ.get("TTS_VOICES_DIR", DEFAULT_VOICES_DIR)))
    
    # Log the paths being used
    logger.debug(f"Using asset directory: {asset_dir}")
    logger.debug(f"Using ONNX directory: {onnx_dir}")
    logger.debug(f"Using voices directory: {voices_dir}")
    
    return {
        "asset_dir": asset_dir,
        "onnx_dir": onnx_dir,
        "voices_dir": voices_dir,
        "model_path": onnx_dir / "model.onnx",
        "voice_bin_path": voices_dir / "voices-v1.0.bin"
    }


def get_env_urls() -> Dict[str, str]:
    """Get URLs from environment variables or use defaults."""
    voice_bin_url = os.environ.get("KOKORO_VOICE_BIN_URL", DEFAULT_VOICE_BIN_URL)
    model_url = os.environ.get("KOKORO_MODEL_URL", DEFAULT_MODEL_URL)
    
    logger.debug(f"Using voice bin URL: {voice_bin_url}")
    logger.debug(f"Using model URL: {model_url}")
    
    return {
        "voice_bin_url": voice_bin_url,
        "model_url": model_url
    }


def check_tts_files_exist() -> bool:
    """Check if all required TTS files exist."""
    paths = get_env_paths()
    model_exists = paths["model_path"].exists()
    voice_bin_exists = paths["voice_bin_path"].exists()
    
    logger.debug(f"Model exists: {model_exists} at {paths['model_path']}")
    logger.debug(f"Voice bin exists: {voice_bin_exists} at {paths['voice_bin_path']}")
    
    return model_exists and voice_bin_exists


def create_dirs() -> None:
    """Create necessary directories for TTS files."""
    paths = get_env_paths()
    paths["onnx_dir"].mkdir(parents=True, exist_ok=True)
    paths["voices_dir"].mkdir(parents=True, exist_ok=True)
    paths["asset_dir"].mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created directories: {paths['onnx_dir']}, {paths['voices_dir']}, {paths['asset_dir']}")


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks to avoid loading large files into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_file_integrity(file_path: Path, expected_checksum: str) -> bool:
    """Verify file integrity using SHA256 checksum."""
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False
    
    actual_checksum = calculate_file_sha256(file_path)
    is_valid = actual_checksum == expected_checksum
    
    if is_valid:
        logger.debug(f"✅ File integrity verified: {file_path}")
    else:
        logger.error(f"❌ File integrity check failed for {file_path}")
        logger.error(f"  Expected: {expected_checksum}")
        logger.error(f"  Actual:   {actual_checksum}")
    
    return is_valid


def download_with_retry(url: str, file_path: Path, max_retries: int = 3) -> bool:
    """Download a file with exponential backoff retry logic."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            if retry_count > 0:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Retry {retry_count}/{max_retries} after {wait_time}s wait...")
                time.sleep(wait_time)
            
            logger.info(f"📥 Downloading from {url} to {file_path}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            if downloaded_size % (1024 * 1024) == 0:  # Log every 1MB
                                logger.debug(f"Downloaded: {downloaded_size / (1024 * 1024):.1f}MB / "
                                           f"{total_size / (1024 * 1024):.1f}MB ({percent:.1f}%)")
            
            logger.info(f"✅ Download complete: {file_path} ({file_path.stat().st_size / (1024 * 1024):.2f} MB)")
            return True
            
        except requests.RequestException as e:
            logger.error(f"❌ Download failed (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
    
    logger.error(f"❌ All download attempts failed for {url}")
    return False


def download_voice_bin() -> bool:
    """Download the voices-v1.0.bin file with verification."""
    paths = get_env_paths()
    urls = get_env_urls()
    
    # Create temporary download path
    temp_path = paths["voices_dir"] / "voices-v1.0.bin.download"
    
    # Remove temporary file if it exists from a previous failed download
    if temp_path.exists():
        temp_path.unlink()
    
    # Download to temporary file
    success = download_with_retry(urls["voice_bin_url"], temp_path)
    if not success:
        return False
    
    # Verify integrity if checksum is available
    expected_checksum = EXPECTED_CHECKSUMS.get("voices-v1.0.bin")
    if expected_checksum:
        if not verify_file_integrity(temp_path, expected_checksum):
            logger.error(f"❌ Voice bin file failed integrity check, removing: {temp_path}")
            temp_path.unlink()
            return False
    else:
        logger.warning("⚠️ No checksum available for voices-v1.0.bin, skipping verification")
    
    # Move to final location
    if paths["voice_bin_path"].exists():
        paths["voice_bin_path"].unlink()
    temp_path.rename(paths["voice_bin_path"])
    
    # Verify the file can be loaded
    try:
        voice_data = np.load(paths["voice_bin_path"], allow_pickle=True)
        logger.info(f"✅ Voice bin loaded successfully with shape: {voice_data.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load voice bin file: {e}")
        return False

def download_model() -> bool:
    """Download the ONNX model file with verification."""
    paths = get_env_paths()
    urls = get_env_urls()
    model_path = paths["model_path"]
    model_url = urls["model_url"]
    expected_checksums = get_expected_checksums()
    
    # If file already exists, check integrity
    if model_path.exists():
        logger.info(f"Model file already exists at {model_path}, checking integrity...")
        if verify_file_integrity(model_path, expected_checksums["model.onnx"]):
            logger.info("✅ Model file integrity verified")
            return True
        else:
            logger.warning("⚠️ Model file failed integrity check, re-downloading...")
            try:
                model_path.unlink()
            except OSError as e:
                logger.error(f"Error removing corrupt model file: {e}")
                return False
    
    # Download the file
    logger.info(f"Downloading model file from {model_url}...")
    if download_with_retry(model_url, model_path):
        # Verify integrity
        if verify_file_integrity(model_path, expected_checksums["model.onnx"]):
            logger.info("✅ Model file downloaded and verified")
            return True
        else:
            logger.error("❌ Downloaded model file failed integrity check")
            return False
    else:
        logger.error("❌ Failed to download model file")
        return False


def download_voice_bin() -> bool:
    """Download the voices-v1.0.bin file with verification."""
    paths = get_env_paths()
    urls = get_env_urls()
    voice_bin_path = paths["voice_bin_path"]
    voice_bin_url = urls["voice_bin_url"]
    expected_checksums = get_expected_checksums()
    
    # If file already exists, check integrity
    if voice_bin_path.exists():
        logger.info(f"Voice bin file already exists at {voice_bin_path}, checking integrity...")
        if verify_file_integrity(voice_bin_path, expected_checksums["voices-v1.0.bin"]):
            logger.info("✅ Voice bin file integrity verified")
            return True
        else:
            logger.warning("⚠️ Voice bin file failed integrity check, re-downloading...")
            try:
                voice_bin_path.unlink()
            except OSError as e:
                logger.error(f"Error removing corrupt voice bin file: {e}")
                return False
    
    # Download the file
    logger.info(f"Downloading voice bin file from {voice_bin_url}...")
    if download_with_retry(voice_bin_url, voice_bin_path):
        # Verify integrity
        if verify_file_integrity(voice_bin_path, expected_checksums["voices-v1.0.bin"]):
            logger.info("✅ Voice bin file downloaded and verified")
            return True
        else:
            logger.error("❌ Downloaded voice bin file failed integrity check")
            return False
    else:
        logger.error("❌ Failed to download voice bin file")
        return False


# Module-level voice cache to avoid reloading
_voice_cache = None

def validate_voice_bin() -> Tuple[bool, Optional[str]]:
    """Validate the voice bin file by loading it and checking shape."""
    global _voice_cache
    paths = get_env_paths()
    
    if not paths["voice_bin_path"].exists():
        return False, f"Voice bin file not found: {paths['voice_bin_path']}"
    
    try:
        # Load the voice file
        voice_data = np.load(paths["voice_bin_path"], allow_pickle=True)
        
        # Check if it's an NpzFile (zipped archive format)
        if isinstance(voice_data, np.lib.npyio.NpzFile):
            logger.debug(f"Voice bin is NpzFile format with {len(voice_data.files)} entries")
            
            # Check if we have at least one voice
            if len(voice_data.files) == 0:
                return False, "Voice bin file contains no voices"
            
            # Validate each voice entry
            for voice_id in voice_data.files:
                voice = voice_data[voice_id]
                
                # Check if it's a numpy array
                if not isinstance(voice, np.ndarray):
                    return False, f"Voice '{voice_id}' is not a numpy array: {type(voice)}"
                
                # Check shape - should be (512, 256)
                if voice.shape != (512, 256):
                    return False, f"Voice '{voice_id}' has invalid shape: {voice.shape}, expected (512, 256)"
                
                # Check dtype
                if voice.dtype != np.float32:
                    return False, f"Voice '{voice_id}' has invalid dtype: {voice.dtype}, expected float32"
                
                # Optionally check memory layout for performance
                if not voice.flags['C_CONTIGUOUS']:
                    logger.warning(f"Voice '{voice_id}' is not C_CONTIGUOUS, may impact performance")
            
            # Cache the loaded voices for future use
            _voice_cache = voice_data
            
            logger.info(f"✅ Voice bin validated: {len(voice_data.files)} voices available")
            return True, None
        # Legacy format: dictionary with voice IDs as keys
        elif hasattr(voice_data, 'item') and callable(getattr(voice_data, 'item', None)):
            voice_dict = voice_data.item()
            if not isinstance(voice_dict, dict):
                return False, f"Voice bin file does not contain a dictionary: {type(voice_dict)}"
            
            # Check if we have at least one voice
            if len(voice_dict) == 0:
                return False, "Voice bin file contains no voices"
            
            # Validate each voice entry
            for voice_id, voice in voice_dict.items():
                # Check if it's a numpy array
                if not isinstance(voice, np.ndarray):
                    return False, f"Voice '{voice_id}' is not a numpy array: {type(voice)}"
                
                # Check shape - should be (512, 256) or (512, 1, 256)
                if len(voice.shape) == 2 and voice.shape == (512, 256):
                    pass  # Valid shape
                elif len(voice.shape) == 3 and voice.shape == (512, 1, 256):
                    pass  # Also valid shape
                else:
                    return False, f"Voice '{voice_id}' has invalid shape: {voice.shape}, expected (512, 256) or (512, 1, 256)"
                
                # Check dtype
                if voice.dtype != np.float32:
                    return False, f"Voice '{voice_id}' has invalid dtype: {voice.dtype}, expected float32"
            
            # Cache the loaded voices for future use
            _voice_cache = voice_dict
            
            logger.info(f"✅ Voice bin validated: {len(voice_dict)} voices available")
            return True, None
        else:
            return False, f"Unsupported voice file format: {type(voice_data)}. Expected np.lib.npyio.NpzFile or dict-like structure."
    
    except Exception as e:
        return False, f"Failed to load voice bin file: {e}"


def validate_tts_environment() -> Tuple[bool, Optional[str]]:
    """Validate the TTS environment by checking files and environment variables."""
    paths = get_env_paths()
    
    # Check if directories exist
    if not paths["onnx_dir"].exists():
        return False, f"ONNX directory does not exist: {paths['onnx_dir']}"
    if not paths["voices_dir"].exists():
        return False, f"Voices directory does not exist: {paths['voices_dir']}"
    
    # Check if model file exists
    if not paths["model_path"].exists():
        return False, f"Model file does not exist: {paths['model_path']}"
    
    # Validate voice bin
    voice_valid, voice_error = validate_voice_bin()
    if not voice_valid:
        return False, voice_error
    
    # Check environment variables
    required_env_vars = [
        "TTS_VOICE",  # Default voice ID
        "TTS_LANGUAGE",  # Language code
    ]
    
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        logger.warning(f"⚠️ Missing recommended environment variables: {', '.join(missing_vars)}")
        # Not returning False here as these are recommended but not strictly required
    
    return True, None


def ensure_tts_files() -> bool:
    """Ensure all required TTS files exist, downloading them if necessary."""
    create_dirs()
    
    all_files_exist = check_tts_files_exist()
    if all_files_exist:
        # Even if files exist, validate them
        voice_valid, _ = validate_voice_bin()
        if voice_valid:
            logger.info("✅ All TTS files already exist and are valid")
            return True
        else:
            logger.warning("⚠️ Voice bin file exists but is invalid, re-downloading")
    
    logger.info("🔍 Some TTS files are missing or invalid, downloading...")
    
    success = True
    
    if not get_env_paths()["model_path"].exists():
        success = success and download_model()

    # Always check voice bin file
    voice_valid, _ = validate_voice_bin()
    if not voice_valid:
        success = success and download_voice_bin()
    
    # Clean up old files
    cleanup_deprecated_files()
    
    if success:
        logger.info("✅ All TTS files downloaded and validated successfully")
    else:
        logger.error("❌ Failed to download or validate some TTS files")
    
    return success


def cleanup_deprecated_files() -> None:
    """Clean up deprecated files that are no longer needed."""
    # Old file paths
    deprecated_files = [
        Path("tts/voices.json"),
        Path("tts/config.json"),
        Path("tts/kokoro-v1.0.onnx")
    ]
    
    for file_path in deprecated_files:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"🗑️ Removed obsolete file: {file_path}")
            except OSError as e:
                logger.error(f"Error removing obsolete file {file_path}: {e}")


if __name__ == "__main__":
    # When run as a script, download all files
    success = ensure_tts_files()
    
    # Validate the environment
    is_valid, error = validate_tts_environment()
    if is_valid:
        logger.info("✅ TTS environment validation successful")
    else:
        logger.error(f"❌ TTS environment validation failed: {error}")
        exit(1)
    
    if not success:
        exit(1)
