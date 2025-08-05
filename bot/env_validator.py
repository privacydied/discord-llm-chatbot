"""
Environment consistency validator for Discord bot.
Checks that all required environment variables and files are present and valid.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import TTS utilities for validation
try:
    from .tts.validation import validate_tts_environment, get_env_paths
    use_enhanced_utils = True
except ImportError:
    use_enhanced_utils = False
    
logger = logging.getLogger(__name__)

# Required environment variables by category
REQUIRED_ENV_VARS = {
    "core": [
        "DISCORD_TOKEN",  # Discord bot token
    ],
    "ai": [
        "TEXT_BACKEND",   # AI backend (openai, ollama)
    ],
    "openai": [
        "OPENAI_API_KEY", # OpenAI API key (required if TEXT_BACKEND=openai)
    ],
    "ollama": [
        "OLLAMA_BASE_URL", # Ollama base URL (required if TEXT_BACKEND=ollama)
        "OLLAMA_MODEL",    # Ollama model name (required if TEXT_BACKEND=ollama)
    ],
    "tts": [
        "TTS_VOICE",      # Default TTS voice ID
        "TTS_LANGUAGE",   # TTS language code
    ]
}

# Required directories
REQUIRED_DIRS = [
    "temp",              # Temporary files directory
    "user_profiles",     # User profiles directory
    "server_profiles",   # Server profiles directory
]

def validate_environment() -> Tuple[bool, Optional[str]]:
    """
    Validate the environment by checking required environment variables and directories.
    Returns (is_valid, error_message) tuple.
    """
    # Check core environment variables
    missing_core = [var for var in REQUIRED_ENV_VARS["core"] if not os.environ.get(var)]
    if missing_core:
        return False, f"Missing required environment variables: {', '.join(missing_core)}"
    
    # Check AI backend environment variables
    backend = os.environ.get("TEXT_BACKEND", "").lower()
    if backend == "openai":
        missing_openai = [var for var in REQUIRED_ENV_VARS["openai"] if not os.environ.get(var)]
        if missing_openai:
            return False, f"OpenAI backend selected but missing required variables: {', '.join(missing_openai)}"
    elif backend == "ollama":
        missing_ollama = [var for var in REQUIRED_ENV_VARS["ollama"] if not os.environ.get(var)]
        if missing_ollama:
            return False, f"Ollama backend selected but missing required variables: {', '.join(missing_ollama)}"
    else:
        return False, f"Invalid TEXT_BACKEND value: {backend}. Must be 'openai' or 'ollama'."
    
    # Check TTS environment variables
    missing_tts = [var for var in REQUIRED_ENV_VARS["tts"] if not os.environ.get(var)]
    if missing_tts:
        logger.warning(f"⚠️ Missing recommended TTS variables: {', '.join(missing_tts)}")
        # Not failing for TTS variables as they're not critical for core functionality
    
    # Check required directories
    for dir_name in REQUIRED_DIRS:
        dir_path = Path(os.environ.get(f"{dir_name.upper()}_DIR", dir_name))
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created missing directory: {dir_path}")
            except Exception as e:
                return False, f"Failed to create required directory {dir_path}: {e}"
    
    # Validate TTS environment if enhanced utils are available
    if use_enhanced_utils:
        try:
            is_valid, error = validate_tts_environment()
            if not is_valid:
                logger.warning(f"⚠️ TTS environment validation failed: {error}")
                # Not failing for TTS validation as it's not critical for core functionality
        except Exception as e:
            logger.error(f"❌ Error validating TTS environment: {e}", exc_info=True)
    
    return True, None

def get_environment_info() -> Dict[str, Any]:
    """
    Get information about the current environment.
    Returns a dictionary with environment details.
    """
    info = {
        "backend": os.environ.get("TEXT_BACKEND", "unknown"),
        "model": os.environ.get("OPENAI_TEXT_MODEL", os.environ.get("OPENAI_MODEL", os.environ.get("OLLAMA_MODEL", "unknown"))),
        "tts_voice": os.environ.get("TTS_VOICE", "unknown"),
        "tts_language": os.environ.get("TTS_LANGUAGE", "unknown"),
        "directories": {}
    }
    
    # Add directory information
    for dir_name in REQUIRED_DIRS:
        dir_path = Path(os.environ.get(f"{dir_name.upper()}_DIR", dir_name))
        info["directories"][dir_name] = {
            "path": str(dir_path),
            "exists": dir_path.exists(),
            "is_dir": dir_path.is_dir() if dir_path.exists() else False,
        }
    
    # Add TTS paths if available
    if use_enhanced_utils:
        try:
            tts_paths = get_env_paths()
            info["tts_paths"] = {k: str(v) for k, v in tts_paths.items()}
        except Exception as e:
            info["tts_paths"] = {"error": str(e)}
    
    return info

def validate_on_startup() -> bool:
    """
    Validate environment on startup and log results.
    Returns True if validation passed, False otherwise.
    """
    logger.info("Validating environment...")
    is_valid, error = validate_environment()
    
    if is_valid:
        logger.info("✅ Environment validation passed")
        env_info = get_environment_info()
        logger.info(f"Environment info: backend={env_info['backend']}, model={env_info['model']}, "
                   f"tts_voice={env_info['tts_voice']}")
        return True
    else:
        logger.error(f"❌ Environment validation failed: {error}")
        return False

if __name__ == "__main__":
    # Configure logging when run as script
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run validation
    if validate_on_startup():
        print("Environment validation passed")
        exit(0)
    else:
        print("Environment validation failed")
        exit(1)
