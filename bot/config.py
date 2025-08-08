"""Configuration loading and environment setup."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from .exceptions import ConfigurationError
from .util.logging import get_logger
import time
from typing import Dict, Any, Optional

# CHANGE: Enhanced .env loading with comprehensive audit and logging
logger = get_logger(__name__)

# Load environment variables from .env file with explicit path
load_dotenv(dotenv_path=Path.cwd() / '.env', verbose=True)

# Also try loading from the project root in case we're running from a subdirectory
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env', verbose=False)



def audit_env_file() -> None:
    """
    Audit .env file by reading and logging every variable.
    CHANGE: Enhanced comprehensive .env audit with critical environment variable verification.
    """
    env_file_path = Path.cwd() / '.env'
    if not env_file_path.exists():
        env_file_path = Path(__file__).parent.parent / '.env'
    
    logger.debug("=== STARTUP .ENV FILE AUDIT ===")
    if env_file_path.exists():
        logger.debug(f"Found .env file at: {env_file_path}")
        with open(env_file_path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    logger.debug(f".env:{line_no} → {line}")
        
        # CHANGE: Verify critical multimodal variables are loaded
        critical_vars = ['PROMPT_FILE', 'VL_PROMPT_FILE', 'VL_MODEL', 'OPENAI_TEXT_MODEL']
        logger.debug("=== CRITICAL VARIABLE VERIFICATION ===")
        for var in critical_vars:
            value = os.getenv(var)
            if value:
                logger.debug(f"✅ {var} = {value}")
            else:
                logger.error(f"❌ {var} is missing or empty!")
        
        logger.debug("=== END .ENV AUDIT ===")
    else:
        logger.error("❌ No .env file found for audit")
        raise ConfigurationError("No .env file found")


def validate_required_env() -> None:
    """
    Validate that all required environment variables are present.
    CHANGE: Enhanced validation to include PROMPT_FILE and VL_PROMPT_FILE.
    """
    required_vars = [
        "DISCORD_TOKEN",
        "PROMPT_FILE",
        "VL_PROMPT_FILE"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            logger.debug(f"✅ {var}: {value}")
    
    if missing_vars:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def validate_prompt_files() -> None:
    """
    Validate that prompt files exist and are readable.
    """
    prompt_file = os.getenv("PROMPT_FILE")
    vl_prompt_file = os.getenv("VL_PROMPT_FILE")

    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise ConfigurationError(f"PROMPT_FILE not found: {prompt_path}")
        logger.debug(f"✅ Text prompt file found: {prompt_path}")

    if vl_prompt_file:
        vl_prompt_path = Path(vl_prompt_file)
        if not vl_prompt_path.exists():
            raise ConfigurationError(f"VL_PROMPT_FILE not found: {vl_prompt_path}")
        logger.debug(f"✅ VL prompt file found: {vl_prompt_path}")


def load_system_prompts() -> dict[str, str]:
    """Loads system prompts from files specified in .env and returns them as a dictionary."""
    prompts = {}
    try:
        prompt_file = os.getenv("PROMPT_FILE", "prompts/prompt-yoroi-super-chill.txt")
        vl_prompt_file = os.getenv("VL_PROMPT_FILE", "prompts/vl-prompt.txt")

        prompts["text_prompt"] = Path(prompt_file).read_text()
        prompts["vl_prompt"] = Path(vl_prompt_file).read_text()

        logger.info(f"✅ Loaded system prompts: {list(prompts.keys())}")
        return prompts
    except FileNotFoundError as e:
        logger.error(f"❌ Critical error: Prompt file not found at {e.filename}. Please check your .env and file paths.")
        # This is a critical failure, so we exit.
        sys.exit(1)

def check_venv_activation() -> None:
    """
    Enforce exclusive .venv usage as specified in requirements.
    CHANGE: Added .venv enforcement check to ensure proper environment usage.
    """
    if ".venv" not in sys.prefix:
        logger.warning("⚠️  Running outside .venv—please activate .venv before using uv run")
        logger.warning(f"Current Python path: {sys.prefix}")
    else:
        logger.debug(f"✅ Running in .venv: {sys.prefix}")


def _safe_int(value: str, default: str, var_name: str) -> int:
    """Safely convert environment variable to int, handling malformed values."""
    try:
        # Clean value by removing comments and whitespace
        clean_value = value.split('#')[0].strip() if value else default
        return int(clean_value)
    except (ValueError, AttributeError):
        print(f"Warning: Invalid {var_name} value '{value}', using default {default}")
        return int(default)


def _safe_float(value: str, default: str, var_name: str) -> float:
    """Safely convert environment variable to float, handling malformed values."""
    try:
        # Clean value by removing comments and whitespace
        clean_value = value.split('#')[0].strip() if value else default
        return float(clean_value)
    except (ValueError, AttributeError):
        print(f"Warning: Invalid {var_name} value '{value}', using default {default}")
        return float(default)


def _clean_env_value(value: str) -> str:
    """
    Clean environment variable value by removing inline comments.
    CHANGE: Added to handle .env files with inline comments.
    """
    if not value:
        return value
    # Split on # and take the first part, then strip whitespace
    return value.split('#')[0].strip()


# Global config cache for performance optimization
_config_cache: Optional[Dict[str, Any]] = None
_cache_timestamp: float = 0
CACHE_TTL = 300  # 5 minute cache TTL

def load_config():
    """
    Load configuration from environment variables with intelligent caching.
    """
    global _config_cache, _cache_timestamp
    
    # Check if we have a valid cached config (performance optimization)
    current_time = time.time()
    if _config_cache and (current_time - _cache_timestamp) < CACHE_TTL:
        return _config_cache
    
    def _safe_int(value, default, var_name):
        """Safely convert environment variable to int, handling malformed values."""
        try:
            # Clean value by removing comments and whitespace
            clean_value = value.split('#')[0].strip() if value else default
            return int(clean_value)
        except (ValueError, AttributeError):
            print(f"Warning: Invalid {var_name} value '{value}', using default {default}")
            return int(default)

    config = {
        # DISCORD BOT SETTINGS
        "DISCORD_TOKEN": os.getenv("DISCORD_TOKEN"),
        "TEXT_BACKEND": os.getenv("TEXT_BACKEND", "openai"),
        
        # OPENAI / OPENROUTER SETTINGS
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        "OPENAI_TEXT_MODEL": os.getenv("OPENAI_TEXT_MODEL"),
        "VL_MODEL": _clean_env_value(os.getenv("VL_MODEL")),  # CHANGE: Added VL_MODEL for vision-language processing
        
        # OLLAMA SETTINGS
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3"),
        "TEXT_MODEL": os.getenv("TEXT_MODEL"),  # CHANGE: Added TEXT_MODEL for Ollama
        
        # BOT BEHAVIOR / CONTEXT / MEMORY
        "TEMPERATURE": _safe_float(os.getenv("TEMPERATURE"), "0.7", "TEMPERATURE"),
        "TIMEOUT": _safe_float(os.getenv("TIMEOUT"), "120.0", "TIMEOUT"),
        "CHANGE_NICKNAME": os.getenv("CHANGE_NICKNAME", "False").lower() == "true",
        "MAX_CONVERSATION_LENGTH": _safe_int(os.getenv("MAX_CONVERSATION_LENGTH"), "1000", "MAX_CONVERSATION_LENGTH"),
        "MAX_TEXT_ATTACHMENT_SIZE": _safe_int(os.getenv("MAX_TEXT_ATTACHMENT_SIZE"), "20000", "MAX_TEXT_ATTACHMENT_SIZE"),
        "MAX_FILE_SIZE": _safe_int(os.getenv("MAX_FILE_SIZE"), "2097152", "MAX_FILE_SIZE"),  # 2 MB
        "MAX_ATTACHMENT_SIZE_MB": _safe_int(os.getenv("MAX_ATTACHMENT_SIZE_MB"), "25", "MAX_ATTACHMENT_SIZE_MB"),
        
        # PROMPT FILES - CRITICAL FOR MULTIMODAL FUNCTIONALITY
        "PROMPT_FILE": _clean_env_value(os.getenv("PROMPT_FILE")),  # CHANGE: Added PROMPT_FILE for text model prompts
        "VL_PROMPT_FILE": _clean_env_value(os.getenv("VL_PROMPT_FILE")),  # CHANGE: Added VL_PROMPT_FILE for vision prompts
        
        # STT SETTINGS
        "STT_ENGINE": os.getenv("STT_ENGINE", "faster-whisper"),
        "STT_FALLBACK": os.getenv("STT_FALLBACK", "whispercpp"),
        "WHISPER_MODEL_SIZE": os.getenv("WHISPER_MODEL_SIZE", "medium-int8"),
        "WHISPER_CPP_MODEL": os.getenv("WHISPER_CPP_MODEL", "ggml-medium.bin"),
        
        # WHISPER SETTINGS
        "WHISPER_API_KEY": os.getenv("WHISPER_API_KEY"),
        "WHISPER_API_BASE": os.getenv("WHISPER_API_BASE"),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL", "whisper-1"),
        
        # MEMORY SETTINGS
        "MAX_USER_MEMORY": _safe_int(os.getenv("MAX_USER_MEMORY"), "20", "MAX_USER_MEMORY"),
        "MAX_SERVER_MEMORY": _safe_int(os.getenv("MAX_SERVER_MEMORY"), "100", "MAX_SERVER_MEMORY"),
        "MEMORY_SAVE_INTERVAL": _safe_int(os.getenv("MEMORY_SAVE_INTERVAL"), "30", "MEMORY_SAVE_INTERVAL"),
        "CONTEXT_FILE_PATH": os.getenv("CONTEXT_FILE_PATH", "runtime/context.json"),
        "MAX_CONTEXT_MESSAGES": _safe_int(os.getenv("MAX_CONTEXT_MESSAGES"), "10", "MAX_CONTEXT_MESSAGES"),
        
        # DIRECTORY SETTINGS
        "USER_PROFILE_DIR": Path(os.getenv("USER_PROFILE_DIR", "user_profiles")),
        "SERVER_PROFILE_DIR": Path(os.getenv("SERVER_PROFILE_DIR", "server_profiles")),
        "USER_LOGS_DIR": Path(os.getenv("USER_LOGS_DIR", "user_logs")),
        "DM_LOGS_DIR": Path(os.getenv("DM_LOGS_DIR", "dm_logs")),
        "TEMP_DIR": Path(os.getenv("TEMP_DIR", "temp")),
        "LOGS_DIR": Path(os.getenv("LOGS_DIR", "logs")),
        
        # TTS SETTINGS
        "TTS_BACKEND": os.getenv("TTS_BACKEND", "kokoro-onnx"),
        "TTS_VOICE": os.getenv("TTS_VOICE", "af"),
        
        # OPTIONAL SETTINGS
        "TTS_PREFS_FILE": os.getenv("TTS_PREFS_FILE"),
        "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
        "MAX_CONVERSATION_LOG_SIZE": _safe_int(os.getenv("MAX_CONVERSATION_LOG_SIZE"), "1000", "MAX_CONVERSATION_LOG_SIZE"),
        
        # LEGACY COMPATIBILITY
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4"),
        "MAX_MEMORIES": _safe_int(os.getenv("MAX_MEMORIES"), "100", "MAX_MEMORIES"),
        "DEFAULT_TIMEOUT": _safe_int(os.getenv("DEFAULT_TIMEOUT"), "30", "DEFAULT_TIMEOUT"),
        "MAX_CONTEXT_LENGTH": _safe_int(os.getenv("MAX_CONTEXT_LENGTH"), "4000", "MAX_CONTEXT_LENGTH"),
        "MAX_RESPONSE_TOKENS": _safe_int(os.getenv("MAX_RESPONSE_TOKENS"), "1000", "MAX_RESPONSE_TOKENS"),
        "TOP_P": _safe_float(os.getenv("TOP_P"), "0.9", "TOP_P"),
        "FREQUENCY_PENALTY": _safe_float(os.getenv("FREQUENCY_PENALTY"), "0.0", "FREQUENCY_PENALTY"),
        "PRESENCE_PENALTY": _safe_float(os.getenv("PRESENCE_PENALTY"), "0.0", "PRESENCE_PENALTY"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "COMMAND_PREFIX": os.getenv("COMMAND_PREFIX", "!"),
        "OWNER_IDS": [int(id.strip()) for id in os.getenv("OWNER_IDS", "").split(",") if id.strip()],
        "LOG_FILE": os.getenv("LOG_FILE", "logs/bot.jsonl"),
    }
    
    # Cache the config for performance (avoid repeated env var lookups)
    _config_cache = config
    _cache_timestamp = current_time
    logger.debug(f"✅ Configuration cached for {CACHE_TTL}s")
    
    return config
