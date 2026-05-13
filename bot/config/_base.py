"""Configuration loading and environment setup."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from ..exceptions import ConfigurationError
from ..utils.logging import get_logger
import time
from typing import Dict, Any, Optional

# CHANGE: Enhanced .env loading with comprehensive audit and logging
logger = get_logger(__name__)

# Load environment variables from .env file with explicit path
load_dotenv(dotenv_path=Path.cwd() / ".env", verbose=True)

# Also try loading from the project root in case we're running from a subdirectory
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", verbose=False)


def audit_env_file() -> None:
    """
    Audit .env file by reading and logging every variable.
    CHANGE: Enhanced comprehensive .env audit with critical environment variable verification.
    """
    env_file_path = Path.cwd() / ".env"
    if not env_file_path.exists():
        env_file_path = Path(__file__).parent.parent / ".env"

    logger.debug("=== STARTUP .ENV FILE AUDIT ===")
    if env_file_path.exists():
        logger.debug(f"Found .env file at: {env_file_path}")
        with open(env_file_path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    key = line.split("=")[0].strip() if "=" in line else "???"
                    logger.debug(f".env:{line_no} → {key}=[SET]")

        # CHANGE: Verify critical multimodal variables are loaded
        critical_vars = [
            "PROMPT_FILE",
            "VL_PROMPT_FILE",
            "VL_MODEL",
            "OPENAI_TEXT_MODEL",
        ]
        logger.debug("=== CRITICAL VARIABLE VERIFICATION ===")
        for var in critical_vars:
            value = os.getenv(var)
            if value:
                logger.debug(f"✅ {var} = [SET]")
            else:
                logger.error(f"❌ {var} is missing or empty!")

        logger.debug("=== END .ENV AUDIT ===")
    else:
        logger.warning("No .env file found for audit")
        return


def validate_required_env() -> None:
    """
    Validate that all required environment variables are present.
    CHANGE: Enhanced validation to include PROMPT_FILE and VL_PROMPT_FILE.
    """
    required_vars = ["DISCORD_TOKEN", "PROMPT_FILE", "VL_PROMPT_FILE"]

    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            logger.debug(f"✅ {var}: [SET]")

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
    """Loads system prompts from files specified in .env and returns them as a dictionary.

    Supports non-breaking synonyms:
    - TEXT_PROMPT_PATH ≙ PROMPT_FILE
    - VL_PROMPT_PATH   ≙ VL_PROMPT_FILE
    """
    prompts = {}
    try:
        # Prefer new PATH-style keys if present; fall back to existing FILE keys, then defaults
        prompt_file = os.getenv("TEXT_PROMPT_PATH") or os.getenv(
            "PROMPT_FILE", "prompts/prompt-yoroi-super-chill.txt"
        )
        vl_prompt_file = os.getenv("VL_PROMPT_PATH") or os.getenv(
            "VL_PROMPT_FILE", "prompts/vl-prompt.txt"
        )

        prompts["text_prompt"] = Path(prompt_file).read_text()
        prompts["vl_prompt"] = Path(vl_prompt_file).read_text()

        logger.info(f"✅ Loaded system prompts: {list(prompts.keys())}")
        return prompts
    except FileNotFoundError as e:
        logger.warning(
            f"⚠️ Prompt file not found at {e.filename}; using minimal fallback prompts for startup."
        )
        prompts.setdefault("text_prompt", "You are a helpful assistant.")
        prompts.setdefault("vl_prompt", "Describe the image succinctly.")
        return prompts


def check_venv_activation() -> None:
    """
    Enforce exclusive .venv usage as specified in requirements.
    CHANGE: Added .venv enforcement check to ensure proper environment usage.
    """
    if ".venv" not in sys.prefix:
        logger.warning(
            "⚠️  Running outside .venv—please activate .venv before using uv run"
        )
        logger.warning(f"Current Python path: {sys.prefix}")
    else:
        logger.debug(f"✅ Running in .venv: {sys.prefix}")


def _safe_int(value: str, default: str, var_name: str) -> int:
    """Safely convert environment variable to int, handling malformed values."""
    try:
        # Clean value by removing comments and whitespace
        clean_value = value.split("#")[0].strip() if value else default
        return int(clean_value)
    except (ValueError, AttributeError):
        logger.warning("Invalid %s value '%s', using default %s", var_name, value, default)
        return int(default)


def _safe_float(value: str, default: str, var_name: str) -> float:
    """Safely convert environment variable to float, handling malformed values."""
    try:
        # Clean value by removing comments and whitespace
        clean_value = value.split("#")[0].strip() if value else default
        return float(clean_value)
    except (ValueError, AttributeError):
        logger.warning("Invalid %s value '%s', using default %s", var_name, value, default)
        return float(default)


def _clean_env_value(value: str) -> str:
    """
    Clean environment variable value by removing inline comments.
    CHANGE: Added to handle .env files with inline comments.
    """
    if not value:
        return value
    # Split on # and take the first part, then strip whitespace
    return value.split("#")[0].strip()


# Robust boolean parsing to avoid bool("off") traps [IV][CMV]
def _parse_bool_str(raw: Optional[str], default: bool) -> bool:
    """Parse truthy/falsey strings with explicit tokens. ENV > default."""
    if raw is None:
        return default
    s = str(raw).strip().lower()
    true_tokens = {"1", "true", "yes", "on", "enabled", "enable"}
    false_tokens = {"0", "false", "no", "off", "disabled", "disable"}
    if s in true_tokens:
        return True
    if s in false_tokens:
        return False
    return default


# Global config cache for performance optimization
_config_cache: Optional[Dict[str, Any]] = None
_cache_timestamp: float = 0
CACHE_TTL = 300  # 5 minute cache TTL


def _parse_model_list(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


_DEFAULT_VL_MODEL_LADDER = [
    "moonshotai/kimi-vl-a3b-thinking:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
]


def load_config():
    """
    Load configuration from environment variables with intelligent caching.
    """
    global _config_cache, _cache_timestamp

    # Check if we have a valid cached config (performance optimization)
    current_time = time.time()
    if _config_cache and (current_time - _cache_timestamp) < CACHE_TTL:
        return _config_cache

    # Centralized VISION flags with robust parsing and defaults [CA]
    _ve_raw = _clean_env_value(os.getenv("VISION_ENABLED"))
    _t2i_raw = _clean_env_value(os.getenv("VISION_T2I_ENABLED"))
    _ve = _parse_bool_str(_ve_raw, True)  # default ON when unset
    _t2i = _parse_bool_str(_t2i_raw, True)  # default ON when unset

    config = {
        # DISCORD BOT SETTINGS
        "DISCORD_TOKEN": os.getenv("DISCORD_TOKEN"),
        "TEXT_BACKEND": os.getenv("TEXT_BACKEND", "openai"),
        # OPENAI / OPENROUTER SETTINGS
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        "OPENAI_TEXT_MODEL": os.getenv("OPENAI_TEXT_MODEL"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "TEXT_FALLBACK_MODELS": os.getenv("TEXT_FALLBACK_MODELS"),
        "TEXT_FALLBACK_TIMEOUTS": os.getenv("TEXT_FALLBACK_TIMEOUTS"),
        "TEXT_FALLBACK_MAX_ATTEMPTS": os.getenv("TEXT_FALLBACK_MAX_ATTEMPTS"),
        "NVIDIA_NIM_API_KEY": os.getenv("NVIDIA_NIM_API_KEY"),
        "NVIDIA_NIM_API_BASE": os.getenv("NVIDIA_NIM_API_BASE"),
        "NVIDIA_NIM_TEXT_MODEL": os.getenv("NVIDIA_NIM_TEXT_MODEL"),
        "VL_MODEL": _clean_env_value(
            os.getenv("VL_MODEL")
        ),  # CHANGE: Added VL_MODEL for vision-language processing
        # OLLAMA SETTINGS
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3"),
        "TEXT_MODEL": os.getenv("TEXT_MODEL"),  # CHANGE: Added TEXT_MODEL for Ollama
        # BOT BEHAVIOR / CONTEXT / MEMORY
        "TEMPERATURE": _safe_float(os.getenv("TEMPERATURE"), "0.7", "TEMPERATURE"),
        "TIMEOUT": _safe_float(os.getenv("TIMEOUT"), "120.0", "TIMEOUT"),
        "CHANGE_NICKNAME": _parse_bool_str(
            _clean_env_value(os.getenv("CHANGE_NICKNAME")), False
        ),
        "MAX_CONVERSATION_LENGTH": _safe_int(
            os.getenv("MAX_CONVERSATION_LENGTH"), "1000", "MAX_CONVERSATION_LENGTH"
        ),
        "MAX_TEXT_ATTACHMENT_SIZE": _safe_int(
            os.getenv("MAX_TEXT_ATTACHMENT_SIZE"), "20000", "MAX_TEXT_ATTACHMENT_SIZE"
        ),
        "MAX_FILE_SIZE": _safe_int(
            os.getenv("MAX_FILE_SIZE"), "2097152", "MAX_FILE_SIZE"
        ),  # 2 MB
        "MAX_ATTACHMENT_SIZE_MB": _safe_int(
            os.getenv("MAX_ATTACHMENT_SIZE_MB"), "25", "MAX_ATTACHMENT_SIZE_MB"
        ),
        # SILENCE GATE - SPEAK ONLY WHEN SPOKEN TO
        "BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO": _parse_bool_str(
            _clean_env_value(os.getenv("BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO")), True
        ),
        "REQUIRE_MENTION_IN_GUILDS": _parse_bool_str(
            _clean_env_value(os.getenv("REQUIRE_MENTION_IN_GUILDS")), True
        ),
        "ALLOW_REPLY_TO_BOT_WITHOUT_MENTION": _parse_bool_str(
            _clean_env_value(os.getenv("ALLOW_REPLY_TO_BOT_WITHOUT_MENTION")), True
        ),
        "DM_REQUIRE_MENTION": _parse_bool_str(
            _clean_env_value(os.getenv("DM_REQUIRE_MENTION")), False
        ),
        # Comma-separated list of triggers: dm, mention, reply, bot_threads, owner, command_prefix
        "REPLY_TRIGGERS": [
            s.strip()
            for s in os.getenv(
                "REPLY_TRIGGERS", "dm,mention,reply,bot_threads,owner,command_prefix"
            ).split(",")
            if s.strip()
        ],
        # PROMPT FILES - CRITICAL FOR MULTIMODAL FUNCTIONALITY
        "PROMPT_FILE": _clean_env_value(
            os.getenv("PROMPT_FILE")
        ),  # CHANGE: Added PROMPT_FILE for text model prompts
        "VL_PROMPT_FILE": _clean_env_value(
            os.getenv("VL_PROMPT_FILE")
        ),  # CHANGE: Added VL_PROMPT_FILE for vision prompts
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
        "MAX_USER_MEMORY": _safe_int(
            os.getenv("MAX_USER_MEMORY"), "20", "MAX_USER_MEMORY"
        ),
        "MAX_SERVER_MEMORY": _safe_int(
            os.getenv("MAX_SERVER_MEMORY"), "100", "MAX_SERVER_MEMORY"
        ),
        "MEMORY_SAVE_INTERVAL": _safe_int(
            os.getenv("MEMORY_SAVE_INTERVAL"), "30", "MEMORY_SAVE_INTERVAL"
        ),
        "PERSISTENT_MEMORY_ENABLE": _parse_bool_str(
            os.getenv("PERSISTENT_MEMORY_ENABLE"), True
        ),
        "PERSISTENT_MEMORY_SQLITE_PATH": os.getenv(
            "PERSISTENT_MEMORY_SQLITE_PATH", "./data/memory.db"
        ),
        "PERSISTENT_MEMORY_CHROMA_PATH": os.getenv(
            "PERSISTENT_MEMORY_CHROMA_PATH", "./chroma_db"
        ),
        "PERSISTENT_MEMORY_CHROMA_COLLECTION": os.getenv(
            "PERSISTENT_MEMORY_CHROMA_COLLECTION", "curated_memories"
        ),
        "PERSISTENT_MEMORY_QUEUE_MAX": _safe_int(
            os.getenv("PERSISTENT_MEMORY_QUEUE_MAX"),
            "256",
            "PERSISTENT_MEMORY_QUEUE_MAX",
        ),
        "PERSISTENT_MEMORY_WORKERS": _low_resource_int(
            "PERSISTENT_MEMORY_WORKERS", 1, 1
        ),
        "PERSISTENT_MEMORY_TOP_K": _low_resource_int(
            "PERSISTENT_MEMORY_TOP_K",
            6,
            3,
        ),
        "PERSISTENT_MEMORY_MAX_PROMPT_CHARS": _safe_int(
            os.getenv("PERSISTENT_MEMORY_MAX_PROMPT_CHARS"),
            "1200",
            "PERSISTENT_MEMORY_MAX_PROMPT_CHARS",
        ),
        # Memory semantic dedupe & resource caps [Phase 6-9]
        "MEMORY_SEMANTIC_DEDUPE_ENABLED": _low_resource_bool(
            "MEMORY_SEMANTIC_DEDUPE_ENABLED", True, True
        ),
        "MEMORY_MAX_TEXT_CHARS": _low_resource_int(
            "MEMORY_MAX_TEXT_CHARS", 500, 300
        ),
        "MEMORY_SEMANTIC_TOP_K": _low_resource_int(
            "MEMORY_SEMANTIC_TOP_K", 5, 2
        ),
        "MEMORY_INGEST_WORKERS": _low_resource_int(
            "MEMORY_INGEST_WORKERS", 2, 1
        ),
        "MEMORY_RECALL_CACHE_TTL_S": _low_resource_int(
            "MEMORY_RECALL_CACHE_TTL_S", 30, 15
        ),
        "MEMORY_AUTO_CURATION_ENABLED": _low_resource_bool(
            "MEMORY_AUTO_CURATION_ENABLED", True, False
        ),
        "MEMORY_DISTILL_INTERVAL_S": _low_resource_int(
            "MEMORY_DISTILL_INTERVAL_S", 900, 3600
        ),
        "CHROMADB_MAX_RESULTS": _low_resource_int(
            "CHROMADB_MAX_RESULTS", 5, 3
        ),
        "PERSISTENT_MEMORY_DEFAULT_TTL_DAYS": _safe_int(
            os.getenv("PERSISTENT_MEMORY_DEFAULT_TTL_DAYS"),
            "180",
            "PERSISTENT_MEMORY_DEFAULT_TTL_DAYS",
        ),
        "PERSISTENT_MEMORY_TEMP_TTL_DAYS": _safe_int(
            os.getenv("PERSISTENT_MEMORY_TEMP_TTL_DAYS"),
            "14",
            "PERSISTENT_MEMORY_TEMP_TTL_DAYS",
        ),
        "PERSISTENT_MEMORY_MIN_IMPORTANCE": _safe_float(
            os.getenv("PERSISTENT_MEMORY_MIN_IMPORTANCE"),
            "0.55",
            "PERSISTENT_MEMORY_MIN_IMPORTANCE",
        ),
        # SERVER ARCHIVE SETTINGS
        # Support both the newer *_ENABLED / *_ARCHIVE_BOT_MESSAGES keys and the
        # original aliases for backwards compatibility.
        "SERVER_ARCHIVE_ENABLED": _parse_bool_str(
            _clean_env_value(
                os.getenv("SERVER_ARCHIVE_ENABLED")
                or os.getenv("SERVER_ARCHIVE_ENABLE")
            ),
            False,
        ),
        "SERVER_ARCHIVE_ENABLE": _parse_bool_str(
            _clean_env_value(
                os.getenv("SERVER_ARCHIVE_ENABLED")
                or os.getenv("SERVER_ARCHIVE_ENABLE")
            ),
            False,
        ),
        "SERVER_ARCHIVE_DB_PATH": os.getenv(
            "SERVER_ARCHIVE_DB_PATH", "./data/server_archive.db"
        ),
        "SERVER_ARCHIVE_QUEUE_MAX": _low_resource_int(
            "SERVER_ARCHIVE_QUEUE_MAX", 1000, 200
        ),
        "SERVER_ARCHIVE_BATCH_SIZE": _low_resource_int(
            "SERVER_ARCHIVE_BATCH_SIZE", 100, 20
        ),
        "SERVER_ARCHIVE_DISTILL_INTERVAL_S": _low_resource_int(
            "SERVER_ARCHIVE_DISTILL_INTERVAL_S", 900, 3600
        ),
        "SERVER_ARCHIVE_SEARCH_LIMIT": _safe_int(
            os.getenv("SERVER_ARCHIVE_SEARCH_LIMIT"),
            "10",
            "SERVER_ARCHIVE_SEARCH_LIMIT",
        ),
        "SERVER_ARCHIVE_ADMIN_ONLY": _parse_bool_str(
            _clean_env_value(os.getenv("SERVER_ARCHIVE_ADMIN_ONLY")), True
        ),
        "SERVER_ARCHIVE_SYNC_ON_START": _parse_bool_str(
            _clean_env_value(os.getenv("SERVER_ARCHIVE_SYNC_ON_START")), True
        ),
        "SERVER_ARCHIVE_LIVE_TAIL": _parse_bool_str(
            _clean_env_value(os.getenv("SERVER_ARCHIVE_LIVE_TAIL")), True
        ),
        "SERVER_ARCHIVE_MAX_MESSAGE_CHARS": _safe_int(
            os.getenv("SERVER_ARCHIVE_MAX_MESSAGE_CHARS"),
            "8000",
            "SERVER_ARCHIVE_MAX_MESSAGE_CHARS",
        ),
        "SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES": _parse_bool_str(
            _clean_env_value(
                os.getenv("SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES")
                or os.getenv("SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES")
            ),
            False,
        ),
        "SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES": _parse_bool_str(
            _clean_env_value(
                os.getenv("SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES")
                or os.getenv("SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES")
            ),
            False,
        ),
        # MEMORY DISTILLER SETTINGS
        "MEMORY_DISTILLER_ENABLED": _parse_bool_str(
            _clean_env_value(os.getenv("MEMORY_DISTILLER_ENABLED")),
            False,
        ),
        "MEMORY_DISTILLER_DRY_RUN": _parse_bool_str(
            _clean_env_value(os.getenv("MEMORY_DISTILLER_DRY_RUN")),
            True,
        ),
        "MEMORY_DISTILLER_BATCH_SIZE": _safe_int(
            os.getenv("MEMORY_DISTILLER_BATCH_SIZE"),
            "200",
            "MEMORY_DISTILLER_BATCH_SIZE",
        ),
        "MEMORY_DISTILLER_INTERVAL_SECONDS": _low_resource_int(
            "MEMORY_DISTILLER_INTERVAL_SECONDS",
            900,
            3600,
        ),
        "MEMORY_DISTILLER_WINDOW_MESSAGES": _safe_int(
            os.getenv("MEMORY_DISTILLER_WINDOW_MESSAGES"),
            "25",
            "MEMORY_DISTILLER_WINDOW_MESSAGES",
        ),
        "MEMORY_DISTILLER_MIN_CONFIDENCE": _safe_float(
            os.getenv("MEMORY_DISTILLER_MIN_CONFIDENCE"),
            "0.85",
            "MEMORY_DISTILLER_MIN_CONFIDENCE",
        ),
        "MEMORY_DISTILLER_MAX_MEMORIES_PER_WINDOW": _safe_int(
            os.getenv("MEMORY_DISTILLER_MAX_MEMORIES_PER_WINDOW"),
            "3",
            "MEMORY_DISTILLER_MAX_MEMORIES_PER_WINDOW",
        ),
        "MEMORY_DISTILLER_EXCLUDE_BOT_MESSAGES": _parse_bool_str(
            _clean_env_value(os.getenv("MEMORY_DISTILLER_EXCLUDE_BOT_MESSAGES")),
            True,
        ),
        "CONTEXT_FILE_PATH": os.getenv("CONTEXT_FILE_PATH", "runtime/context.json"),
        "MAX_CONTEXT_MESSAGES": _safe_int(
            os.getenv("MAX_CONTEXT_MESSAGES"), "10", "MAX_CONTEXT_MESSAGES"
        ),
        "MEM_MAX_MSGS": _safe_int(os.getenv("MEM_MAX_MSGS"), "40", "MEM_MAX_MSGS"),
        "MEM_MAX_CHARS": _safe_int(os.getenv("MEM_MAX_CHARS"), "8000", "MEM_MAX_CHARS"),
        "MEM_MAX_AGE_MIN": _safe_int(
            os.getenv("MEM_MAX_AGE_MIN"), "240", "MEM_MAX_AGE_MIN"
        ),
        "MEM_FETCH_TIMEOUT_S": _safe_float(
            os.getenv("MEM_FETCH_TIMEOUT_S"), "5", "MEM_FETCH_TIMEOUT_S"
        ),
        "MEM_LOG_SUBSYS": os.getenv("MEM_LOG_SUBSYS", "mem.ctx"),
        # Thread-tail reply + context collector [CMV][REH]
        "THREAD_CONTEXT_TAIL_COUNT": _safe_int(
            os.getenv("THREAD_CONTEXT_TAIL_COUNT"), "5", "THREAD_CONTEXT_TAIL_COUNT"
        ),
        # Context trimming — character and message limits
        "CONTEXT_MAX_MESSAGES": CONTEXT_MAX_MESSAGES,
        "CONTEXT_MAX_CHARS_PER_MESSAGE": CONTEXT_MAX_CHARS_PER_MESSAGE,
        "CONTEXT_MAX_TOTAL_CHARS": CONTEXT_MAX_TOTAL_CHARS,
        "CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS": CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS,
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
        # Timeouts for TTS synthesis [CMV][IV]
        # Defaults preserve legacy behavior; cold/warm may be overridden via env.
        "TTS_TIMEOUT_S": _safe_float(
            os.getenv("TTS_TIMEOUT_S"), "25.0", "TTS_TIMEOUT_S"
        ),
        "TTS_TIMEOUT_COLD_S": _safe_float(
            os.getenv("TTS_TIMEOUT_COLD_S") or os.getenv("TTS_TIMEOUT_S"),
            "25.0",
            "TTS_TIMEOUT_COLD_S",
        ),
        "TTS_TIMEOUT_WARM_S": _safe_float(
            os.getenv("TTS_TIMEOUT_WARM_S") or os.getenv("TTS_TIMEOUT_S"),
            "25.0",
            "TTS_TIMEOUT_WARM_S",
        ),
        # Native Discord voice messages toggle [CMV]
        "VOICE_ENABLE_NATIVE": _parse_bool_str(
            _clean_env_value(os.getenv("VOICE_ENABLE_NATIVE")), False
        ),
        # Voice Publisher HTTP timeouts (aiohttp) [CMV][IV]
        # Global fallback if specific values are unset
        "VOICE_PUBLISHER_TIMEOUT_S": _safe_float(
            os.getenv("VOICE_PUBLISHER_TIMEOUT_S"), "0.0", "VOICE_PUBLISHER_TIMEOUT_S"
        ),
        "VOICE_PUBLISHER_ATTACHMENTS_CREATE_TIMEOUT_S": _safe_float(
            os.getenv("VOICE_PUBLISHER_ATTACHMENTS_CREATE_TIMEOUT_S"),
            "30.0",
            "VOICE_PUBLISHER_ATTACHMENTS_CREATE_TIMEOUT_S",
        ),
        "VOICE_PUBLISHER_UPLOAD_TIMEOUT_S": _safe_float(
            os.getenv("VOICE_PUBLISHER_UPLOAD_TIMEOUT_S"),
            "60.0",
            "VOICE_PUBLISHER_UPLOAD_TIMEOUT_S",
        ),
        "VOICE_PUBLISHER_MESSAGE_POST_TIMEOUT_S": _safe_float(
            os.getenv("VOICE_PUBLISHER_MESSAGE_POST_TIMEOUT_S"),
            "30.0",
            "VOICE_PUBLISHER_MESSAGE_POST_TIMEOUT_S",
        ),
        # Opus encoding parameters for native voice messages [CMV]
        "VOICE_PUBLISHER_OPUS_BITRATE": os.getenv(
            "VOICE_PUBLISHER_OPUS_BITRATE", "64k"
        ),
        "VOICE_PUBLISHER_OPUS_VBR": os.getenv("VOICE_PUBLISHER_OPUS_VBR", "on"),
        "VOICE_PUBLISHER_OPUS_COMP_LEVEL": _safe_int(
            os.getenv("VOICE_PUBLISHER_OPUS_COMP_LEVEL"),
            "10",
            "VOICE_PUBLISHER_OPUS_COMP_LEVEL",
        ),
        # OPTIONAL SETTINGS
        "TTS_PREFS_FILE": os.getenv("TTS_PREFS_FILE"),
        "DEBUG": _parse_bool_str(_clean_env_value(os.getenv("DEBUG")), False),
        "MAX_CONVERSATION_LOG_SIZE": _safe_int(
            os.getenv("MAX_CONVERSATION_LOG_SIZE"), "1000", "MAX_CONVERSATION_LOG_SIZE"
        ),
        # LEGACY COMPATIBILITY
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4"),
        "MAX_MEMORIES": _safe_int(
            os.getenv("MAX_MEMORIES") or os.getenv("MAX_USER_MEMORY"),
            "100",
            "MAX_MEMORIES",
        ),
        "DEFAULT_TIMEOUT": _safe_int(
            os.getenv("DEFAULT_TIMEOUT"), "30", "DEFAULT_TIMEOUT"
        ),
        "MAX_CONTEXT_LENGTH": _safe_int(
            os.getenv("MAX_CONTEXT_LENGTH"), "4000", "MAX_CONTEXT_LENGTH"
        ),
        "MAX_RESPONSE_TOKENS": _safe_int(
            os.getenv("MAX_RESPONSE_TOKENS"), "1000", "MAX_RESPONSE_TOKENS"
        ),
        "TOP_P": _safe_float(os.getenv("TOP_P"), "0.9", "TOP_P"),
        "FREQUENCY_PENALTY": _safe_float(
            os.getenv("FREQUENCY_PENALTY"), "0.0", "FREQUENCY_PENALTY"
        ),
        "PRESENCE_PENALTY": _safe_float(
            os.getenv("PRESENCE_PENALTY"), "0.0", "PRESENCE_PENALTY"
        ),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "COMMAND_PREFIX": os.getenv("COMMAND_PREFIX", "!"),
        "OWNER_IDS": [
            int(id.strip())
            for id in os.getenv("OWNER_IDS", "").split(",")
            if id.strip()
        ],
        "LOG_FILE": os.getenv("LOG_FILE", "logs/bot.jsonl"),
        # ADMIN ALERT SYSTEM [CA][CMV]
        # Keep as strings to preserve existing cog parsing semantics
        # - ALERT_ENABLE checked via .lower() == 'true'
        # - ALERT_SESSION_TIMEOUT_S cast to int in cog
        # - ALERT_ADMIN_USER_IDS parsed as comma-separated ints in cog
        "ALERT_ENABLE": _clean_env_value(os.getenv("ALERT_ENABLE", "false")),
        "ALERT_SESSION_TIMEOUT_S": _clean_env_value(
            os.getenv("ALERT_SESSION_TIMEOUT_S", "1800")
        ),
        "ALERT_ADMIN_USER_IDS": _clean_env_value(os.getenv("ALERT_ADMIN_USER_IDS", "")),
        # SEARCH SUBSYSTEM [CA][CMV][IV]
        # Provider selection: 'ddg' (default) or 'custom'
        "SEARCH_PROVIDER": os.getenv("SEARCH_PROVIDER", "ddg").lower(),
        # Default knobs
        "SEARCH_MAX_RESULTS": _safe_int(
            os.getenv("SEARCH_MAX_RESULTS"), "5", "SEARCH_MAX_RESULTS"
        ),
        "SEARCH_SAFE": os.getenv(
            "SEARCH_SAFE", "moderate"
        ).lower(),  # off|moderate|strict
        "SEARCH_LOCALE": os.getenv("SEARCH_LOCALE", ""),
        # DuckDuckGo provider options (DDG typically requires no API key; kept for pluggability)
        "DDG_API_ENDPOINT": os.getenv(
            "DDG_API_ENDPOINT", "https://html.duckduckgo.com/html/"
        ),
        # Force legacy HTML endpoint instead of ddgs client. [CMV]
        "DDG_FORCE_HTML": _parse_bool_str(
            _clean_env_value(os.getenv("DDG_FORCE_HTML")), True
        ),
        "DDG_API_KEY": os.getenv("DDG_API_KEY"),
        "DDG_TIMEOUT_MS": _safe_int(
            os.getenv("DDG_TIMEOUT_MS"), "5000", "DDG_TIMEOUT_MS"
        ),
        # Custom provider HTTP options
        "CUSTOM_SEARCH_API_ENDPOINT": os.getenv("CUSTOM_SEARCH_API_ENDPOINT", ""),
        "CUSTOM_SEARCH_API_KEY": os.getenv("CUSTOM_SEARCH_API_KEY", ""),
        # Optional JSON headers, comma-separated key:value pairs
        "CUSTOM_SEARCH_HEADERS": os.getenv("CUSTOM_SEARCH_HEADERS", ""),
        "CUSTOM_SEARCH_TIMEOUT_MS": _safe_int(
            os.getenv("CUSTOM_SEARCH_TIMEOUT_MS"), "8000", "CUSTOM_SEARCH_TIMEOUT_MS"
        ),
        # Optional JSONPath-like comma-separated selectors for result extraction
        "CUSTOM_SEARCH_RESULT_PATHS": os.getenv("CUSTOM_SEARCH_RESULT_PATHS", ""),
        # Shared HTTP pool
        "SEARCH_POOL_MAX_CONNECTIONS": _low_resource_int(
            "SEARCH_POOL_MAX_CONNECTIONS",
            10,
            3,
        ),
        # Circuit breaker (search)
        "SEARCH_BREAKER_FAILURE_WINDOW": _safe_int(
            os.getenv("SEARCH_BREAKER_FAILURE_WINDOW"),
            "5",
            "SEARCH_BREAKER_FAILURE_WINDOW",
        ),
        "SEARCH_BREAKER_OPEN_MS": _safe_int(
            os.getenv("SEARCH_BREAKER_OPEN_MS"), "15000", "SEARCH_BREAKER_OPEN_MS"
        ),
        "SEARCH_BREAKER_HALFOPEN_PROB": _safe_float(
            os.getenv("SEARCH_BREAKER_HALFOPEN_PROB"),
            "0.25",
            "SEARCH_BREAKER_HALFOPEN_PROB",
        ),
        # X (Twitter) API Integration [CA][CMV][SFT]
        # Feature flag and auth
        "X_API_ENABLED": _parse_bool_str(
            _clean_env_value(os.getenv("X_API_ENABLED")), False
        ),
        "X_API_AUTH_MODE": os.getenv("X_API_AUTH_MODE", "oauth2_app"),
        "X_API_BEARER_TOKEN": _clean_env_value(
            os.getenv("X_API_BEARER_TOKEN")
        ),  # never log token
        # Fallback rules
        "X_API_REQUIRE_API_FOR_TWITTER": _parse_bool_str(
            _clean_env_value(os.getenv("X_API_REQUIRE_API_FOR_TWITTER")), False
        ),
        "X_API_ALLOW_FALLBACK_ON_5XX": _parse_bool_str(
            _clean_env_value(os.getenv("X_API_ALLOW_FALLBACK_ON_5XX")), True
        ),
        # X Syndication Tier [CMV]
        # Hardcoded default: enabled unless explicitly disabled
        "X_SYNDICATION_ENABLED": _parse_bool_str(
            _clean_env_value(os.getenv("X_SYNDICATION_ENABLED")), True
        ),
        # Fast probe: attempt STT on X URLs before API/syndication [CMV][PA]
        "X_TWITTER_STT_PROBE_FIRST": _parse_bool_str(
            _clean_env_value(os.getenv("X_TWITTER_STT_PROBE_FIRST")), True
        ),
        # Routing: enable photo media to VL analysis path [CMV]
        # Hardcoded default: enabled (route photos to VL)
        "X_API_ROUTE_PHOTOS_TO_VL": _parse_bool_str(
            _clean_env_value(os.getenv("X_API_ROUTE_PHOTOS_TO_VL")), True
        ),
        # Networking and resilience knobs
        "X_API_TIMEOUT_MS": _safe_int(
            os.getenv("X_API_TIMEOUT_MS"), "8000", "X_API_TIMEOUT_MS"
        ),
        "X_API_RETRY_MAX_ATTEMPTS": _safe_int(
            os.getenv("X_API_RETRY_MAX_ATTEMPTS"), "5", "X_API_RETRY_MAX_ATTEMPTS"
        ),
        "X_API_BREAKER_FAILURE_WINDOW": _safe_int(
            os.getenv("X_API_BREAKER_FAILURE_WINDOW"),
            "5",
            "X_API_BREAKER_FAILURE_WINDOW",
        ),
        "X_API_BREAKER_OPEN_MS": _safe_int(
            os.getenv("X_API_BREAKER_OPEN_MS"), "15000", "X_API_BREAKER_OPEN_MS"
        ),
        "X_API_BREAKER_HALFOPEN_PROB": _safe_float(
            os.getenv("X_API_BREAKER_HALFOPEN_PROB"),
            "0.25",
            "X_API_BREAKER_HALFOPEN_PROB",
        ),
        # Field hydration (comma-separated lists) [CMV]
        "X_TWEET_FIELDS": [
            s.strip()
            for s in os.getenv(
                "X_TWEET_FIELDS",
                "id,text,created_at,author_id,public_metrics,possibly_sensitive,lang,attachments,entities,referenced_tweets,conversation_id",
            ).split(",")
            if s.strip()
        ],
        "X_EXPANSIONS": [
            s.strip()
            for s in os.getenv(
                "X_EXPANSIONS",
                "author_id,attachments.media_keys,referenced_tweets.id,referenced_tweets.id.author_id",
            ).split(",")
            if s.strip()
        ],
        "X_MEDIA_FIELDS": [
            s.strip()
            for s in os.getenv(
                "X_MEDIA_FIELDS",
                "media_key,type,url,preview_image_url,variants,width,height,alt_text,public_metrics",
            ).split(",")
            if s.strip()
        ],
        "X_USER_FIELDS": [
            s.strip()
            for s in os.getenv(
                "X_USER_FIELDS", "id,name,username,profile_image_url,verified,protected"
            ).split(",")
            if s.strip()
        ],
        "X_POLL_FIELDS": [
            s.strip()
            for s in os.getenv(
                "X_POLL_FIELDS",
                "id,options,duration_minutes,end_datetime,voting_status",
            ).split(",")
            if s.strip()
        ],
        "X_PLACE_FIELDS": [
            s.strip()
            for s in os.getenv(
                "X_PLACE_FIELDS", "full_name,id,country_code,geo,name,place_type"
            ).split(",")
            if s.strip()
        ],
        # Twitter/X thread unroll feature flags [CMV]
        "TWITTER_UNROLL_ENABLED": _parse_bool_str(
            _clean_env_value(os.getenv("TWITTER_UNROLL_ENABLED")), True
        ),
        "TWITTER_UNROLL_MAX_TWEETS": _safe_int(
            os.getenv("TWITTER_UNROLL_MAX_TWEETS"), "30", "TWITTER_UNROLL_MAX_TWEETS"
        ),
        "TWITTER_UNROLL_MAX_CHARS": _safe_int(
            os.getenv("TWITTER_UNROLL_MAX_CHARS"), "6000", "TWITTER_UNROLL_MAX_CHARS"
        ),
        "TWITTER_UNROLL_TIMEOUT_S": _safe_float(
            os.getenv("TWITTER_UNROLL_TIMEOUT_S"), "15", "TWITTER_UNROLL_TIMEOUT_S"
        ),
        # Routing defaults
        "TWITTER_ROUTE_DEFAULT": os.getenv("TWITTER_ROUTE_DEFAULT", "api_first"),
        # STREAMING STATUS CARDS [CA][CMV]
        # Global enable for streaming card UX (text-only remains non-streaming)
        "STREAMING_ENABLE": _parse_bool_str(
            _clean_env_value(os.getenv("STREAMING_ENABLE")), True
        ),
        # Style preset: 'compact' | 'detailed'
        "STREAMING_EMBED_STYLE": os.getenv("STREAMING_EMBED_STYLE", "compact"),
        # Edit throttle and max step count
        "STREAMING_TICK_MS": _safe_int(
            os.getenv("STREAMING_TICK_MS"), "750", "STREAMING_TICK_MS"
        ),
        "STREAMING_MAX_STEPS": _safe_int(
            os.getenv("STREAMING_MAX_STEPS"), "8", "STREAMING_MAX_STEPS"
        ),
        # Domain-specific eligibility gates [CMV]
        # Defaults: text/search/rag disabled, media enabled
        "STREAMING_ENABLE_TEXT": _parse_bool_str(
            _clean_env_value(os.getenv("STREAMING_ENABLE_TEXT")), False
        ),
        "STREAMING_ENABLE_SEARCH": _parse_bool_str(
            _clean_env_value(os.getenv("STREAMING_ENABLE_SEARCH")), False
        ),
        "STREAMING_ENABLE_RAG": _parse_bool_str(
            _clean_env_value(os.getenv("STREAMING_ENABLE_RAG")), False
        ),
        "STREAMING_ENABLE_MEDIA": _parse_bool_str(
            _clean_env_value(os.getenv("STREAMING_ENABLE_MEDIA")), True
        ),
        # STT ORCHESTRATION [CA][CMV] =====
        # Global toggle for STT orchestrator (falls back to legacy path when disabled)
        "STT_ENABLE": _parse_bool_str(_clean_env_value(os.getenv("STT_ENABLE")), True),
        # ===== VISION GENERATION SYSTEM [CA][CMV][SFT][REH] =====
        # Master toggles (parsed via robust tokens; default ON when unset)
        "VISION_ENABLED": _ve,
        "VISION_T2I_ENABLED": _t2i,
        # Reply-image VL toggles
        "VISION_REPLY_IMAGE_FORCE_VL": _parse_bool_str(
            _clean_env_value(os.getenv("VISION_REPLY_IMAGE_FORCE_VL")), True
        ),
        "VISION_REPLY_IMAGE_SILENT": _parse_bool_str(
            _clean_env_value(os.getenv("VISION_REPLY_IMAGE_SILENT")), True
        ),
        # Hybrid perception routing: VL feeds notes to TEXT
        "HYBRID_FORCE_PERCEPTION_ON_REPLY": _parse_bool_str(
            _clean_env_value(os.getenv("HYBRID_FORCE_PERCEPTION_ON_REPLY")), True
        ),
        # VL concise output knobs
        "VL_REPLY_MAX_CHARS": _safe_int(
            os.getenv("VL_REPLY_MAX_CHARS"), "420", "VL_REPLY_MAX_CHARS"
        ),
        "VL_STRIP_REASONING": _parse_bool_str(
            _clean_env_value(os.getenv("VL_STRIP_REASONING")), True
        ),
        # Perception notes and final text caps
        "VL_NOTES_MAX_CHARS": _safe_int(
            os.getenv("VL_NOTES_MAX_CHARS"), "600", "VL_NOTES_MAX_CHARS"
        ),
        "TEXT_FINAL_MAX_CHARS": _safe_int(
            os.getenv("TEXT_FINAL_MAX_CHARS"), "420", "TEXT_FINAL_MAX_CHARS"
        ),
        # Single credential for Vision Gateway (provider secrets handled behind gateway)
        "VISION_API_KEY": _clean_env_value(os.getenv("VISION_API_KEY")),
        # Provider configuration
        "VISION_ALLOWED_PROVIDERS": [
            s.strip()
            for s in os.getenv("VISION_ALLOWED_PROVIDERS", "together,novita").split(",")
            if s.strip()
        ],
        "VISION_DEFAULT_PROVIDER": os.getenv("VISION_DEFAULT_PROVIDER", "together"),
        "VISION_MODEL": _clean_env_value(os.getenv("VISION_MODEL")) or "",
        "VISION_IMAGE_FALLBACK_MODELS": _clean_env_value(
            os.getenv("VISION_IMAGE_FALLBACK_MODELS")
            or os.getenv("IMAGE_FALLBACK_MODELS")
        )
        or "",
        # Policy and data paths
        "VISION_POLICY_PATH": os.getenv(
            "VISION_POLICY_PATH", "configs/vision_policy.json"
        ),
        "VISION_DATA_DIR": Path(os.getenv("VISION_DATA_DIR", "vision_data")),
        "VISION_ARTIFACTS_DIR": Path(
            os.getenv("VISION_ARTIFACTS_DIR", "vision_data/artifacts")
        ),
        "VISION_JOBS_DIR": Path(os.getenv("VISION_JOBS_DIR", "vision_data/jobs")),
        "VISION_LEDGER_PATH": os.getenv(
            "VISION_LEDGER_PATH", "vision_data/ledger.jsonl"
        ),
        # Intent routing thresholds
        "VISION_INTENT_THRESHOLD": _safe_float(
            os.getenv("VISION_INTENT_THRESHOLD"), "0.7", "VISION_INTENT_THRESHOLD"
        ),
        "VISION_FORCE_OPENROUTER_THRESHOLD": _safe_float(
            os.getenv("VISION_FORCE_OPENROUTER_THRESHOLD"),
            "0.3",
            "VISION_FORCE_OPENROUTER_THRESHOLD",
        ),
        # Concurrency and performance limits
        "VISION_MAX_CONCURRENT_JOBS": _low_resource_int(
            "VISION_MAX_CONCURRENT_JOBS", 3, 1
        ),
        "VISION_MAX_USER_CONCURRENT_JOBS": _safe_int(
            os.getenv("VISION_MAX_USER_CONCURRENT_JOBS"),
            "1",
            "VISION_MAX_USER_CONCURRENT_JOBS",
        ),
        "VISION_JOB_TIMEOUT_SECONDS": _safe_int(
            os.getenv("VISION_JOB_TIMEOUT_SECONDS"), "300", "VISION_JOB_TIMEOUT_SECONDS"
        ),
        # Artifact management
        "VISION_ARTIFACT_TTL_DAYS": _safe_int(
            os.getenv("VISION_ARTIFACT_TTL_DAYS"), "7", "VISION_ARTIFACT_TTL_DAYS"
        ),
        "VISION_MAX_ARTIFACT_SIZE_MB": _safe_int(
            os.getenv("VISION_MAX_ARTIFACT_SIZE_MB"),
            "50",
            "VISION_MAX_ARTIFACT_SIZE_MB",
        ),
        "VISION_MAX_TOTAL_ARTIFACTS_GB": _safe_int(
            os.getenv("VISION_MAX_TOTAL_ARTIFACTS_GB"),
            "10",
            "VISION_MAX_TOTAL_ARTIFACTS_GB",
        ),
        # Logging and observability
        "VISION_LOG_LEVEL": os.getenv("VISION_LOG_LEVEL", "INFO"),
        "VISION_AUDIT_ENABLED": _parse_bool_str(
            _clean_env_value(os.getenv("VISION_AUDIT_ENABLED")), True
        ),
        # Provider-specific timeouts and retries
        "VISION_PROVIDER_TIMEOUT_MS": _safe_int(
            os.getenv("VISION_PROVIDER_TIMEOUT_MS"),
            "30000",
            "VISION_PROVIDER_TIMEOUT_MS",
        ),
        "VISION_PROVIDER_MAX_RETRIES": _safe_int(
            os.getenv("VISION_PROVIDER_MAX_RETRIES"), "3", "VISION_PROVIDER_MAX_RETRIES"
        ),
        "VISION_PROVIDER_RETRY_DELAY_MS": _safe_int(
            os.getenv("VISION_PROVIDER_RETRY_DELAY_MS"),
            "1000",
            "VISION_PROVIDER_RETRY_DELAY_MS",
        ),
        "VL_NOTES_TIMEOUT_S": _safe_float(
            os.getenv("VL_NOTES_TIMEOUT_S"), "120.0", "VL_NOTES_TIMEOUT_S"
        ),
        "VISION_PER_ITEM_BUDGET": _safe_float(
            os.getenv("VISION_PER_ITEM_BUDGET"), "120.0", "VISION_PER_ITEM_BUDGET"
        ),
        "VL_REQUEST_TIMEOUT": _safe_float(
            os.getenv("VL_REQUEST_TIMEOUT"), "30.0", "VL_REQUEST_TIMEOUT"
        ),
        # Discord integration
        "VISION_PROGRESS_UPDATE_INTERVAL_S": _safe_int(
            os.getenv("VISION_PROGRESS_UPDATE_INTERVAL_S"),
            "10",
            "VISION_PROGRESS_UPDATE_INTERVAL_S",
        ),
        "VISION_EPHEMERAL_RESPONSES": _parse_bool_str(
            _clean_env_value(os.getenv("VISION_EPHEMERAL_RESPONSES")), True
        ),
        # Dry run mode for testing routing and cost decisions
        "VISION_DRY_RUN_MODE": _parse_bool_str(
            _clean_env_value(os.getenv("VISION_DRY_RUN_MODE")), False
        ),
        # Orchestration mode: single | cascade_primary_then_fallbacks | parallel_first_acceptable | parallel_best_of | hybrid_draft_then_finalize
        "STT_MODE": os.getenv("STT_MODE", "single"),
        # Active providers (comma-separated). Supported now: local_whisper
        "STT_ACTIVE_PROVIDERS": [
            s.strip()
            for s in os.getenv("STT_ACTIVE_PROVIDERS", "local_whisper").split(",")
            if s.strip()
        ],
        # Minimum confidence to accept result (providers lacking confidence are always acceptable)
        "STT_CONFIDENCE_MIN": _safe_float(
            os.getenv("STT_CONFIDENCE_MIN"), "0.0", "STT_CONFIDENCE_MIN"
        ),
        # Cache TTL for successful transcripts (seconds)
        "STT_CACHE_TTL": _safe_int(os.getenv("STT_CACHE_TTL"), "600", "STT_CACHE_TTL"),
        # Local provider concurrency controls
        "STT_LOCAL_CONCURRENCY": _low_resource_int(
            "STT_LOCAL_CONCURRENCY", 2, 1
        ),
        # ---------------------------------------------------------------
        # LOW_RESOURCE_MODE-adjusted settings (env vars always override)
        # ---------------------------------------------------------------
        # RAG / Embedding lazy-load toggle — when true, skip eager model init
        "RAG_DISABLE_EAGER_LOAD": _low_resource_bool(
            "RAG_DISABLE_EAGER_LOAD", False, True
        ),
        # TTS warmup — skip pre-warm in low-resource mode
        "TTS_SKIP_WARMUP": _low_resource_bool(
            "TTS_SKIP_WARMUP", False, True
        ),
        # Vision parse-fallback: conservative cost when usage parsing fails [Phase 17-23]
        "VISION_BUDGET_PARSE_FALLBACK_COST_USD": _safe_float(
            os.getenv("VISION_BUDGET_PARSE_FALLBACK_COST_USD"),
            "0.02",
            "VISION_BUDGET_PARSE_FALLBACK_COST_USD",
        ),
        "VISION_PARSE_FALLBACK_CHARGE": _low_resource_float(
            "VISION_PARSE_FALLBACK_CHARGE", 0.01, 0.005
        ),
        "VISION_LOW_RESOURCE_RETRIES": _low_resource_int(
            "VISION_LOW_RESOURCE_RETRIES", 3, 2
        ),
        # --- Resource caps [Phase 12-16] ---
        "MULTIMODAL_MAX_ITEMS": _low_resource_int(
            "MULTIMODAL_MAX_ITEMS", 5, 2
        ),
        "MULTIMODAL_MAX_TOTAL_BYTES": _low_resource_int(
            "MULTIMODAL_MAX_TOTAL_BYTES", 50 * 1024 * 1024, 10 * 1024 * 1024
        ),
        "MULTIMODAL_CONCURRENCY": _low_resource_int(
            "MULTIMODAL_CONCURRENCY", 3, 1
        ),
        "IMAGE_MAX_DIMENSION": _low_resource_int(
            "IMAGE_MAX_DIMENSION", 2048, 1024
        ),
        "PDF_MAX_PAGES": _low_resource_int(
            "PDF_MAX_PAGES", 20, 5
        ),
        "VIDEO_MAX_DURATION_S": _low_resource_float(
            "VIDEO_MAX_DURATION_S", 300.0, 60.0
        ),
        "TTS_MAX_CHARS": _low_resource_int(
            "TTS_MAX_CHARS", 4000, 2000
        ),
        "TTS_SKIP_LONG_RESPONSES": _low_resource_bool(
            "TTS_SKIP_LONG_RESPONSES", False, True
        ),
        "VL_MAX_IMAGES": _low_resource_int(
            "VL_MAX_IMAGES", 5, 2
        ),
        "VL_MAX_IMAGE_DIMENSION": _low_resource_int(
            "VL_MAX_IMAGE_DIMENSION", 2048, 1024
        ),
        "SCREENSHOT_MAX_BYTES": _low_resource_int(
            "SCREENSHOT_MAX_BYTES", 5 * 1024 * 1024, 1 * 1024 * 1024
        ),
        "STT_MAX_AUDIO_DURATION_S": _low_resource_float(
            "STT_MAX_AUDIO_DURATION_S", 300.0, 60.0
        ),
        # RAG document parser workers
        "RAG_DOCUMENT_WORKERS": _low_resource_int(
            "RAG_DOCUMENT_WORKERS", 4, 1
        ),
        # HTTP / aiohttp connector pool size (shared)
        "HTTP_POOL_MAX_CONNECTIONS": _low_resource_int(
            "HTTP_POOL_MAX_CONNECTIONS", 50, 10
        ),
        # HTTP response body cap [Phase 17-23]
        "URL_MAX_RESPONSE_BYTES": _low_resource_int(
            "URL_MAX_RESPONSE_BYTES", 500 * 1024, 200 * 1024
        ),
        "HTTP_READ_TIMEOUT_S": _low_resource_float(
            "HTTP_READ_TIMEOUT_S", 30.0, 15.0
        ),
        # Config watcher debounce [Phase 17-23]
        "CONFIG_WATCH_DEBOUNCE_S": _low_resource_float(
            "CONFIG_WATCH_DEBOUNCE_S", 1.0, 2.0
        ),
        # Log sanitization [Phase 17-23]
        "LOG_RATE_LIMIT_WINDOW_S": _safe_int(
            os.getenv("LOG_RATE_LIMIT_WINDOW_S"), "60", "LOG_RATE_LIMIT_WINDOW_S"
        ),
        "LOG_MAX_STRING_LENGTH": _low_resource_int(
            "LOG_MAX_STRING_LENGTH", 1000, 300
        ),
        # Playwright concurrency (browser instances)
        "PLAYWRIGHT_MAX_CONCURRENT": _low_resource_int(
            "PLAYWRIGHT_MAX_CONCURRENT", 3, 1
        ),
        # Discord internal message cache
        "DISCORD_MESSAGE_CACHE_MAX": _low_resource_int(
            "DISCORD_MESSAGE_CACHE_MAX", 256, 64
        ),
        # TTS synthesis concurrency
        "TTS_CONCURRENCY": _low_resource_int(
            "TTS_CONCURRENCY", 4, 1
        ),
        # Concurrency manager pool sizes
        "ROUTER_MAX_CONCURRENCY_LIGHT": _low_resource_int(
            "ROUTER_MAX_CONCURRENCY_LIGHT", 8, 4
        ),
        "ROUTER_MAX_CONCURRENCY_NETWORK": _low_resource_int(
            "ROUTER_MAX_CONCURRENCY_NETWORK", 32, 8
        ),
        "ROUTER_MAX_CONCURRENCY_HEAVY": _low_resource_int(
            "ROUTER_MAX_CONCURRENCY_HEAVY", 2, 1
        ),
        # MULTIMODAL STT FALLBACK CONFIGURATION [CA][REH]
        # Enable multimodal fallback when primary STT fails
        "STT_MULTIMODAL_FALLBACK_ENABLED": _parse_bool_str(
            _clean_env_value(os.getenv("STT_MULTIMODAL_FALLBACK_ENABLED")), False
        ),
        # Comma-separated list of multimodal models for fallback (OpenRouter format)
        "STT_MULTIMODAL_FALLBACK_MODELS": os.getenv(
            "STT_MULTIMODAL_FALLBACK_MODELS",
            "openrouter/openai-whisper-large-v3,openrouter/meta-llama-3-8b-instruct:free",
        ),
        # Timeout for multimodal fallback API calls (seconds)
        "STT_MULTIMODAL_FALLBACK_TIMEOUT_S": _safe_float(
            os.getenv("STT_MULTIMODAL_FALLBACK_TIMEOUT_S"),
            "30.0",
            "STT_MULTIMODAL_FALLBACK_TIMEOUT_S",
        ),
        # Minimum confidence threshold for fallback results
        "STT_MULTIMODAL_FALLBACK_MIN_CONFIDENCE": _safe_float(
            os.getenv("STT_MULTIMODAL_FALLBACK_MIN_CONFIDENCE"),
            "0.5",
            "STT_MULTIMODAL_FALLBACK_MIN_CONFIDENCE",
        ),
        # Maximum retry attempts for multimodal fallback
        "STT_MULTIMODAL_FALLBACK_MAX_RETRIES": _safe_int(
            os.getenv("STT_MULTIMODAL_FALLBACK_MAX_RETRIES"),
            "1",
            "STT_MULTIMODAL_FALLBACK_MAX_RETRIES",
        ),
    }

    # Deprecation warnings for legacy config keys [SFT]
    if os.getenv("TEXT_MODEL"):
        logger.warning(
            "⚠️ TEXT_MODEL is deprecated. Use OPENAI_TEXT_MODEL instead. "
            "Support for TEXT_MODEL will be removed in a future release."
        )
    if os.getenv("OPENAI_MODEL"):
        logger.warning(
            "⚠️ OPENAI_MODEL is deprecated. Use OPENAI_TEXT_MODEL instead. "
            "Support for OPENAI_MODEL will be removed in a future release."
        )

    # One-time startup VISION flags summary [PA]
    try:
        ve_src = "env" if _ve_raw is not None else "default"
        t2i_src = "env" if _t2i_raw is not None else "default"
        logger.info(
            f"VISION_FLAGS raw={{VISION_ENABLED:{_ve_raw}, VISION_T2I_ENABLED:{_t2i_raw}}} "
            f"parsed={{vision_enabled:{_ve}, t2i:{_t2i}}} "
            f"source={{vision_enabled:{ve_src}, t2i:{t2i_src}}}"
        )
    except Exception:
        pass

    # Cache the config for performance (avoid repeated env var lookups)
    _config_cache = config
    _cache_timestamp = current_time
    logger.debug(f"✅ Configuration cached for {CACHE_TTL}s")

    return config


def get_vl_model_ladder() -> list[str]:
    """
    Return the configured VL model ladder. Supports comma-separated VL_MODEL.
    Falls back to a safe default ladder when unset.
    """
    config = load_config()
    raw = config.get("VL_MODEL")
    models = _parse_model_list(raw)
    if models:
        return models
    if raw:
        cleaned = raw.strip()
        if cleaned:
            return [cleaned]
    return _DEFAULT_VL_MODEL_LADDER.copy()


def invalidate_config_cache() -> None:
    """Invalidate the in-process config cache to force fresh reads on next load_config().
    Intended for use by the hot-reload path immediately after .env is reloaded. [REH][CMV]
    """
    global _config_cache, _cache_timestamp
    _config_cache = None
    _cache_timestamp = 0.0
    try:
        logger.info(
            "config.cache.invalidate",
            extra={
                "event": "config.cache.invalidate",
                "detail": {"reason": "reload_request"},
            },
        )
    except Exception:
        pass


# Force English IPA route (bypass tokenizer env and disable autodiscovery)
KOKORO_FORCE_IPA_EN = True


# ---------------------------------------------------------------------------
# LOW_RESOURCE_MODE — conservative defaults when memory / CPU is constrained
# ---------------------------------------------------------------------------
_low_resource_mode_env = _parse_bool_str(
    _clean_env_value(os.getenv("LOW_RESOURCE_MODE")),
    False,
)


def _low_resource_int(env_key: str, normal: int, low: int) -> int:
    """Return the env var value if set; otherwise pick low vs normal based on LOW_RESOURCE_MODE."""
    raw = _clean_env_value(os.getenv(env_key))
    if raw is not None:
        try:
            return int(raw.split("#")[0].strip())
        except (ValueError, AttributeError):
            pass
    return low if _low_resource_mode_env else normal


def _low_resource_float(env_key: str, normal: float, low: float) -> float:
    raw = _clean_env_value(os.getenv(env_key))
    if raw is not None:
        try:
            return float(raw.split("#")[0].strip())
        except (ValueError, AttributeError):
            pass
    return low if _low_resource_mode_env else normal


def _low_resource_bool(env_key: str, normal: bool, low: bool) -> bool:
    raw = _clean_env_value(os.getenv(env_key))
    if raw is not None:
        return _parse_bool_str(raw, normal)
    return low if _low_resource_mode_env else normal


# ---------------------------------------------------------------------------
# CONTEXT TRIMMING — limits applied when constructing LLM context strings
# ---------------------------------------------------------------------------
CONTEXT_MAX_MESSAGES = _low_resource_int(
    "CONTEXT_MAX_MESSAGES",
    10,
    5,
)
CONTEXT_MAX_CHARS_PER_MESSAGE = _low_resource_int(
    "CONTEXT_MAX_CHARS_PER_MESSAGE",
    2000,
    500,
)
CONTEXT_MAX_TOTAL_CHARS = _low_resource_int(
    "CONTEXT_MAX_TOTAL_CHARS",
    8000,
    2000,
)
# When building context, skip bot continuation/"more..." lines
CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS = _parse_bool_str(
    _clean_env_value(os.getenv("CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS")),
    True,
)
