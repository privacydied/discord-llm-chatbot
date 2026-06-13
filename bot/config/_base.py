"""Configuration loading and environment setup."""

import contextlib
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from bot.exceptions import ConfigurationError
from bot.utils.logging import get_logger

# CHANGE: Enhanced .env loading with comprehensive audit and logging
logger = get_logger(__name__)

# Load environment variables from .env file with explicit path
load_dotenv(dotenv_path=Path.cwd() / ".env", verbose=True)

# Also try loading from the project root in case we're running from a subdirectory
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", verbose=False)


def audit_env_file() -> None:
    """Audit .env file by reading and logging every variable.
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
    """Validate that all required environment variables are present.
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
        msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        raise ConfigurationError(msg)


def validate_prompt_files() -> None:
    """Validate that prompt files exist and are readable."""
    prompt_file = os.getenv("PROMPT_FILE")
    vl_prompt_file = os.getenv("VL_PROMPT_FILE")

    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            msg = f"PROMPT_FILE not found: {prompt_path}"
            raise ConfigurationError(msg)
        logger.debug(f"✅ Text prompt file found: {prompt_path}")

    if vl_prompt_file:
        vl_prompt_path = Path(vl_prompt_file)
        if not vl_prompt_path.exists():
            msg = f"VL_PROMPT_FILE not found: {vl_prompt_path}"
            raise ConfigurationError(msg)
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
        prompt_file = os.getenv("TEXT_PROMPT_PATH") or os.getenv("PROMPT_FILE", "prompts/prompt-yoroi-super-chill.txt")
        vl_prompt_file = os.getenv("VL_PROMPT_PATH") or os.getenv("VL_PROMPT_FILE", "prompts/vl-prompt.txt")

        prompts["text_prompt"] = Path(prompt_file).read_text()
        prompts["vl_prompt"] = Path(vl_prompt_file).read_text()

        logger.info(f"✅ Loaded system prompts: {list(prompts.keys())}")
        return prompts
    except FileNotFoundError as e:
        logger.warning(f"⚠️ Prompt file not found at {e.filename}; using minimal fallback prompts for startup.")
        prompts.setdefault("text_prompt", "You are a helpful assistant.")
        prompts.setdefault("vl_prompt", "Describe the image succinctly.")
        return prompts


def check_venv_activation() -> None:
    """Enforce exclusive .venv usage as specified in requirements.
    CHANGE: Added .venv enforcement check to ensure proper environment usage.
    """
    if ".venv" not in sys.prefix:
        logger.warning("⚠️  Running outside .venv—please activate .venv before using uv run")
        logger.warning(f"Current Python path: {sys.prefix}")
    else:
        logger.debug(f"✅ Running in .venv: {sys.prefix}")


def _safe_int(value: str | None, default: str, var_name: str) -> int:
    """Safely convert environment variable to int, handling malformed values."""
    try:
        # Clean value by removing comments and whitespace
        clean_value = value.split("#")[0].strip() if value else default
        return int(clean_value)
    except (ValueError, AttributeError):
        logger.warning("Invalid %s value '%s', using default %s", var_name, value, default)
        return int(default)


def _safe_float(value: str | None, default: str, var_name: str) -> float:
    """Safely convert environment variable to float, handling malformed values."""
    try:
        # Clean value by removing comments and whitespace
        clean_value = value.split("#")[0].strip() if value else default
        return float(clean_value)
    except (ValueError, AttributeError):
        logger.warning("Invalid %s value '%s', using default %s", var_name, value, default)
        return float(default)


def _clean_env_value(value: str | None) -> str:
    """Clean environment variable value by removing inline comments.
    CHANGE: Added to handle .env files with inline comments.
    """
    if not value:
        return value or ""
    # Split on # and take the first part, then strip whitespace
    return value.split("#")[0].strip()


# Robust boolean parsing to avoid bool("off") traps [IV][CMV]
def _parse_bool_str(raw: str | None, default: bool) -> bool:
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
_config_cache: dict[str, Any] | None = None
_cache_timestamp: float = 0
CACHE_TTL = 300  # 5 minute cache TTL

# Last known good config for hot-reload rollback
_last_good_config: dict[str, Any] | None = None


def _parse_model_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


_DEFAULT_VL_MODEL_LADDER = [
    "moonshotai/kimi-vl-a3b-thinking:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
]


def _build_config(env_getter: Callable[[str, str | None], str | None]) -> dict[str, Any]:
    """Build configuration dictionary using the provided environment getter function.

    Args:
        env_getter: Function(key, default) -> value. Called for each config key.
                   For load_config(), this is os.getenv. For load_config_candidate(),
                   this reads from the candidate .env file only.

    Returns:
        Configuration dictionary.

    """
    # Centralized VISION flags with robust parsing and defaults [CA]
    _ve_raw = _clean_env_value(env_getter("VISION_ENABLED"))
    _t2i_raw = _clean_env_value(env_getter("VISION_T2I_ENABLED"))
    _ve = _parse_bool_str(_ve_raw, True)  # default ON when unset
    _t2i = _parse_bool_str(_t2i_raw, True)  # default ON when unset

    config = {
        # DISCORD BOT SETTINGS
        "DISCORD_TOKEN": env_getter("DISCORD_TOKEN", None),
        "TEXT_BACKEND": env_getter("TEXT_BACKEND", "openai"),
        # OPENAI / OPENROUTER SETTINGS
        "OPENAI_API_KEY": env_getter("OPENAI_API_KEY", None),
        "OPENAI_API_BASE": env_getter("OPENAI_API_BASE", None),
        "OPENAI_TEXT_MODEL": env_getter("OPENAI_TEXT_MODEL", None),
        "OPENROUTER_API_KEY": env_getter("OPENROUTER_API_KEY", None),
        "TEXT_FALLBACK_MODELS": env_getter("TEXT_FALLBACK_MODELS", None),
        "TEXT_FALLBACK_TIMEOUTS": env_getter("TEXT_FALLBACK_TIMEOUTS", None),
        "TEXT_FALLBACK_MAX_ATTEMPTS": env_getter("TEXT_FALLBACK_MAX_ATTEMPTS", None),
        "NVIDIA_NIM_API_KEY": env_getter("NVIDIA_NIM_API_KEY", None),
        "NVIDIA_NIM_API_BASE": env_getter("NVIDIA_NIM_API_BASE", None),
        "NVIDIA_NIM_TEXT_MODEL": env_getter("NVIDIA_NIM_TEXT_MODEL", None),
        "VL_MODEL": _clean_env_value(env_getter("VL_MODEL", None)),
        # OLLAMA SETTINGS
        "OLLAMA_BASE_URL": env_getter("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": env_getter("OLLAMA_MODEL", "llama3"),
        "TEXT_MODEL": env_getter("TEXT_MODEL", None),
        # BOT BEHAVIOR / CONTEXT / MEMORY
        "TEMPERATURE": _safe_float(env_getter("TEMPERATURE", None), "0.7", "TEMPERATURE"),
        "TIMEOUT": _safe_float(env_getter("TIMEOUT", None), "120.0", "TIMEOUT"),
        "CHANGE_NICKNAME": _parse_bool_str(_clean_env_value(env_getter("CHANGE_NICKNAME", None)), False),
        "MAX_CONVERSATION_LENGTH": _safe_int(env_getter("MAX_CONVERSATION_LENGTH", None), "1000", "MAX_CONVERSATION_LENGTH"),
        "MAX_TEXT_ATTACHMENT_SIZE": _safe_int(env_getter("MAX_TEXT_ATTACHMENT_SIZE", None), "20000", "MAX_TEXT_ATTACHMENT_SIZE"),
        "MAX_FILE_SIZE": _safe_int(env_getter("MAX_FILE_SIZE", None), "2097152", "MAX_FILE_SIZE"),
        "MAX_ATTACHMENT_SIZE_MB": _safe_int(env_getter("MAX_ATTACHMENT_SIZE_MB", None), "25", "MAX_ATTACHMENT_SIZE_MB"),
        # SILENCE GATE - SPEAK ONLY WHEN SPOKEN TO
        "BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO": _parse_bool_str(_clean_env_value(env_getter("BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO", None)), True),
        "REQUIRE_MENTION_IN_GUILDS": _parse_bool_str(_clean_env_value(env_getter("REQUIRE_MENTION_IN_GUILDS", None)), True),
        "ALLOW_REPLY_TO_BOT_WITHOUT_MENTION": _parse_bool_str(_clean_env_value(env_getter("ALLOW_REPLY_TO_BOT_WITHOUT_MENTION", None)), True),
        "DM_REQUIRE_MENTION": _parse_bool_str(_clean_env_value(env_getter("DM_REQUIRE_MENTION", None)), False),
        "REPLY_TRIGGERS": [s.strip() for s in env_getter("REPLY_TRIGGERS", "dm,mention,reply,bot_threads,owner,command_prefix").split(",") if s.strip()],
        # PROMPT FILES - CRITICAL FOR MULTIMODAL FUNCTIONALITY
        "PROMPT_FILE": _clean_env_value(env_getter("PROMPT_FILE", None)),
        "VL_PROMPT_FILE": _clean_env_value(env_getter("VL_PROMPT_FILE", None)),
        # STT SETTINGS
        "STT_ENGINE": env_getter("STT_ENGINE", "faster-whisper"),
        "STT_FALLBACK": env_getter("STT_FALLBACK", "whispercpp"),
        "WHISPER_MODEL_SIZE": env_getter("WHISPER_MODEL_SIZE", "medium-int8"),
        "WHISPER_CPP_MODEL": env_getter("WHISPER_CPP_MODEL", "ggml-medium.bin"),
        # WHISPER SETTINGS
        "WHISPER_API_KEY": env_getter("WHISPER_API_KEY", None),
        "WHISPER_API_BASE": env_getter("WHISPER_API_BASE", None),
        "WHISPER_MODEL": env_getter("WHISPER_MODEL", "whisper-1"),
        # MEMORY SETTINGS
        "MAX_USER_MEMORY": _safe_int(env_getter("MAX_USER_MEMORY", None), "20", "MAX_USER_MEMORY"),
        "MAX_SERVER_MEMORY": _safe_int(env_getter("MAX_SERVER_MEMORY", None), "100", "MAX_SERVER_MEMORY"),
        "MEMORY_SAVE_INTERVAL": _safe_int(env_getter("MEMORY_SAVE_INTERVAL", None), "30", "MEMORY_SAVE_INTERVAL"),
        "PERSISTENT_MEMORY_ENABLE": _parse_bool_str(env_getter("PERSISTENT_MEMORY_ENABLE", None), True),
        "PERSISTENT_MEMORY_SQLITE_PATH": env_getter("PERSISTENT_MEMORY_SQLITE_PATH", "./data/memory.db"),
        "PERSISTENT_MEMORY_CHROMA_PATH": env_getter("PERSISTENT_MEMORY_CHROMA_PATH", "./chroma_db"),
        "PERSISTENT_MEMORY_CHROMA_COLLECTION": env_getter("PERSISTENT_MEMORY_CHROMA_COLLECTION", "curated_memories"),
        "PERSISTENT_MEMORY_QUEUE_MAX": _safe_int(
            env_getter("PERSISTENT_MEMORY_QUEUE_MAX", None),
            "256",
            "PERSISTENT_MEMORY_QUEUE_MAX",
        ),
        "PERSISTENT_MEMORY_WORKERS": _low_resource_int("PERSISTENT_MEMORY_WORKERS", 1, 1),
        "PERSISTENT_MEMORY_TOP_K": _low_resource_int(
            "PERSISTENT_MEMORY_TOP_K",
            6,
            3,
        ),
        "PERSISTENT_MEMORY_MAX_PROMPT_CHARS": _safe_int(
            env_getter("PERSISTENT_MEMORY_MAX_PROMPT_CHARS", None),
            "1200",
            "PERSISTENT_MEMORY_MAX_PROMPT_CHARS",
        ),
        # Memory semantic dedupe & resource caps [Phase 6-9]
        "MEMORY_SEMANTIC_DEDUPE_ENABLED": _low_resource_bool("MEMORY_SEMANTIC_DEDUPE_ENABLED", True, True),
        "MEMORY_MAX_TEXT_CHARS": _low_resource_int("MEMORY_MAX_TEXT_CHARS", 500, 300),
        "MEMORY_SEMANTIC_TOP_K": _low_resource_int("MEMORY_SEMANTIC_TOP_K", 5, 2),
        "MEMORY_INGEST_WORKERS": _low_resource_int("MEMORY_INGEST_WORKERS", 2, 1),
        "MEMORY_RECALL_CACHE_TTL_S": _low_resource_int("MEMORY_RECALL_CACHE_TTL_S", 30, 15),
        "MEMORY_AUTO_CURATION_ENABLED": _low_resource_bool("MEMORY_AUTO_CURATION_ENABLED", True, False),
        "MEMORY_DISTILL_INTERVAL_S": _low_resource_int("MEMORY_DISTILL_INTERVAL_S", 900, 3600),
        "CHROMADB_MAX_RESULTS": _low_resource_int("CHROMADB_MAX_RESULTS", 5, 3),
        "PERSISTENT_MEMORY_DEFAULT_TTL_DAYS": _safe_int(
            env_getter("PERSISTENT_MEMORY_DEFAULT_TTL_DAYS", None),
            "180",
            "PERSISTENT_MEMORY_DEFAULT_TTL_DAYS",
        ),
        "PERSISTENT_MEMORY_TEMP_TTL_DAYS": _safe_int(
            env_getter("PERSISTENT_MEMORY_TEMP_TTL_DAYS", None),
            "14",
            "PERSISTENT_MEMORY_TEMP_TTL_DAYS",
        ),
        "PERSISTENT_MEMORY_MIN_IMPORTANCE": _safe_float(
            env_getter("PERSISTENT_MEMORY_MIN_IMPORTANCE", None),
            "0.55",
            "PERSISTENT_MEMORY_MIN_IMPORTANCE",
        ),
        # SERVER ARCHIVE SETTINGS
        "SERVER_ARCHIVE_ENABLED": _parse_bool_str(
            _clean_env_value(env_getter("SERVER_ARCHIVE_ENABLED", None) or env_getter("SERVER_ARCHIVE_ENABLE", None)),
            False,
        ),
        "SERVER_ARCHIVE_ENABLE": _parse_bool_str(
            _clean_env_value(env_getter("SERVER_ARCHIVE_ENABLED", None) or env_getter("SERVER_ARCHIVE_ENABLE", None)),
            False,
        ),
        "SERVER_ARCHIVE_DB_PATH": env_getter("SERVER_ARCHIVE_DB_PATH", "./data/server_archive.db"),
        "SERVER_ARCHIVE_QUEUE_MAX": _low_resource_int("SERVER_ARCHIVE_QUEUE_MAX", 1000, 200),
        "SERVER_ARCHIVE_BATCH_SIZE": _low_resource_int("SERVER_ARCHIVE_BATCH_SIZE", 100, 20),
        "SERVER_ARCHIVE_DISTILL_INTERVAL_S": _low_resource_int("SERVER_ARCHIVE_DISTILL_INTERVAL_S", 900, 3600),
        "SERVER_ARCHIVE_SEARCH_LIMIT": _safe_int(
            env_getter("SERVER_ARCHIVE_SEARCH_LIMIT", None),
            "10",
            "SERVER_ARCHIVE_SEARCH_LIMIT",
        ),
        "SERVER_ARCHIVE_ADMIN_ONLY": _parse_bool_str(_clean_env_value(env_getter("SERVER_ARCHIVE_ADMIN_ONLY", None)), True),
        "SERVER_ARCHIVE_SYNC_ON_START": _parse_bool_str(_clean_env_value(env_getter("SERVER_ARCHIVE_SYNC_ON_START", None)), True),
        "SERVER_ARCHIVE_LIVE_TAIL": _parse_bool_str(_clean_env_value(env_getter("SERVER_ARCHIVE_LIVE_TAIL", None)), True),
        "SERVER_ARCHIVE_MAX_MESSAGE_CHARS": _safe_int(
            env_getter("SERVER_ARCHIVE_MAX_MESSAGE_CHARS", None),
            "8000",
            "SERVER_ARCHIVE_MAX_MESSAGE_CHARS",
        ),
        "SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES": _parse_bool_str(
            _clean_env_value(env_getter("SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES", None) or env_getter("SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES", None)),
            False,
        ),
        "SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES": _parse_bool_str(
            _clean_env_value(env_getter("SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES", None) or env_getter("SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES", None)),
            False,
        ),
        # MEMORY DISTILLER SETTINGS
        "MEMORY_DISTILLER_ENABLED": _parse_bool_str(
            _clean_env_value(env_getter("MEMORY_DISTILLER_ENABLED", None)),
            False,
        ),
        "MEMORY_DISTILLER_DRY_RUN": _parse_bool_str(
            _clean_env_value(env_getter("MEMORY_DISTILLER_DRY_RUN", None)),
            True,
        ),
        "MEMORY_DISTILLER_BATCH_SIZE": _safe_int(
            env_getter("MEMORY_DISTILLER_BATCH_SIZE", None),
            "200",
            "MEMORY_DISTILLER_BATCH_SIZE",
        ),
        "MEMORY_DISTILLER_INTERVAL_SECONDS": _low_resource_int(
            "MEMORY_DISTILLER_INTERVAL_SECONDS",
            900,
            3600,
        ),
        "MEMORY_DISTILLER_WINDOW_MESSAGES": _safe_int(
            env_getter("MEMORY_DISTILLER_WINDOW_MESSAGES", None),
            "25",
            "MEMORY_DISTILLER_WINDOW_MESSAGES",
        ),
        "MEMORY_DISTILLER_MIN_CONFIDENCE": _safe_float(
            env_getter("MEMORY_DISTILLER_MIN_CONFIDENCE", None),
            "0.85",
            "MEMORY_DISTILLER_MIN_CONFIDENCE",
        ),
        "MEMORY_DISTILLER_MAX_MEMORIES_PER_WINDOW": _safe_int(
            env_getter("MEMORY_DISTILLER_MAX_MEMORIES_PER_WINDOW", None),
            "3",
            "MEMORY_DISTILLER_MAX_MEMORIES_PER_WINDOW",
        ),
        "MEMORY_DISTILLER_EXCLUDE_BOT_MESSAGES": _parse_bool_str(
            _clean_env_value(env_getter("MEMORY_DISTILLER_EXCLUDE_BOT_MESSAGES", None)),
            True,
        ),
        "CONTEXT_FILE_PATH": env_getter("CONTEXT_FILE_PATH", "runtime/context.json"),
        "MAX_CONTEXT_MESSAGES": _safe_int(env_getter("MAX_CONTEXT_MESSAGES", None), "10", "MAX_CONTEXT_MESSAGES"),
        "MEM_MAX_MSGS": _safe_int(env_getter("MEM_MAX_MSGS", None), "40", "MEM_MAX_MSGS"),
        "MEM_MAX_CHARS": _safe_int(env_getter("MEM_MAX_CHARS", None), "8000", "MEM_MAX_CHARS"),
        "MEM_MAX_AGE_MIN": _safe_int(env_getter("MEM_MAX_AGE_MIN", None), "240", "MEM_MAX_AGE_MIN"),
        "MEM_FETCH_TIMEOUT_S": _safe_float(env_getter("MEM_FETCH_TIMEOUT_S", None), "5", "MEM_FETCH_TIMEOUT_S"),
        "MEM_LOG_SUBSYS": env_getter("MEM_LOG_SUBSYS", "mem.ctx"),
        # Thread-tail reply + context collector [CMV][REH]
        "THREAD_CONTEXT_TAIL_COUNT": _safe_int(env_getter("THREAD_CONTEXT_TAIL_COUNT", None), "5", "THREAD_CONTEXT_TAIL_COUNT"),
        # Context trimming — character and message limits
        "CONTEXT_MAX_MESSAGES": CONTEXT_MAX_MESSAGES,
        "CONTEXT_MAX_CHARS_PER_MESSAGE": CONTEXT_MAX_CHARS_PER_MESSAGE,
        "CONTEXT_MAX_TOTAL_CHARS": CONTEXT_MAX_TOTAL_CHARS,
        "CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS": CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS,
        # DIRECTORY SETTINGS
        "USER_PROFILE_DIR": Path(env_getter("USER_PROFILE_DIR", "user_profiles")),
        "SERVER_PROFILE_DIR": Path(env_getter("SERVER_PROFILE_DIR", "server_profiles")),
        "USER_LOGS_DIR": Path(env_getter("USER_LOGS_DIR", "user_logs")),
        "DM_LOGS_DIR": Path(env_getter("DM_LOGS_DIR", "dm_logs")),
        "TEMP_DIR": Path(env_getter("TEMP_DIR", "temp")),
        "LOGS_DIR": Path(env_getter("LOGS_DIR", "logs")),
        # TTS SETTINGS
        "TTS_BACKEND": env_getter("TTS_BACKEND", "kokoro-onnx"),
        "TTS_VOICE": env_getter("TTS_VOICE", "af"),
        # Timeouts for TTS synthesis [CMV][IV]
        "TTS_TIMEOUT_S": _safe_float(env_getter("TTS_TIMEOUT_S", None), "25.0", "TTS_TIMEOUT_S"),
        "TTS_TIMEOUT_COLD_S": _safe_float(
            env_getter("TTS_TIMEOUT_COLD_S", None) or env_getter("TTS_TIMEOUT_S", None),
            "25.0",
            "TTS_TIMEOUT_COLD_S",
        ),
        "TTS_TIMEOUT_WARM_S": _safe_float(
            env_getter("TTS_TIMEOUT_WARM_S", None) or env_getter("TTS_TIMEOUT_S", None),
            "25.0",
            "TTS_TIMEOUT_WARM_S",
        ),
        # Native Discord voice messages toggle [CMV]
        "VOICE_ENABLE_NATIVE": _parse_bool_str(_clean_env_value(env_getter("VOICE_ENABLE_NATIVE", None)), False),
        # Voice Publisher HTTP timeouts (aiohttp) [CMV][IV]
        "VOICE_PUBLISHER_TIMEOUT_S": _safe_float(env_getter("VOICE_PUBLISHER_TIMEOUT_S", None), "0.0", "VOICE_PUBLISHER_TIMEOUT_S"),
        "VOICE_PUBLISHER_ATTACHMENTS_CREATE_TIMEOUT_S": _safe_float(
            env_getter("VOICE_PUBLISHER_ATTACHMENTS_CREATE_TIMEOUT_S", None),
            "30.0",
            "VOICE_PUBLISHER_ATTACHMENTS_CREATE_TIMEOUT_S",
        ),
        "VOICE_PUBLISHER_UPLOAD_TIMEOUT_S": _safe_float(
            env_getter("VOICE_PUBLISHER_UPLOAD_TIMEOUT_S", None),
            "60.0",
            "VOICE_PUBLISHER_UPLOAD_TIMEOUT_S",
        ),
        "VOICE_PUBLISHER_MESSAGE_POST_TIMEOUT_S": _safe_float(
            env_getter("VOICE_PUBLISHER_MESSAGE_POST_TIMEOUT_S", None),
            "30.0",
            "VOICE_PUBLISHER_MESSAGE_POST_TIMEOUT_S",
        ),
        # Opus encoding parameters for native voice messages [CMV]
        "VOICE_PUBLISHER_OPUS_BITRATE": env_getter("VOICE_PUBLISHER_OPUS_BITRATE", "64k"),
        "VOICE_PUBLISHER_OPUS_VBR": env_getter("VOICE_PUBLISHER_OPUS_VBR", "on"),
        "VOICE_PUBLISHER_OPUS_COMP_LEVEL": _safe_int(
            env_getter("VOICE_PUBLISHER_OPUS_COMP_LEVEL", None),
            "10",
            "VOICE_PUBLISHER_OPUS_COMP_LEVEL",
        ),
        # OPTIONAL SETTINGS
        "TTS_PREFS_FILE": env_getter("TTS_PREFS_FILE", None),
        "DEBUG": _parse_bool_str(_clean_env_value(env_getter("DEBUG", None)), False),
        "MAX_CONVERSATION_LOG_SIZE": _safe_int(env_getter("MAX_CONVERSATION_LOG_SIZE", None), "1000", "MAX_CONVERSATION_LOG_SIZE"),
        # LEGACY COMPATIBILITY
        "OPENAI_MODEL": env_getter("OPENAI_MODEL", "gpt-4"),
        "MAX_MEMORIES": _safe_int(
            env_getter("MAX_MEMORIES", None) or env_getter("MAX_USER_MEMORY", None),
            "100",
            "MAX_MEMORIES",
        ),
        "DEFAULT_TIMEOUT": _safe_int(env_getter("DEFAULT_TIMEOUT", None), "30", "DEFAULT_TIMEOUT"),
        "MAX_CONTEXT_LENGTH": _safe_int(env_getter("MAX_CONTEXT_LENGTH", None), "4000", "MAX_CONTEXT_LENGTH"),
        "MAX_RESPONSE_TOKENS": _safe_int(env_getter("MAX_RESPONSE_TOKENS", None), "1000", "MAX_RESPONSE_TOKENS"),
        "TOP_P": _safe_float(env_getter("TOP_P", None), "0.9", "TOP_P"),
        "FREQUENCY_PENALTY": _safe_float(env_getter("FREQUENCY_PENALTY", None), "0.0", "FREQUENCY_PENALTY"),
        "PRESENCE_PENALTY": _safe_float(env_getter("PRESENCE_PENALTY", None), "0.0", "PRESENCE_PENALTY"),
        "LOG_LEVEL": env_getter("LOG_LEVEL", "INFO"),
        "COMMAND_PREFIX": env_getter("COMMAND_PREFIX", "!"),
        "OWNER_IDS": [int(id.strip()) for id in env_getter("OWNER_IDS", "").split(",") if id.strip()],
        "LOG_FILE": env_getter("LOG_FILE", "logs/bot.jsonl"),
        # ADMIN ALERT SYSTEM [CA][CMV]
        "ALERT_ENABLE": _clean_env_value(env_getter("ALERT_ENABLE", "false")),
        "ALERT_SESSION_TIMEOUT_S": _clean_env_value(env_getter("ALERT_SESSION_TIMEOUT_S", "1800")),
        "ALERT_ADMIN_USER_IDS": _clean_env_value(env_getter("ALERT_ADMIN_USER_IDS", "")),
        # SEARCH SUBSYSTEM [CA][CMV][IV]
        "SEARCH_PROVIDER": env_getter("SEARCH_PROVIDER", "ddg").lower(),
        "SEARCH_MAX_RESULTS": _safe_int(env_getter("SEARCH_MAX_RESULTS", None), "5", "SEARCH_MAX_RESULTS"),
        "SEARCH_SAFE": env_getter("SEARCH_SAFE", "moderate").lower(),
        "SEARCH_LOCALE": env_getter("SEARCH_LOCALE", ""),
        "DDG_API_ENDPOINT": env_getter("DDG_API_ENDPOINT", "https://html.duckduckgo.com/html/"),
        "DDG_FORCE_HTML": _parse_bool_str(_clean_env_value(env_getter("DDG_FORCE_HTML", None)), True),
        "DDG_API_KEY": env_getter("DDG_API_KEY", None),
        "DDG_TIMEOUT_MS": _safe_int(env_getter("DDG_TIMEOUT_MS", None), "5000", "DDG_TIMEOUT_MS"),
        "CUSTOM_SEARCH_API_ENDPOINT": env_getter("CUSTOM_SEARCH_API_ENDPOINT", ""),
        "CUSTOM_SEARCH_API_KEY": env_getter("CUSTOM_SEARCH_API_KEY", ""),
        "CUSTOM_SEARCH_HEADERS": env_getter("CUSTOM_SEARCH_HEADERS", ""),
        "CUSTOM_SEARCH_TIMEOUT_MS": _safe_int(env_getter("CUSTOM_SEARCH_TIMEOUT_MS", None), "8000", "CUSTOM_SEARCH_TIMEOUT_MS"),
        "CUSTOM_SEARCH_RESULT_PATHS": env_getter("CUSTOM_SEARCH_RESULT_PATHS", ""),
        "SEARCH_POOL_MAX_CONNECTIONS": _low_resource_int(
            "SEARCH_POOL_MAX_CONNECTIONS",
            10,
            3,
        ),
        # Circuit breaker (search)
        "SEARCH_BREAKER_FAILURE_WINDOW": _safe_int(
            env_getter("SEARCH_BREAKER_FAILURE_WINDOW", None),
            "5",
            "SEARCH_BREAKER_FAILURE_WINDOW",
        ),
        "SEARCH_BREAKER_OPEN_MS": _safe_int(env_getter("SEARCH_BREAKER_OPEN_MS", None), "15000", "SEARCH_BREAKER_OPEN_MS"),
        "SEARCH_BREAKER_HALFOPEN_PROB": _safe_float(
            env_getter("SEARCH_BREAKER_HALFOPEN_PROB", None),
            "0.25",
            "SEARCH_BREAKER_HALFOPEN_PROB",
        ),
        # X (Twitter) API Integration [CA][CMV][SFT]
        "X_API_ENABLED": _parse_bool_str(_clean_env_value(env_getter("X_API_ENABLED", None)), False),
        "X_API_AUTH_MODE": env_getter("X_API_AUTH_MODE", "oauth2_app"),
        "X_API_BEARER_TOKEN": _clean_env_value(env_getter("X_API_BEARER_TOKEN", None)),
        "X_API_REQUIRE_API_FOR_TWITTER": _parse_bool_str(_clean_env_value(env_getter("X_API_REQUIRE_API_FOR_TWITTER", None)), False),
        "X_API_ALLOW_FALLBACK_ON_5XX": _parse_bool_str(_clean_env_value(env_getter("X_API_ALLOW_FALLBACK_ON_5XX", None)), True),
        "X_SYNDICATION_ENABLED": _parse_bool_str(_clean_env_value(env_getter("X_SYNDICATION_ENABLED", None)), True),
        "X_TWITTER_STT_PROBE_FIRST": _parse_bool_str(_clean_env_value(env_getter("X_TWITTER_STT_PROBE_FIRST", None)), True),
        "X_API_ROUTE_PHOTOS_TO_VL": _parse_bool_str(_clean_env_value(env_getter("X_API_ROUTE_PHOTOS_TO_VL", None)), True),
        "X_API_TIMEOUT_MS": _safe_int(env_getter("X_API_TIMEOUT_MS", None), "8000", "X_API_TIMEOUT_MS"),
        "X_API_RETRY_MAX_ATTEMPTS": _safe_int(env_getter("X_API_RETRY_MAX_ATTEMPTS", None), "5", "X_API_RETRY_MAX_ATTEMPTS"),
        "X_API_BREAKER_FAILURE_WINDOW": _safe_int(
            env_getter("X_API_BREAKER_FAILURE_WINDOW", None),
            "5",
            "X_API_BREAKER_FAILURE_WINDOW",
        ),
        "X_API_BREAKER_OPEN_MS": _safe_int(env_getter("X_API_BREAKER_OPEN_MS", None), "15000", "X_API_BREAKER_OPEN_MS"),
        "X_API_BREAKER_HALFOPEN_PROB": _safe_float(
            env_getter("X_API_BREAKER_HALFOPEN_PROB", None),
            "0.25",
            "X_API_BREAKER_HALFOPEN_PROB",
        ),
        "X_TWEET_FIELDS": [
            s.strip()
            for s in env_getter(
                "X_TWEET_FIELDS",
                "id,text,created_at,author_id,public_metrics,possibly_sensitive,lang,attachments,entities,referenced_tweets,conversation_id",
            ).split(",")
            if s.strip()
        ],
        "X_EXPANSIONS": [
            s.strip()
            for s in env_getter(
                "X_EXPANSIONS",
                "author_id,attachments.media_keys,referenced_tweets.id,referenced_tweets.id.author_id",
            ).split(",")
            if s.strip()
        ],
        "X_MEDIA_FIELDS": [
            s.strip()
            for s in env_getter(
                "X_MEDIA_FIELDS",
                "media_key,type,url,preview_image_url,variants,width,height,alt_text,public_metrics",
            ).split(",")
            if s.strip()
        ],
        "X_USER_FIELDS": [s.strip() for s in env_getter("X_USER_FIELDS", "id,name,username,profile_image_url,verified,protected").split(",") if s.strip()],
        "X_POLL_FIELDS": [
            s.strip()
            for s in env_getter(
                "X_POLL_FIELDS",
                "id,options,duration_minutes,end_datetime,voting_status",
            ).split(",")
            if s.strip()
        ],
        "X_PLACE_FIELDS": [s.strip() for s in env_getter("X_PLACE_FIELDS", "full_name,id,country_code,geo,name,place_type").split(",") if s.strip()],
        "TWITTER_UNROLL_ENABLED": _parse_bool_str(_clean_env_value(env_getter("TWITTER_UNROLL_ENABLED", None)), True),
        "TWITTER_UNROLL_MAX_TWEETS": _safe_int(env_getter("TWITTER_UNROLL_MAX_TWEETS", None), "30", "TWITTER_UNROLL_MAX_TWEETS"),
        "TWITTER_UNROLL_MAX_CHARS": _safe_int(env_getter("TWITTER_UNROLL_MAX_CHARS", None), "6000", "TWITTER_UNROLL_MAX_CHARS"),
        "TWITTER_UNROLL_TIMEOUT_S": _safe_float(env_getter("TWITTER_UNROLL_TIMEOUT_S", None), "15", "TWITTER_UNROLL_TIMEOUT_S"),
        "STREAMING_ENABLE": _parse_bool_str(_clean_env_value(env_getter("STREAMING_ENABLE", None)), True),
        "STREAMING_EMBED_STYLE": env_getter("STREAMING_EMBED_STYLE", "compact"),
        "STREAMING_TICK_MS": _safe_int(env_getter("STREAMING_TICK_MS", None), "750", "STREAMING_TICK_MS"),
        "STREAMING_MAX_STEPS": _safe_int(env_getter("STREAMING_MAX_STEPS", None), "8", "STREAMING_MAX_STEPS"),
        "STREAMING_ENABLE_TEXT": _parse_bool_str(_clean_env_value(env_getter("STREAMING_ENABLE_TEXT", None)), False),
        "STREAMING_ENABLE_SEARCH": _parse_bool_str(_clean_env_value(env_getter("STREAMING_ENABLE_SEARCH", None)), False),
        "STREAMING_ENABLE_RAG": _parse_bool_str(_clean_env_value(env_getter("STREAMING_ENABLE_RAG", None)), False),
        "STREAMING_ENABLE_MEDIA": _parse_bool_str(_clean_env_value(env_getter("STREAMING_ENABLE_MEDIA", None)), True),
        "STT_ENABLE": _parse_bool_str(_clean_env_value(env_getter("STT_ENABLE", None)), True),
        "VISION_ENABLED": _ve,
        "VISION_T2I_ENABLED": _t2i,
        "VISION_REPLY_IMAGE_FORCE_VL": _parse_bool_str(_clean_env_value(env_getter("VISION_REPLY_IMAGE_FORCE_VL", None)), True),
        "VISION_REPLY_IMAGE_SILENT": _parse_bool_str(_clean_env_value(env_getter("VISION_REPLY_IMAGE_SILENT", None)), True),
        "HYBRID_FORCE_PERCEPTION_ON_REPLY": _parse_bool_str(_clean_env_value(env_getter("HYBRID_FORCE_PERCEPTION_ON_REPLY", None)), True),
        "VL_REPLY_MAX_CHARS": _safe_int(env_getter("VL_REPLY_MAX_CHARS", None), "420", "VL_REPLY_MAX_CHARS"),
        "VL_STRIP_REASONING": _parse_bool_str(_clean_env_value(env_getter("VL_STRIP_REASONING", None)), True),
        "VL_NOTES_MAX_CHARS": _safe_int(env_getter("VL_NOTES_MAX_CHARS", None), "600", "VL_NOTES_MAX_CHARS"),
        "TEXT_FINAL_MAX_CHARS": _safe_int(env_getter("TEXT_FINAL_MAX_CHARS", None), "420", "TEXT_FINAL_MAX_CHARS"),
        "VISION_API_KEY": _clean_env_value(env_getter("VISION_API_KEY", None)),
        "VISION_ALLOWED_PROVIDERS": [s.strip() for s in env_getter("VISION_ALLOWED_PROVIDERS", "together,novita").split(",") if s.strip()],
        "VISION_DEFAULT_PROVIDER": env_getter("VISION_DEFAULT_PROVIDER", "together"),
        "VISION_MODEL": _clean_env_value(env_getter("VISION_MODEL", None)) or "",
        "VISION_IMAGE_FALLBACK_MODELS": _clean_env_value(env_getter("VISION_IMAGE_FALLBACK_MODELS", None) or env_getter("IMAGE_FALLBACK_MODELS", None)) or "",
        "VISION_POLICY_PATH": env_getter("VISION_POLICY_PATH", "configs/vision_policy.json"),
        "VISION_DATA_DIR": Path(env_getter("VISION_DATA_DIR", "vision_data")),
        "VISION_ARTIFACTS_DIR": Path(env_getter("VISION_ARTIFACTS_DIR", "vision_data/artifacts")),
        "VISION_JOBS_DIR": Path(env_getter("VISION_JOBS_DIR", "vision_data/jobs")),
        "VISION_LEDGER_PATH": env_getter("VISION_LEDGER_PATH", "vision_data/ledger.jsonl"),
        "VISION_INTENT_THRESHOLD": _safe_float(env_getter("VISION_INTENT_THRESHOLD", None), "0.7", "VISION_INTENT_THRESHOLD"),
        "VISION_FORCE_OPENROUTER_THRESHOLD": _safe_float(
            env_getter("VISION_FORCE_OPENROUTER_THRESHOLD", None),
            "0.3",
            "VISION_FORCE_OPENROUTER_THRESHOLD",
        ),
        "VISION_MAX_CONCURRENT_JOBS": _low_resource_int("VISION_MAX_CONCURRENT_JOBS", 3, 1),
        "VISION_MAX_USER_CONCURRENT_JOBS": _safe_int(
            env_getter("VISION_MAX_USER_CONCURRENT_JOBS", None),
            "1",
            "VISION_MAX_USER_CONCURRENT_JOBS",
        ),
        "VISION_JOB_TIMEOUT_SECONDS": _safe_int(env_getter("VISION_JOB_TIMEOUT_SECONDS", None), "300", "VISION_JOB_TIMEOUT_SECONDS"),
        "VISION_ARTIFACT_TTL_DAYS": _safe_int(env_getter("VISION_ARTIFACT_TTL_DAYS", None), "7", "VISION_ARTIFACT_TTL_DAYS"),
        "VISION_MAX_ARTIFACT_SIZE_MB": _safe_int(
            env_getter("VISION_MAX_ARTIFACT_SIZE_MB", None),
            "50",
            "VISION_MAX_ARTIFACT_SIZE_MB",
        ),
        "VISION_MAX_TOTAL_ARTIFACTS_GB": _safe_int(
            env_getter("VISION_MAX_TOTAL_ARTIFACTS_GB", None),
            "10",
            "VISION_MAX_TOTAL_ARTIFACTS_GB",
        ),
        "VISION_LOG_LEVEL": env_getter("VISION_LOG_LEVEL", "INFO"),
        "VISION_AUDIT_ENABLED": _parse_bool_str(_clean_env_value(env_getter("VISION_AUDIT_ENABLED", None)), True),
        "VISION_PROVIDER_TIMEOUT_MS": _safe_int(
            env_getter("VISION_PROVIDER_TIMEOUT_MS", None),
            "30000",
            "VISION_PROVIDER_TIMEOUT_MS",
        ),
        "VISION_PROVIDER_MAX_RETRIES": _safe_int(env_getter("VISION_PROVIDER_MAX_RETRIES", None), "3", "VISION_PROVIDER_MAX_RETRIES"),
        "VISION_PROVIDER_RETRY_DELAY_MS": _safe_int(
            env_getter("VISION_PROVIDER_RETRY_DELAY_MS", None),
            "1000",
            "VISION_PROVIDER_RETRY_DELAY_MS",
        ),
        "VL_NOTES_TIMEOUT_S": _safe_float(env_getter("VL_NOTES_TIMEOUT_S", None), "120.0", "VL_NOTES_TIMEOUT_S"),
        "VISION_PER_ITEM_BUDGET": _safe_float(env_getter("VISION_PER_ITEM_BUDGET", None), "120.0", "VISION_PER_ITEM_BUDGET"),
        "VL_REQUEST_TIMEOUT": _safe_float(env_getter("VL_REQUEST_TIMEOUT", None), "30.0", "VL_REQUEST_TIMEOUT"),
        "VISION_PROGRESS_UPDATE_INTERVAL_S": _safe_int(
            env_getter("VISION_PROGRESS_UPDATE_INTERVAL_S", None),
            "10",
            "VISION_PROGRESS_UPDATE_INTERVAL_S",
        ),
        "VISION_EPHEMERAL_RESPONSES": _parse_bool_str(_clean_env_value(env_getter("VISION_EPHEMERAL_RESPONSES", None)), True),
        "VISION_DRY_RUN_MODE": _parse_bool_str(_clean_env_value(env_getter("VISION_DRY_RUN_MODE", None)), False),
        "STT_MODE": env_getter("STT_MODE", "single"),
        "STT_ACTIVE_PROVIDERS": [s.strip() for s in env_getter("STT_ACTIVE_PROVIDERS", "local_whisper").split(",") if s.strip()],
        "STT_CONFIDENCE_MIN": _safe_float(env_getter("STT_CONFIDENCE_MIN", None), "0.0", "STT_CONFIDENCE_MIN"),
        "STT_CACHE_TTL": _safe_int(env_getter("STT_CACHE_TTL", None), "600", "STT_CACHE_TTL"),
        "STT_LOCAL_CONCURRENCY": _low_resource_int("STT_LOCAL_CONCURRENCY", 2, 1),
        "RAG_DISABLE_EAGER_LOAD": _low_resource_bool("RAG_DISABLE_EAGER_LOAD", False, True),
        "TTS_SKIP_WARMUP": _low_resource_bool("TTS_SKIP_WARMUP", False, True),
        "VL_MAX_TOTAL_BYTES": _low_resource_int("VL_MAX_TOTAL_BYTES", 20 * 1024 * 1024, 5 * 1024 * 1024),
        "SCREENSHOT_LOW_RESOURCE_WIDTH": _low_resource_int("SCREENSHOT_LOW_RESOURCE_WIDTH", 1920, 1280),
        "SCREENSHOT_LOW_RESOURCE_HEIGHT": _low_resource_int("SCREENSHOT_LOW_RESOURCE_HEIGHT", 1080, 720),
        "VISION_BUDGET_PARSE_FALLBACK_COST_USD": _safe_float(
            env_getter("VISION_BUDGET_PARSE_FALLBACK_COST_USD", None),
            "0.02",
            "VISION_BUDGET_PARSE_FALLBACK_COST_USD",
        ),
        "VISION_PARSE_FALLBACK_CHARGE": _low_resource_float("VISION_PARSE_FALLBACK_CHARGE", 0.01, 0.005),
        "VISION_LOW_RESOURCE_RETRIES": _low_resource_int("VISION_LOW_RESOURCE_RETRIES", 3, 2),
        "MULTIMODAL_MAX_ITEMS": _low_resource_int("MULTIMODAL_MAX_ITEMS", 5, 2),
        "MULTIMODAL_MAX_TOTAL_BYTES": _low_resource_int("MULTIMODAL_MAX_TOTAL_BYTES", 50 * 1024 * 1024, 10 * 1024 * 1024),
        "MULTIMODAL_CONCURRENCY": _low_resource_int("MULTIMODAL_CONCURRENCY", 3, 1),
        "IMAGE_MAX_DIMENSION": _low_resource_int("IMAGE_MAX_DIMENSION", 2048, 1024),
        "PDF_MAX_PAGES": _low_resource_int("PDF_MAX_PAGES", 20, 5),
        "VIDEO_MAX_DURATION_S": _low_resource_float("VIDEO_MAX_DURATION_S", 300.0, 60.0),
        "TTS_MAX_CHARS": _low_resource_int("TTS_MAX_CHARS", 4000, 2000),
        "TTS_SKIP_LONG_RESPONSES": _low_resource_bool("TTS_SKIP_LONG_RESPONSES", False, True),
        "VL_MAX_IMAGES": _low_resource_int("VL_MAX_IMAGES", 5, 2),
        "VL_MAX_IMAGE_DIMENSION": _low_resource_int("VL_MAX_IMAGE_DIMENSION", 2048, 1024),
        "SCREENSHOT_MAX_BYTES": _low_resource_int("SCREENSHOT_MAX_BYTES", 5 * 1024 * 1024, 1 * 1024 * 1024),
        "STT_MAX_AUDIO_DURATION_S": _low_resource_float("STT_MAX_AUDIO_DURATION_S", 300.0, 60.0),
        "RAG_DOCUMENT_WORKERS": _low_resource_int("RAG_DOCUMENT_WORKERS", 4, 1),
        "HTTP_POOL_MAX_CONNECTIONS": _low_resource_int("HTTP_POOL_MAX_CONNECTIONS", 50, 10),
        "URL_MAX_RESPONSE_BYTES": _low_resource_int("URL_MAX_RESPONSE_BYTES", 500 * 1024, 200 * 1024),
        "HTTP_READ_TIMEOUT_S": _low_resource_float("HTTP_READ_TIMEOUT_S", 30.0, 15.0),
        "CONFIG_WATCH_DEBOUNCE_S": _low_resource_float("CONFIG_WATCH_DEBOUNCE_S", 1.0, 2.0),
        "LOG_RATE_LIMIT_WINDOW_S": _safe_int(env_getter("LOG_RATE_LIMIT_WINDOW_S", None), "60", "LOG_RATE_LIMIT_WINDOW_S"),
        "LOG_MAX_STRING_LENGTH": _low_resource_int("LOG_MAX_STRING_LENGTH", 1000, 300),
        "PLAYWRIGHT_MAX_CONCURRENT": _low_resource_int("PLAYWRIGHT_MAX_CONCURRENT", 3, 1),
        "DISCORD_MESSAGE_CACHE_MAX": _low_resource_int("DISCORD_MESSAGE_CACHE_MAX", 256, 64),
        "TTS_CONCURRENCY": _low_resource_int("TTS_CONCURRENCY", 4, 1),
        "ROUTER_MAX_CONCURRENCY_LIGHT": _low_resource_int("ROUTER_MAX_CONCURRENCY_LIGHT", 8, 4),
        "ROUTER_MAX_CONCURRENCY_NETWORK": _low_resource_int("ROUTER_MAX_CONCURRENCY_NETWORK", 32, 8),
        "ROUTER_MAX_CONCURRENCY_HEAVY": _low_resource_int("ROUTER_MAX_CONCURRENCY_HEAVY", 2, 1),
        # MULTIMODAL STT FALLBACK CONFIGURATION [CA][REH]
        "STT_MULTIMODAL_FALLBACK_ENABLED": _parse_bool_str(_clean_env_value(env_getter("STT_MULTIMODAL_FALLBACK_ENABLED", None)), False),
        "STT_MULTIMODAL_FALLBACK_MODELS": env_getter(
            "STT_MULTIMODAL_FALLBACK_MODELS",
            "openrouter/openai-whisper-large-v3,openrouter/meta-llama-3-8b-instruct:free",
        ),
        "STT_MULTIMODAL_FALLBACK_TIMEOUT_S": _safe_float(
            env_getter("STT_MULTIMODAL_FALLBACK_TIMEOUT_S", None),
            "30.0",
            "STT_MULTIMODAL_FALLBACK_TIMEOUT_S",
        ),
        "STT_MULTIMODAL_FALLBACK_MIN_CONFIDENCE": _safe_float(
            env_getter("STT_MULTIMODAL_FALLBACK_MIN_CONFIDENCE", None),
            "0.5",
            "STT_MULTIMODAL_FALLBACK_MIN_CONFIDENCE",
        ),
        "STT_MULTIMODAL_FALLBACK_MAX_RETRIES": _safe_int(
            env_getter("STT_MULTIMODAL_FALLBACK_MAX_RETRIES", None),
            "1",
            "STT_MULTIMODAL_FALLBACK_MAX_RETRIES",
        ),
        # DASHBOARD SETTINGS [PA]
        "DASHBOARD_ENABLED": _parse_bool_str(_clean_env_value(env_getter("DASHBOARD_ENABLED", None)), False),
        "DASHBOARD_HOST": _clean_env_value(env_getter("DASHBOARD_HOST", "127.0.0.1")),
        "DASHBOARD_PORT": _safe_int(env_getter("DASHBOARD_PORT", None), "8011", "DASHBOARD_PORT"),
        "DASHBOARD_PUBLIC_BIND": _parse_bool_str(_clean_env_value(env_getter("DASHBOARD_PUBLIC_BIND", None)), False),
        "DASHBOARD_AUTH_TOKEN": _clean_env_value(env_getter("DASHBOARD_AUTH_TOKEN", None)),
        "DASHBOARD_SESSION_SECRET": _clean_env_value(env_getter("DASHBOARD_SESSION_SECRET", None)),
        "DASHBOARD_OWNER_IDS": _clean_env_value(env_getter("DASHBOARD_OWNER_IDS", "")),
        "DASHBOARD_RATE_LIMIT_SENDS_PER_MINUTE": _safe_int(
            env_getter("DASHBOARD_RATE_LIMIT_SENDS_PER_MINUTE", None),
            "5",
            "DASHBOARD_RATE_LIMIT_SENDS_PER_MINUTE",
        ),
        "DASHBOARD_AUDIT_DB_PATH": env_getter("DASHBOARD_AUDIT_DB_PATH", "./data/dashboard_audit.db"),
        "DASHBOARD_DM_ARCHIVE_ENABLED": _parse_bool_str(_clean_env_value(env_getter("DASHBOARD_DM_ARCHIVE_ENABLED", None)), True),
        "DASHBOARD_DM_RETENTION_DAYS": _safe_int(
            env_getter("DASHBOARD_DM_RETENTION_DAYS", None),
            "90",
            "DASHBOARD_DM_RETENTION_DAYS",
        ),
        "DASHBOARD_AUDIT_RETENTION_DAYS": _safe_int(
            env_getter("DASHBOARD_AUDIT_RETENTION_DAYS", None),
            "180",
            "DASHBOARD_AUDIT_RETENTION_DAYS",
        ),
        "DASHBOARD_MAX_MESSAGE_CHARS": _safe_int(
            env_getter("DASHBOARD_MAX_MESSAGE_CHARS", None),
            "1800",
            "DASHBOARD_MAX_MESSAGE_CHARS",
        ),
        "DASHBOARD_SHOW_MESSAGE_PREVIEWS": _parse_bool_str(
            _clean_env_value(env_getter("DASHBOARD_SHOW_MESSAGE_PREVIEWS", None)),
            True,
        ),
    }

    # Deprecation warnings for legacy config keys [SFT]
    if env_getter("TEXT_MODEL", None):
        logger.warning("⚠️ TEXT_MODEL is deprecated. Use OPENAI_TEXT_MODEL instead. Support for TEXT_MODEL will be removed in a future release.")
    if env_getter("OPENAI_MODEL", None):
        logger.warning("⚠️ OPENAI_MODEL is deprecated. Use OPENAI_TEXT_MODEL instead. Support for OPENAI_MODEL will be removed in a future release.")

    return config


def load_config() -> dict[str, Any]:
    """Load configuration from environment variables with intelligent caching."""
    global _config_cache, _cache_timestamp, _last_good_config

    # Check if we have a valid cached config (performance optimization)
    current_time = time.time()
    if _config_cache and (current_time - _cache_timestamp) < CACHE_TTL:
        return _config_cache

    # Build config using os.environ
    def os_env_getter(key: str, default: str | None = None) -> str | None:
        return os.getenv(key, default)

    config = _build_config(os_env_getter)

    # One-time startup VISION flags summary [PA]
    try:
        _ve_raw = _clean_env_value(os.getenv("VISION_ENABLED"))
        _t2i_raw = _clean_env_value(os.getenv("VISION_T2I_ENABLED"))
        ve_src = "env" if _ve_raw is not None else "default"
        t2i_src = "env" if _t2i_raw is not None else "default"
        _ve = _parse_bool_str(_ve_raw, True)
        _t2i = _parse_bool_str(_t2i_raw, True)
        logger.info(f"VISION_FLAGS raw={{VISION_ENABLED:{_ve_raw}, VISION_T2I_ENABLED:{_t2i_raw}}} parsed={{vision_enabled:{_ve}, t2i:{_t2i}}} source={{vision_enabled:{ve_src}, t2i:{t2i_src}}}")
    except Exception as e:
        logger.debug(f"Failed to log VISION_FLAGS: {e}")

    # Cache the config for performance (avoid repeated env var lookups)
    _config_cache = config
    _cache_timestamp = current_time
    _last_good_config = config.copy()  # Track last known good config for rollback
    logger.debug(f"✅ Configuration cached for {CACHE_TTL}s")

    return config


def load_config_candidate(env_path: Path) -> dict[str, Any]:
    """Load configuration from a specific .env file WITHOUT mutating global state.

    This is used for hot-reload candidate validation — it reads the .env file
    directly and builds a config dict from its values, without touching
    os.environ, the global cache, or _last_good_config.

    Args:
        env_path: Path to the .env file to load.

    Returns:
        Configuration dictionary built from the candidate .env file.

    """
    from dotenv import dotenv_values

    # Read the candidate .env file directly
    try:
        candidate_env = {str(key): str(value) for key, value in dotenv_values(env_path).items() if key and value is not None}
    except Exception:
        candidate_env = {}

    # Build config using ONLY the candidate env values (NO fallback to os.environ)
    # This ensures validation tests the candidate file in isolation
    def candidate_env_getter(key: str, default: str | None = None) -> str | None:
        return candidate_env.get(key, default)

    return _build_config(candidate_env_getter)


def get_vl_model_ladder() -> list[str]:
    """Return the configured VL model ladder. Supports comma-separated VL_MODEL.
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


def get_last_good_config() -> dict[str, Any] | None:
    """Get the last known good configuration (for hot-reload rollback)."""
    return _last_good_config.copy() if _last_good_config else None


def validate_config_candidate(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a candidate configuration for hot-reload.

    Returns:
        Tuple of (is_valid, missing_required_keys).
        If is_valid is False, missing_required_keys contains the names of
        missing or invalid required variables.

    """
    missing = []

    # Baseline required variables (always required)
    required_vars = ["DISCORD_TOKEN", "PROMPT_FILE", "VL_PROMPT_FILE"]
    for var in required_vars:
        if not config.get(var):
            missing.append(var)

    # Provider-specific validation
    text_backend = config.get("TEXT_BACKEND", "openai").lower()

    if text_backend == "openai":
        # OpenAI/OpenRouter backend requires OPENAI_API_KEY
        if not config.get("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
    elif text_backend == "nvidia":
        # NVIDIA NIM backend requires NVIDIA_NIM_API_KEY (or OPENAI_API_KEY as alias)
        if not config.get("NVIDIA_NIM_API_KEY") and not config.get("OPENAI_API_KEY"):
            missing.append("NVIDIA_NIM_API_KEY")
    elif text_backend == "ollama":
        # Ollama backend does not require OPENAI_API_KEY
        pass
    # Other backends: no additional required keys beyond baseline

    return len(missing) == 0, missing


def invalidate_config_cache() -> None:
    """Invalidate the in-process config cache to force fresh reads on next load_config()."""
    global _config_cache, _cache_timestamp
    _config_cache = None
    _cache_timestamp = 0.0
    with contextlib.suppress(Exception):
        logger.info(
            "config.cache.invalidate",
            extra={
                "event": "config.cache.invalidate",
                "detail": {"reason": "reload_request"},
            },
        )


def get_config_cache() -> dict[str, Any] | None:
    """Get the current config cache (for test introspection)."""
    return _config_cache


def get_cache_timestamp() -> float:
    """Get the cache timestamp (for test introspection)."""
    return _cache_timestamp


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
