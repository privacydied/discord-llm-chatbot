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
        
        # SILENCE GATE - SPEAK ONLY WHEN SPOKEN TO
        "BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO": os.getenv("BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO", "True").lower() == "true",
        # Comma-separated list of triggers: dm, mention, reply, bot_threads, owner, command_prefix
        "REPLY_TRIGGERS": [s.strip() for s in os.getenv(
            "REPLY_TRIGGERS",
            "dm,mention,reply,bot_threads,owner,command_prefix"
        ).split(",") if s.strip()],
        
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
        # Native Discord voice messages toggle [CMV]
        "VOICE_ENABLE_NATIVE": os.getenv("VOICE_ENABLE_NATIVE", "false").lower() == "true",
        
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

        # ADMIN ALERT SYSTEM [CA][CMV]
        # Keep as strings to preserve existing cog parsing semantics
        # - ALERT_ENABLE checked via .lower() == 'true'
        # - ALERT_SESSION_TIMEOUT_S cast to int in cog
        # - ALERT_ADMIN_USER_IDS parsed as comma-separated ints in cog
        "ALERT_ENABLE": _clean_env_value(os.getenv("ALERT_ENABLE", "false")),
        "ALERT_SESSION_TIMEOUT_S": _clean_env_value(os.getenv("ALERT_SESSION_TIMEOUT_S", "1800")),
        "ALERT_ADMIN_USER_IDS": _clean_env_value(os.getenv("ALERT_ADMIN_USER_IDS", "")),

        # SEARCH SUBSYSTEM [CA][CMV][IV]
        # Provider selection: 'ddg' (default) or 'custom'
        "SEARCH_PROVIDER": os.getenv("SEARCH_PROVIDER", "ddg").lower(),
        # Default knobs
        "SEARCH_MAX_RESULTS": _safe_int(os.getenv("SEARCH_MAX_RESULTS"), "5", "SEARCH_MAX_RESULTS"),
        "SEARCH_SAFE": os.getenv("SEARCH_SAFE", "moderate").lower(),  # off|moderate|strict
        "SEARCH_LOCALE": os.getenv("SEARCH_LOCALE", ""),
        # DuckDuckGo provider options (DDG typically requires no API key; kept for pluggability)
        "DDG_API_ENDPOINT": os.getenv("DDG_API_ENDPOINT", "https://html.duckduckgo.com/html/"),
        # Force legacy HTML endpoint instead of ddgs client. [CMV]
        "DDG_FORCE_HTML": os.getenv("DDG_FORCE_HTML", "true").lower() == "true",
        "DDG_API_KEY": os.getenv("DDG_API_KEY"),
        "DDG_TIMEOUT_MS": _safe_int(os.getenv("DDG_TIMEOUT_MS"), "5000", "DDG_TIMEOUT_MS"),
        # Custom provider HTTP options
        "CUSTOM_SEARCH_API_ENDPOINT": os.getenv("CUSTOM_SEARCH_API_ENDPOINT", ""),
        "CUSTOM_SEARCH_API_KEY": os.getenv("CUSTOM_SEARCH_API_KEY", ""),
        # Optional JSON headers, comma-separated key:value pairs
        "CUSTOM_SEARCH_HEADERS": os.getenv("CUSTOM_SEARCH_HEADERS", ""),
        "CUSTOM_SEARCH_TIMEOUT_MS": _safe_int(os.getenv("CUSTOM_SEARCH_TIMEOUT_MS"), "8000", "CUSTOM_SEARCH_TIMEOUT_MS"),
        # Optional JSONPath-like comma-separated selectors for result extraction
        "CUSTOM_SEARCH_RESULT_PATHS": os.getenv("CUSTOM_SEARCH_RESULT_PATHS", ""),
        # Shared HTTP pool
        "SEARCH_POOL_MAX_CONNECTIONS": _safe_int(os.getenv("SEARCH_POOL_MAX_CONNECTIONS"), "10", "SEARCH_POOL_MAX_CONNECTIONS"),
        # Circuit breaker (search)
        "SEARCH_BREAKER_FAILURE_WINDOW": _safe_int(os.getenv("SEARCH_BREAKER_FAILURE_WINDOW"), "5", "SEARCH_BREAKER_FAILURE_WINDOW"),
        "SEARCH_BREAKER_OPEN_MS": _safe_int(os.getenv("SEARCH_BREAKER_OPEN_MS"), "15000", "SEARCH_BREAKER_OPEN_MS"),
        "SEARCH_BREAKER_HALFOPEN_PROB": _safe_float(os.getenv("SEARCH_BREAKER_HALFOPEN_PROB"), "0.25", "SEARCH_BREAKER_HALFOPEN_PROB"),

        # X (Twitter) API Integration [CA][CMV][SFT]
        # Feature flag and auth
        "X_API_ENABLED": os.getenv("X_API_ENABLED", "false").lower() == "true",
        "X_API_AUTH_MODE": os.getenv("X_API_AUTH_MODE", "oauth2_app"),
        "X_API_BEARER_TOKEN": _clean_env_value(os.getenv("X_API_BEARER_TOKEN")),  # never log token

        # Fallback rules
        "X_API_REQUIRE_API_FOR_TWITTER": os.getenv("X_API_REQUIRE_API_FOR_TWITTER", "false").lower() == "true",
        "X_API_ALLOW_FALLBACK_ON_5XX": os.getenv("X_API_ALLOW_FALLBACK_ON_5XX", "true").lower() == "true",

        # X Syndication Tier [CMV]
        # Hardcoded default: enabled unless explicitly disabled
        "X_SYNDICATION_ENABLED": os.getenv("X_SYNDICATION_ENABLED", "true").lower() == "true",

        # Fast probe: attempt STT on X URLs before API/syndication [CMV][PA]
        "X_TWITTER_STT_PROBE_FIRST": os.getenv("X_TWITTER_STT_PROBE_FIRST", "true").lower() == "true",

        # Routing: enable photo media to VL analysis path [CMV]
        # Hardcoded default: enabled (route photos to VL)
        "X_API_ROUTE_PHOTOS_TO_VL": os.getenv("X_API_ROUTE_PHOTOS_TO_VL", "true").lower() == "true",

        # Networking and resilience knobs
        "X_API_TIMEOUT_MS": _safe_int(os.getenv("X_API_TIMEOUT_MS"), "8000", "X_API_TIMEOUT_MS"),
        "X_API_RETRY_MAX_ATTEMPTS": _safe_int(os.getenv("X_API_RETRY_MAX_ATTEMPTS"), "5", "X_API_RETRY_MAX_ATTEMPTS"),
        "X_API_BREAKER_FAILURE_WINDOW": _safe_int(os.getenv("X_API_BREAKER_FAILURE_WINDOW"), "5", "X_API_BREAKER_FAILURE_WINDOW"),
        "X_API_BREAKER_OPEN_MS": _safe_int(os.getenv("X_API_BREAKER_OPEN_MS"), "15000", "X_API_BREAKER_OPEN_MS"),
        "X_API_BREAKER_HALFOPEN_PROB": _safe_float(os.getenv("X_API_BREAKER_HALFOPEN_PROB"), "0.25", "X_API_BREAKER_HALFOPEN_PROB"),

        # Field hydration (comma-separated lists) [CMV]
        "X_TWEET_FIELDS": [s.strip() for s in os.getenv(
            "X_TWEET_FIELDS",
            "id,text,created_at,author_id,public_metrics,possibly_sensitive,lang,attachments,entities,referenced_tweets,conversation_id"
        ).split(",") if s.strip()],
        "X_EXPANSIONS": [s.strip() for s in os.getenv(
            "X_EXPANSIONS",
            "author_id,attachments.media_keys,referenced_tweets.id,referenced_tweets.id.author_id"
        ).split(",") if s.strip()],
        "X_MEDIA_FIELDS": [s.strip() for s in os.getenv(
            "X_MEDIA_FIELDS",
            "media_key,type,url,preview_image_url,variants,width,height,alt_text,public_metrics"
        ).split(",") if s.strip()],
        "X_USER_FIELDS": [s.strip() for s in os.getenv(
            "X_USER_FIELDS",
            "id,name,username,profile_image_url,verified,protected"
        ).split(",") if s.strip()],
        "X_POLL_FIELDS": [s.strip() for s in os.getenv(
            "X_POLL_FIELDS",
            "id,options,duration_minutes,end_datetime,voting_status"
        ).split(",") if s.strip()],
        "X_PLACE_FIELDS": [s.strip() for s in os.getenv(
            "X_PLACE_FIELDS",
            "full_name,id,country_code,geo,name,place_type"
        ).split(",") if s.strip()],

        # Routing defaults
        "TWITTER_ROUTE_DEFAULT": os.getenv("TWITTER_ROUTE_DEFAULT", "api_first"),

        # STREAMING STATUS CARDS [CA][CMV]
        # Global enable for streaming card UX (text-only remains non-streaming)
        "STREAMING_ENABLE": os.getenv("STREAMING_ENABLE", "true").lower() == "true",
        # Style preset: 'compact' | 'detailed'
        "STREAMING_EMBED_STYLE": os.getenv("STREAMING_EMBED_STYLE", "compact"),
        # Edit throttle and max step count
        "STREAMING_TICK_MS": _safe_int(os.getenv("STREAMING_TICK_MS"), "750", "STREAMING_TICK_MS"),
        "STREAMING_MAX_STEPS": _safe_int(os.getenv("STREAMING_MAX_STEPS"), "8", "STREAMING_MAX_STEPS"),
        # Domain-specific eligibility gates [CMV]
        # Defaults: text/search/rag disabled, media enabled
        "STREAMING_ENABLE_TEXT": os.getenv("STREAMING_ENABLE_TEXT", "false").lower() == "true",
        "STREAMING_ENABLE_SEARCH": os.getenv("STREAMING_ENABLE_SEARCH", "false").lower() == "true",
        "STREAMING_ENABLE_RAG": os.getenv("STREAMING_ENABLE_RAG", "false").lower() == "true",
        "STREAMING_ENABLE_MEDIA": os.getenv("STREAMING_ENABLE_MEDIA", "true").lower() == "true",

        # STT ORCHESTRATION [CA][CMV] =====
        # Global toggle for STT orchestrator (falls back to legacy path when disabled)
        "STT_ENABLE": os.getenv("STT_ENABLE", "true").lower() == "true",

        # ===== VISION GENERATION SYSTEM [CA][CMV][SFT][REH] =====
        # Master toggle for entire Vision generation feature set
        "VISION_ENABLED": os.getenv("VISION_ENABLED", "false").lower() == "true",
        # Single credential for Vision Gateway (provider secrets handled behind gateway)
        "VISION_API_KEY": _clean_env_value(os.getenv("VISION_API_KEY")),
        # Provider configuration
        "VISION_ALLOWED_PROVIDERS": [s.strip() for s in os.getenv("VISION_ALLOWED_PROVIDERS", "together,novita").split(",") if s.strip()],
        "VISION_DEFAULT_PROVIDER": os.getenv("VISION_DEFAULT_PROVIDER", "together"),
        # Policy and data paths
        "VISION_POLICY_PATH": os.getenv("VISION_POLICY_PATH", "configs/vision_policy.json"),
        "VISION_DATA_DIR": Path(os.getenv("VISION_DATA_DIR", "vision_data")),
        "VISION_ARTIFACTS_DIR": Path(os.getenv("VISION_ARTIFACTS_DIR", "vision_data/artifacts")),
        "VISION_JOBS_DIR": Path(os.getenv("VISION_JOBS_DIR", "vision_data/jobs")),
        "VISION_LEDGER_PATH": os.getenv("VISION_LEDGER_PATH", "vision_data/ledger.jsonl"),
        # Intent routing thresholds
        "VISION_INTENT_THRESHOLD": _safe_float(os.getenv("VISION_INTENT_THRESHOLD"), "0.7", "VISION_INTENT_THRESHOLD"),
        "VISION_FORCE_OPENROUTER_THRESHOLD": _safe_float(os.getenv("VISION_FORCE_OPENROUTER_THRESHOLD"), "0.3", "VISION_FORCE_OPENROUTER_THRESHOLD"),
        # Concurrency and performance limits
        "VISION_MAX_CONCURRENT_JOBS": _safe_int(os.getenv("VISION_MAX_CONCURRENT_JOBS"), "3", "VISION_MAX_CONCURRENT_JOBS"),
        "VISION_MAX_USER_CONCURRENT_JOBS": _safe_int(os.getenv("VISION_MAX_USER_CONCURRENT_JOBS"), "1", "VISION_MAX_USER_CONCURRENT_JOBS"),
        "VISION_JOB_TIMEOUT_SECONDS": _safe_int(os.getenv("VISION_JOB_TIMEOUT_SECONDS"), "300", "VISION_JOB_TIMEOUT_SECONDS"),
        # Artifact management
        "VISION_ARTIFACT_TTL_DAYS": _safe_int(os.getenv("VISION_ARTIFACT_TTL_DAYS"), "7", "VISION_ARTIFACT_TTL_DAYS"),
        "VISION_MAX_ARTIFACT_SIZE_MB": _safe_int(os.getenv("VISION_MAX_ARTIFACT_SIZE_MB"), "50", "VISION_MAX_ARTIFACT_SIZE_MB"),
        "VISION_MAX_TOTAL_ARTIFACTS_GB": _safe_int(os.getenv("VISION_MAX_TOTAL_ARTIFACTS_GB"), "10", "VISION_MAX_TOTAL_ARTIFACTS_GB"),
        # Logging and observability
        "VISION_LOG_LEVEL": os.getenv("VISION_LOG_LEVEL", "INFO"),
        "VISION_AUDIT_ENABLED": os.getenv("VISION_AUDIT_ENABLED", "true").lower() == "true",
        # Provider-specific timeouts and retries
        "VISION_PROVIDER_TIMEOUT_MS": _safe_int(os.getenv("VISION_PROVIDER_TIMEOUT_MS"), "30000", "VISION_PROVIDER_TIMEOUT_MS"),
        "VISION_PROVIDER_MAX_RETRIES": _safe_int(os.getenv("VISION_PROVIDER_MAX_RETRIES"), "3", "VISION_PROVIDER_MAX_RETRIES"),
        "VISION_PROVIDER_RETRY_DELAY_MS": _safe_int(os.getenv("VISION_PROVIDER_RETRY_DELAY_MS"), "1000", "VISION_PROVIDER_RETRY_DELAY_MS"),
        # Discord integration
        "VISION_PROGRESS_UPDATE_INTERVAL_S": _safe_int(os.getenv("VISION_PROGRESS_UPDATE_INTERVAL_S"), "10", "VISION_PROGRESS_UPDATE_INTERVAL_S"),
        "VISION_EPHEMERAL_RESPONSES": os.getenv("VISION_EPHEMERAL_RESPONSES", "true").lower() == "true",
        # Dry run mode for testing routing and cost decisions
        "VISION_DRY_RUN_MODE": os.getenv("VISION_DRY_RUN_MODE", "false").lower() == "true",
        # Orchestration mode: single | cascade_primary_then_fallbacks | parallel_first_acceptable | parallel_best_of | hybrid_draft_then_finalize
        "STT_MODE": os.getenv("STT_MODE", "single"),
        # Active providers (comma-separated). Supported now: local_whisper
        "STT_ACTIVE_PROVIDERS": [s.strip() for s in os.getenv("STT_ACTIVE_PROVIDERS", "local_whisper").split(",") if s.strip()],
        # Minimum confidence to accept result (providers lacking confidence are always acceptable)
        "STT_CONFIDENCE_MIN": _safe_float(os.getenv("STT_CONFIDENCE_MIN"), "0.0", "STT_CONFIDENCE_MIN"),
        # Cache TTL for successful transcripts (seconds)
        "STT_CACHE_TTL": _safe_int(os.getenv("STT_CACHE_TTL"), "600", "STT_CACHE_TTL"),
        # Local provider concurrency controls
        "STT_LOCAL_CONCURRENCY": _safe_int(os.getenv("STT_LOCAL_CONCURRENCY"), "2", "STT_LOCAL_CONCURRENCY"),
    }
    
    # Cache the config for performance (avoid repeated env var lookups)
    _config_cache = config
    _cache_timestamp = current_time
    logger.debug(f"✅ Configuration cached for {CACHE_TTL}s")
    
    return config


# Force English IPA route (bypass tokenizer env and disable autodiscovery)
KOKORO_FORCE_IPA_EN = True
