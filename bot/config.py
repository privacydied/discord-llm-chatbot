"""
Configuration loading and environment setup.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def validate_required_env() -> None:
    """Validate that all required environment variables are present."""
    required_vars = [
        "DISCORD_TOKEN",
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def load_config():
    """Load configuration from environment variables."""
    return {
        "DISCORD_TOKEN": os.getenv("DISCORD_TOKEN"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4"),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL", "base"),
        "TEMP_DIR": os.getenv("TEMP_DIR", "temp"),
        "MAX_MEMORIES": int(os.getenv("MAX_MEMORIES", "100")),
        "MAX_SERVER_MEMORY": int(os.getenv("MAX_SERVER_MEMORY", "100")),
        "MEMORY_SAVE_INTERVAL": int(os.getenv("MEMORY_SAVE_INTERVAL", "300")),  # 5 minutes
        "USER_PROFILE_DIR": Path(os.getenv("USER_PROFILE_DIR", "user_profiles")),
        "SERVER_PROFILE_DIR": Path(os.getenv("SERVER_PROFILE_DIR", "server_profiles")),
        "USER_LOGS_DIR": Path(os.getenv("USER_LOGS_DIR", "user_logs")),
        "DEFAULT_TIMEOUT": int(os.getenv("DEFAULT_TIMEOUT", "30")),  # seconds
        "MAX_CONTEXT_LENGTH": int(os.getenv("MAX_CONTEXT_LENGTH", "4000")),  # tokens
        "MAX_RESPONSE_TOKENS": int(os.getenv("MAX_RESPONSE_TOKENS", "1000")),
        "TEMPERATURE": float(os.getenv("TEMPERATURE", "0.7")),
        "TOP_P": float(os.getenv("TOP_P", "0.9")),
        "FREQUENCY_PENALTY": float(os.getenv("FREQUENCY_PENALTY", "0.0")),
        "PRESENCE_PENALTY": float(os.getenv("PRESENCE_PENALTY", "0.0")),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "COMMAND_PREFIX": os.getenv("COMMAND_PREFIX", "!"),
        "OWNER_IDS": [int(id.strip()) for id in os.getenv("OWNER_IDS", "").split(",") if id.strip()],
    }
