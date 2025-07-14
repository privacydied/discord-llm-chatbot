"""
Discord LLM Chatbot Package

A production-ready Discord bot with AI capabilities, featuring:
- Multi-backend AI support (OpenAI, Ollama)
- Text-to-Speech (TTS) integration
- Conversation memory management
- Web content extraction
- PDF processing
- Comprehensive logging and error handling
"""

# Package metadata
__title__ = "Discord LLM Chatbot"
__version__ = "0.1.0"
__author__ = "Discord Bot Developer"
__description__ = "Production-ready Discord bot with AI capabilities"
__url__ = "https://github.com/user/discord-llm-chatbot"
__license__ = "MIT"

# Core package components - exposed for external use
from .config import load_config, validate_required_env, ConfigurationError
from .logger import setup_logging, get_logger
from .memory import (
    get_profile, save_profile, 
    get_server_profile, save_server_profile,
    add_memory, load_all_profiles, save_all_profiles
)
from .utils import send_chunks, download_file, is_text_file
from .main import LLMBot

# Package-level exports
__all__ = [
    # Core classes
    'LLMBot',
    'ConfigurationError',
    
    # Configuration
    'load_config',
    'validate_required_env',
    
    # Logging
    'setup_logging',
    'get_logger',
    
    # Memory management
    'get_profile',
    'save_profile',
    'get_server_profile', 
    'save_server_profile',
    'add_memory',
    'load_all_profiles',
    'save_all_profiles',
    
    # Utilities
    'send_chunks',
    'download_file',
    'is_text_file',
    
    # Package metadata
    '__version__',
    '__title__',
    '__author__',
    '__description__',
    '__url__',
    '__license__',
]
