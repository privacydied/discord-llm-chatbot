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
__author__ = "Discord Bot Developer"
__description__ = "Production-ready Discord bot with AI capabilities"
__url__ = "https://github.com/user/discord-llm-chatbot"
__license__ = "MIT"

# Core package components - exposed for external use
from .core.bot import LLMBot

__all__ = ["LLMBot"]
