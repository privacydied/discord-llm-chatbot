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

# Avoid importing heavy submodules at package import time to keep tests lightweight
__all__ = []

def __getattr__(name: str):
    """Lazy loader for heavy symbols to prevent side effects during tests.

    Accessing bot.LLMBot will import it on demand, otherwise importing
    submodules like bot.commands.* won't pull the entire runtime.
    """
    if name == "LLMBot":
        from .core.bot import LLMBot as _LLMBot
        return _LLMBot
    raise AttributeError(name)
