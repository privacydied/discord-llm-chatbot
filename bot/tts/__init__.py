"""
TTS (Text-to-Speech) package for the Discord bot.

This package contains modules for TTS functionality, including
the kokoro_bootstrap module for registering the EspeakWrapper tokenizer.
"""

from .kokoro_bootstrap import TOKENIZER_ALIASES
from .interface import TTSManager

__all__ = ['TTSManager', 'TOKENIZER_ALIASES']
