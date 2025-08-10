"""
TTS (Text-to-Speech) package for the Discord bot.

This package contains modules for TTS functionality, including
the kokoro_bootstrap module for registering the EspeakWrapper tokenizer.
"""

from .kokoro_bootstrap import TOKENIZER_ALIASES
from .interface import TTSManager

# Minimal placeholder class to support tests that patch `bot.tts.TTS`.
class TTS:  # pragma: no cover - test helper symbol
    pass

__all__ = ['TTSManager', 'TOKENIZER_ALIASES', 'TTS']
