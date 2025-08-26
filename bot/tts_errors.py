"""
Compatibility shim for legacy imports.

Legacy path: bot.tts_errors
New path:   bot.tts.errors

This module re-exports the error classes for backward compatibility with
existing tests and any code still importing the legacy path.
"""
from .tts.errors import *  # noqa: F401,F403
