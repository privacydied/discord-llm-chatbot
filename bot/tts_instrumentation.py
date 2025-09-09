"""
Compatibility shim for legacy imports.

Legacy path: bot.tts_instrumentation
New path:   bot.tts.instrumentation

This module re-exports the instrumentation utilities for backward compatibility
with existing tests and any code still importing the legacy path.
"""

from .tts.instrumentation import (
    log_tts_config,
    log_phonemiser_selection,
    log_voice_loading,
    log_tts_generation,
    log_tts_error,
    log_gibberish_detection,
    log_cache_event,
    timed_function,
    get_tts_metrics,
    reset_tts_metrics,
)

__all__ = [
    "log_tts_config",
    "log_phonemiser_selection",
    "log_voice_loading",
    "log_tts_generation",
    "log_tts_error",
    "log_gibberish_detection",
    "log_cache_event",
    "timed_function",
    "get_tts_metrics",
    "reset_tts_metrics",
]
