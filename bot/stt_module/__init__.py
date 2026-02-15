"""
Multimodal STT fallback module for the Discord bot.

This module provides:
- Failure classification for multimodal fallback (failure_classifier.py)
- Multimodal fallback provider (multimodal_fallback.py)
"""

from .failure_classifier import STTFailureClassifier, FailureClassification
from .multimodal_fallback import multimodal_fallback_provider, FallbackTranscriptResult

__all__ = [
    "STTFailureClassifier",
    "FailureClassification",
    "multimodal_fallback_provider",
    "FallbackTranscriptResult",
]
