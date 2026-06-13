"""Multimodal STT fallback module for the Discord bot.

This module provides:
- Failure classification for multimodal fallback (failure_classifier.py)
- Multimodal fallback provider (multimodal_fallback.py)
"""

from .failure_classifier import FailureClassification, STTFailureClassifier
from .multimodal_fallback import FallbackTranscriptResult, multimodal_fallback_provider

__all__ = [
    "FailureClassification",
    "FallbackTranscriptResult",
    "STTFailureClassifier",
    "multimodal_fallback_provider",
]
