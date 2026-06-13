"""STT Failure Classification System.

Categorizes transcription failures to determine when multimodal fallback should be attempted.
Provides clear failure classification for debugging and fallback decision logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bot.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


@dataclass
class FailureClassification:
    """Categorize STT failures for precise fallback handling."""

    category: str  # "extraction", "decode", "zero_length", "whisper_runtime", "corrupted", "timeout", "memory"
    severity: str  # "soft", "hard"
    recoverable: bool
    detail: str
    confidence_score: float = 0.0  # 0.0-1.0 confidence in classification

    def __str__(self) -> str:
        return f"{self.category}({self.severity}, {'recoverable' if self.recoverable else 'unrecoverable'}): {self.detail}"


class STTFailureClassifier:
    """Classifies STT failures to determine fallback eligibility."""

    # Patterns for failure detection
    ERROR_PATTERNS = {
        "extraction": [
            r"file not found",
            r"no such file",
            r"download failed",
            r"youtube.*error",
            r"connection.*timeout",
            r"network.*error",
        ],
        "decode": [
            r"invalid.*format",
            r"unsupported.*codec",
            r"corrupted.*file",
            r"demuxer.*error",
            r"decoder.*error",
            r"could not find tag",
        ],
        "zero_length": [
            r"zero.*duration",
            r"empty.*audio",
            r"no.*audio",
            r"silence.*detected",
            r"length.*zero",
        ],
        "whisper_runtime": [
            r"runtime error",
            r"out of memory",
            r"cpu.*threads",
            r"torch.*error",
            r"model.*load",
            r"computation.*error",
        ],
        "timeout": [r"timeout", r"deadline exceeded", r"timed out", r"operation timed"],
        "memory": [
            r"memory.*exceeded",
            r"out of memory",
            r"ram.*guard",
            r"memory.*limit",
        ],
        "corrupted": [
            r"corrupted.*data",
            r"invalid.*data",
            r"broken.*file",
            r"damaged.*file",
        ],
    }

    @classmethod
    def classify_failure(
        cls,
        error: Exception,
        pre_result: Any | None = None,
        audio_path: Path | None = None,
    ) -> FailureClassification:
        """Classify a STT failure to determine if multimodal fallback should be attempted.

        Args:
            error: The exception that caused the failure
            pre_result: Optional preprocessing result (for audio analysis)
            audio_path: Path to the audio file (for file inspection)

        Returns:
            FailureClassification object with detailed categorization

        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check file-based classifications first
        if audio_path and audio_path.exists():
            file_classification = cls._classify_file_failure(audio_path)
            if file_classification:
                return file_classification

        # Check error message patterns
        for category, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_str) or re.search(pattern, error_type):
                    return cls._create_classification(category, error, error_str)

        # Default to whisper_runtime for unknown errors
        return cls._create_classification("whisper_runtime", error, error_str)

    @classmethod
    def _classify_file_failure(cls, audio_path: Path) -> FailureClassification | None:
        """Classify failures based on file inspection."""
        try:
            # Check file size
            file_size = audio_path.stat().st_size

            if file_size == 0:
                return FailureClassification(
                    category="extraction",
                    severity="soft",
                    recoverable=False,
                    detail="Empty audio file",
                    confidence_score=1.0,
                )

            # Check if it's a valid audio file OR container format that can contain audio [REH]
            # Container formats (.mp4, .mov, etc.) are allowed because the extraction stage
            # can convert them to PCM. Rejecting them here would prevent the Tier 2 retry.
            suffix = audio_path.suffix.lower()
            allowed_formats = [
                ".wav",
                ".mp3",
                ".m4a",
                ".ogg",
                ".flac",
                ".webm",
                # Container formats that can contain audio - extraction will handle conversion
                ".mp4",
                ".mp4a",
                ".mov",
                ".avi",
                ".mkv",
                ".ts",
            ]
            if suffix not in allowed_formats:
                return FailureClassification(
                    category="decode",
                    severity="hard",
                    recoverable=False,
                    detail=f"Unsupported audio format: {audio_path.suffix}",
                    confidence_score=0.9,
                )

            # For small files, check if they might be corrupted
            if file_size < 1024:  # Less than 1KB
                return FailureClassification(
                    category="corrupted",
                    severity="soft",
                    recoverable=True,
                    detail=f"Very small audio file ({file_size} bytes)",
                    confidence_score=0.7,
                )

        except Exception as e:
            return FailureClassification(
                category="extraction",
                severity="hard",
                recoverable=False,
                detail=f"File inspection failed: {e}",
                confidence_score=0.5,
            )

        return None

    @classmethod
    def _create_classification(cls, category: str, error: Exception, error_str: str) -> FailureClassification:
        """Create a failure classification with appropriate severity and recoverability."""
        # Determine severity and recoverability based on category
        if category in ["extraction", "timeout", "memory"]:
            severity = "soft"
            recoverable = True
        elif category in ["decode", "corrupted"]:
            severity = "soft" if "corrupted" in category else "hard"
            recoverable = category == "corrupted"  # Corrupted might be recoverable with multimodal
        else:  # whisper_runtime, zero_length
            severity = "soft"
            recoverable = True

        return FailureClassification(
            category=category,
            severity=severity,
            recoverable=recoverable,
            detail=f"{error.__class__.__name__}: {error!s}",
            confidence_score=cls._calculate_confidence(category, error_str),
        )

    @classmethod
    def _calculate_confidence(cls, category: str, error_str: str) -> float:
        """Calculate confidence score for the classification."""
        base_confidence = {
            "extraction": 0.8,
            "decode": 0.9,
            "zero_length": 0.95,
            "whisper_runtime": 0.6,
            "timeout": 0.8,
            "memory": 0.7,
            "corrupted": 0.7,
        }.get(category, 0.5)

        # Boost confidence if we see multiple matching patterns
        pattern_matches = sum(1 for patterns in cls.ERROR_PATTERNS.values() for pattern in patterns if re.search(pattern, error_str, re.IGNORECASE))

        if pattern_matches > 1:
            base_confidence = min(1.0, base_confidence + 0.2)

        return base_confidence

    @classmethod
    def should_attempt_fallback(
        cls,
        classification: FailureClassification,
        has_audio_data: bool = True,
        pre_duration: float = 0.0,
    ) -> bool:
        """Determine if multimodal fallback should be attempted based on failure classification.

        Args:
            classification: The classified failure
            has_audio_data: Whether audio data is available for processing
            pre_duration: Duration of preprocessed audio (for zero-length checks)

        Returns:
            True if fallback should be attempted, False otherwise

        """
        # Never attempt fallback for these categories
        if classification.category == "extraction" and classification.severity == "hard":
            return False

        # Never attempt fallback if no audio data available
        if not has_audio_data:
            return False

        # For zero-length audio, only fallback if we have some duration
        if classification.category == "zero_length" and pre_duration <= 0.1:
            return False

        # Never attempt fallback for unrecoverable failures
        if not classification.recoverable:
            return False

        # Always attempt fallback for these categories
        if classification.category in ["decode", "corrupted", "timeout"]:
            return True

        # For whisper_runtime and memory errors, attempt fallback with caution
        if classification.category in ["whisper_runtime", "memory"]:
            return classification.severity == "soft"

        return False
