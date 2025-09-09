"""Instrumentation utilities for TTS monitoring and debugging."""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants for instrumentation
TTS_METRICS = {
    "tts_generation_count": 0,
    "tts_generation_time_total": 0.0,
    "tts_generation_errors": 0,
    "tts_gibberish_detected": 0,
    "tts_cache_hits": 0,
    "tts_cache_misses": 0,
}

# Tracking for warning messages
TTS_WARNINGS_SHOWN = {
    "ocr_missing": False,
    "tokenizer_missing": False,
    "voice_mismatch": False,
    "sample_rate_mismatch": False,
}


def log_tts_config(config: Dict[str, Any]) -> None:
    """
    Log the TTS configuration at startup.

    Args:
        config: Dictionary containing TTS configuration
    """
    # Filter out sensitive information
    safe_config = {
        k: v
        for k, v in config.items()
        if not k.lower().endswith(("key", "secret", "password", "token"))
    }

    # Log the configuration
    logger.info(
        f"TTS configuration loaded with {len(config)} parameters",
        extra={"subsys": "tts", "event": "config.loaded", "config": safe_config},
    )

    # Log specific important configuration items
    if "TTS_LANGUAGE" in config:
        logger.info(
            f"TTS language set to {config['TTS_LANGUAGE']}",
            extra={
                "subsys": "tts",
                "event": "config.language",
                "language": config["TTS_LANGUAGE"],
            },
        )

    if "TTS_VOICE" in config:
        logger.info(
            f"TTS voice set to {config['TTS_VOICE']}",
            extra={
                "subsys": "tts",
                "event": "config.voice",
                "voice": config["TTS_VOICE"],
            },
        )

    if "TTS_BACKEND" in config:
        logger.info(
            f"TTS backend set to {config['TTS_BACKEND']}",
            extra={
                "subsys": "tts",
                "event": "config.backend",
                "backend": config["TTS_BACKEND"],
            },
        )


def log_phonemiser_selection(
    language: str, selected: str, available: Dict[str, bool]
) -> None:
    """
    Log the phonemiser selection for a language.

    Args:
        language: The language code
        selected: The selected phonemiser
        available: Dictionary of available phonemisers
    """
    logger.info(
        f"Selected phonemiser '{selected}' for language '{language}'",
        extra={
            "subsys": "tts",
            "event": "phonemiser.selected",
            "language": language,
            "phonemiser": selected,
            "available": available,
        },
    )


def log_voice_loading(voice_id: str, vector_shape: tuple, vector_norm: float) -> None:
    """
    Log voice loading details.

    Args:
        voice_id: The voice identifier
        vector_shape: Shape of the voice vector
        vector_norm: Norm of the voice vector
    """
    logger.debug(
        f"Loaded voice '{voice_id}' with shape {vector_shape} and norm {vector_norm:.4f}",
        extra={
            "subsys": "tts",
            "event": "voice.loaded",
            "voice_id": voice_id,
            "vector_shape": vector_shape,
            "vector_norm": vector_norm,
        },
    )


def log_tts_generation(
    text: str, voice_id: str, output_path: Path, duration_ms: float
) -> None:
    """
    Log TTS generation details.

    Args:
        text: The input text (truncated)
        voice_id: The voice identifier
        output_path: Path to the output file
        duration_ms: Generation duration in milliseconds
    """
    # Truncate text for logging
    text_truncated = text[:50] + "..." if len(text) > 50 else text

    logger.info(
        f"Generated TTS for '{text_truncated}' with voice '{voice_id}' in {duration_ms:.2f}ms",
        extra={
            "subsys": "tts",
            "event": "generation.complete",
            "voice_id": voice_id,
            "text_length": len(text),
            "output_path": str(output_path),
            "duration_ms": duration_ms,
        },
    )

    # Update metrics
    global TTS_METRICS
    TTS_METRICS["tts_generation_count"] += 1
    TTS_METRICS["tts_generation_time_total"] += (
        duration_ms / 1000.0
    )  # Convert to seconds


def log_tts_error(
    error_type: str, error_message: str, details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log TTS error details.

    Args:
        error_type: Type of error
        error_message: Error message
        details: Additional error details
    """
    if details is None:
        details = {}

    logger.error(
        f"TTS error: {error_message} ({error_type})",
        extra={
            "subsys": "tts",
            "event": "error",
            "error_type": error_type,
            "error_message": error_message,
            **details,
        },
    )

    # Update metrics
    global TTS_METRICS
    TTS_METRICS["tts_generation_errors"] += 1


def log_gibberish_detection(metrics: Dict[str, float]) -> None:
    """
    Log gibberish detection details.

    Args:
        metrics: Dictionary of gibberish detection metrics
    """
    logger.warning(
        f"Gibberish audio detected with metrics: {metrics}",
        extra={"subsys": "tts", "event": "gibberish_detection", **metrics},
    )

    # Update metrics
    global TTS_METRICS
    TTS_METRICS["tts_gibberish_detected"] += 1


def log_cache_event(text_hash: str, hit: bool) -> None:
    """
    Log TTS cache hit/miss.

    Args:
        text_hash: Hash of the input text
        hit: True if cache hit, False if miss
    """
    event = "hit" if hit else "miss"

    logger.debug(
        f"TTS cache {event} for hash {text_hash}",
        extra={"subsys": "tts", "event": f"cache.{event}", "text_hash": text_hash},
    )

    # Update metrics
    global TTS_METRICS
    if hit:
        TTS_METRICS["tts_cache_hits"] += 1
    else:
        TTS_METRICS["tts_cache_misses"] += 1


def timed_function(func: Callable) -> Callable:
    """
    Decorator to time a function and log its execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log function execution time
            logger.debug(
                f"{func.__name__} executed in {duration_ms:.2f}ms",
                extra={
                    "subsys": "tts",
                    "event": "function.timed",
                    "function": func.__name__,
                    "duration_ms": duration_ms,
                },
            )

            return result
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log function error
            logger.error(
                f"{func.__name__} failed after {duration_ms:.2f}ms: {str(e)}",
                extra={
                    "subsys": "tts",
                    "event": "function.error",
                    "function": func.__name__,
                    "duration_ms": duration_ms,
                    "error": str(e),
                },
            )

            raise

    return wrapper


def get_tts_metrics() -> Dict[str, Any]:
    """
    Get current TTS metrics.

    Returns:
        Dictionary of TTS metrics
    """
    global TTS_METRICS

    # Calculate derived metrics
    metrics = dict(TTS_METRICS)

    # Calculate average generation time if there are any generations
    if metrics["tts_generation_count"] > 0:
        metrics["tts_generation_time_avg"] = (
            metrics["tts_generation_time_total"] / metrics["tts_generation_count"]
        )
    else:
        metrics["tts_generation_time_avg"] = 0.0

    # Calculate cache hit rate if there are any cache accesses
    cache_accesses = metrics["tts_cache_hits"] + metrics["tts_cache_misses"]
    if cache_accesses > 0:
        metrics["tts_cache_hit_rate"] = metrics["tts_cache_hits"] / cache_accesses
    else:
        metrics["tts_cache_hit_rate"] = 0.0

    # Calculate error rate if there are any generations
    if metrics["tts_generation_count"] > 0:
        metrics["tts_error_rate"] = (
            metrics["tts_generation_errors"] / metrics["tts_generation_count"]
        )
    else:
        metrics["tts_error_rate"] = 0.0

    return metrics


def reset_tts_metrics() -> None:
    """Reset all TTS metrics to zero."""
    global TTS_METRICS
    for key in TTS_METRICS:
        TTS_METRICS[key] = 0 if isinstance(TTS_METRICS[key], int) else 0.0
