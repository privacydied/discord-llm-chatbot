"""Helpers for STT model spec selection."""

from __future__ import annotations

from typing import Any


def predict_model_spec(*, manager: Any, duration_in_s: float, downgrade_threshold_s: float) -> Any:
    """The spec ``select_initial_model_spec`` will pick for this duration, without logging."""
    spec = manager.default_spec
    if duration_in_s > downgrade_threshold_s:
        return manager.downgrade_spec(spec) or spec
    return spec


def select_initial_model_spec(
    *,
    manager: Any,
    duration_in_s: float,
    downgrade_threshold_s: float,
    logger: Any,
) -> Any:
    """Select initial whisper spec with long-audio downgrade semantics."""
    spec = predict_model_spec(manager=manager, duration_in_s=duration_in_s, downgrade_threshold_s=downgrade_threshold_s)
    if spec != manager.default_spec:
        logger.info(
            "whisper.model_downgrade from=%s to=%s reason=long_audio",
            manager.default_spec.size,
            spec.size,
        )
    return spec
