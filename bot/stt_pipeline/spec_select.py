"""Helpers for STT model spec selection."""

from __future__ import annotations

from typing import Any


def select_initial_model_spec(
    *,
    manager: Any,
    duration_in_s: float,
    downgrade_threshold_s: float,
    logger: Any,
) -> Any:
    """Select initial whisper spec with long-audio downgrade semantics."""
    spec = manager.default_spec
    if duration_in_s > downgrade_threshold_s:
        downgraded = manager.downgrade_spec(spec)
        if downgraded:
            logger.info(
                "whisper.model_downgrade from=%s to=%s reason=long_audio",
                spec.size,
                downgraded.size,
            )
            spec = downgraded
    return spec
