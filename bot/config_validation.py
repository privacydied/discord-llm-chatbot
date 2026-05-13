"""Config validation helper — checks fallback ladders, timeouts, and
structural invariants at startup and on hot-reload.

Usage::

    from bot.config_validation import validate_config
    warnings = validate_config(cfg)  # returns list of warning strings
"""

from __future__ import annotations

import os
from typing import Any


def _parse_model_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def validate_config(cfg: dict[str, Any] | None = None) -> list[str]:
    """Validate the loaded config for structural correctness.

    Returns a list of warning strings. Does NOT raise on recoverable
    issues. Returns an empty list on success.
    """
    if cfg is None:
        from bot.config import load_config

        cfg = load_config()

    warnings: list[str] = []

    # --- Fallback ladder duplicate check ---
    ladders = {
        "TEXT_FALLBACK_MODELS": cfg.get("TEXT_FALLBACK_MODELS"),
        "VISION_FALLBACK_MODELS": cfg.get("VISION_IMAGE_FALLBACK_MODELS"),
        "STT_MULTIMODAL_FALLBACK_MODELS": cfg.get("STT_MULTIMODAL_FALLBACK_MODELS"),
    }
    for name, raw in ladders.items():
        models = _parse_model_list(raw) if raw else _parse_model_list(
            os.getenv(name)
        )
        if len(models) != len(set(models)):
            dupes = [m for m in models if models.count(m) > 1]
            warnings.append(
                f"Config: {name} contains duplicate models: {dupes}"
            )

    # --- Timeout ladder length mismatch ---
    # Check that the number of timeout values matches the number of models
    for name, raw in ladders.items():
        if not raw:
            continue
        models = _parse_model_list(raw)
        timeout_name = f"{name.replace('_MODELS', '')}_TIMEOUTS"
        raw_timeouts = os.getenv(timeout_name)
        if raw_timeouts:
            timeouts = [t.strip() for t in raw_timeouts.split(",") if t.strip()]
            if len(timeouts) != len(models) and len(timeouts) > 0:
                warnings.append(
                    f"Config: {timeout_name} has {len(timeouts)} entries but "
                    f"{name} has {len(models)} models"
                )

    return warnings
