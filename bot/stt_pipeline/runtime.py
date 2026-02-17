"""Compatibility runtime helpers for STT pipeline execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    norm = str(raw).strip().lower()
    if norm in {"1", "true", "yes", "on", "enabled"}:
        return True
    if norm in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


@dataclass(frozen=True)
class STTRuntimeCompat:
    youtube_transcript_first: bool
    max_ram_mb: int | None


def load_stt_runtime_compat() -> STTRuntimeCompat:
    """Load STT runtime feature toggles with legacy defaults."""
    return STTRuntimeCompat(
        youtube_transcript_first=_env_bool("YOUTUBE_TRANSCRIPT_FIRST", True),
        max_ram_mb=parse_stt_max_ram_mb(),
    )


def parse_stt_max_ram_mb() -> int | None:
    """Parse STT max RAM limit from env, preserving legacy semantics."""
    try:
        raw = os.getenv("STT_MAX_RAM_MB")
        value = int(raw) if raw else None
        if value is not None and value <= 0:
            return None
        return value
    except Exception:
        return None


async def ensure_stt_manager_ready(manager: Any) -> bool:
    """Check manager readiness with compatibility fallbacks.

    Behavior:
    - If `is_available` exists and returns False, return False.
    - If `ensure_ready` exists, await/call it and return its truthiness.
    - If `ensure_ready` is absent, fail-open to True (legacy test stubs).
    """
    is_available = getattr(manager, "is_available", None)
    if callable(is_available):
        try:
            if not bool(is_available()):
                return False
        except Exception:
            return False

    ensure_ready = getattr(manager, "ensure_ready", None)
    if ensure_ready is None:
        return True

    try:
        ready = ensure_ready()
        if hasattr(ready, "__await__"):
            ready = await ready
        return bool(ready)
    except Exception:
        return False
