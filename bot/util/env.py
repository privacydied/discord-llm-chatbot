"""Environment parsing helpers for consistent boolean/numeric handling. [IV]"""
from __future__ import annotations

import os
from typing import Optional


def get_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var with common truthy values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

