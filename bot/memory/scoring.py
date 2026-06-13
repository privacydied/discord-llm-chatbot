"""Simple scoring helpers for curated long-term memory."""

from __future__ import annotations

import math
from datetime import UTC, datetime


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        value = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def recency_score(
    created_at: str | None,
    last_accessed_at: str | None = None,
    expires_at: str | None = None,
    now: datetime | None = None,
) -> float:
    """Return a 0..1 recency score with a light preference for recent access."""
    now = now or datetime.now(UTC)
    anchor = _parse_dt(last_accessed_at) or _parse_dt(created_at) or now
    age_days = max((now - anchor).total_seconds() / 86400.0, 0.0)

    # Exponential decay with a gentle half-life; temporary memories fall off faster.
    score = math.exp(-age_days / 90.0)

    exp = _parse_dt(expires_at)
    if exp is not None:
        remaining_days = (exp - now).total_seconds() / 86400.0
        if remaining_days <= 0:
            return 0.0
        score *= min(1.0, remaining_days / 14.0)

    return max(0.0, min(1.0, score))


def combined_score(
    semantic_score: float,
    importance: float,
    created_at: str | None,
    last_accessed_at: str | None = None,
    expires_at: str | None = None,
    now: datetime | None = None,
    scope_boost: float = 0.0,
) -> float:
    """Combine semantic, importance, and recency into a single rank score."""
    sem = max(0.0, min(1.0, float(semantic_score)))
    imp = max(0.0, min(1.0, float(importance)))
    rec = recency_score(created_at, last_accessed_at=last_accessed_at, expires_at=expires_at, now=now)

    final_score = sem * 0.65 + imp * 0.20 + rec * 0.15 + max(0.0, scope_boost)
    return max(0.0, min(1.0, final_score))
