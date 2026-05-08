"""Search helpers for the raw server archive."""

from __future__ import annotations

import re


def normalize_query(query: str) -> str:
    tokens = re.findall(r"[\w@#:/.-]+", query or "")
    return " ".join(tokens).strip()


def sanitize_snippet(text: str, *, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return "[empty]"
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)] + "…"
