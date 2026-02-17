"""Helper utilities for media ingestion flows."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SAFE_FIELDS: Dict[str, Optional[int]] = {
    "title": 200,
    "uploader": 100,
    "source": 50,
    "duration_seconds": None,  # numeric
    "upload_date": 20,
    "url": 500,
}


def sanitize_metadata(
    metadata: Optional[Dict[str, Any]],
    *,
    safe_fields: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Sanitize metadata for safe prompt/context usage."""
    if not metadata:
        return {}

    fields = safe_fields or DEFAULT_SAFE_FIELDS
    sanitized: Dict[str, Any] = {}

    for field, max_length in fields.items():
        if field not in metadata:
            continue

        value = metadata[field]
        if isinstance(value, str):
            cleaned = "".join(
                char for char in value if ord(char) >= 32 or char in "\n\t\r"
            )
            if max_length and len(cleaned) > max_length:
                cleaned = cleaned[:max_length] + "..."
            sanitized[field] = cleaned
            continue

        if isinstance(value, (int, float)):
            sanitized[field] = value
            continue

        str_value = str(value)
        cleaned = "".join(
            char for char in str_value if ord(char) >= 32 or char in "\n\t\r"
        )
        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        sanitized[field] = cleaned

    return sanitized
