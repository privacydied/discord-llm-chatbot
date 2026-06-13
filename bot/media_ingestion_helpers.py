"""Helper utilities for media ingestion flows."""

from __future__ import annotations

from typing import Any

from .utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SAFE_FIELDS: dict[str, int | None] = {
    "title": 200,
    "uploader": 100,
    "source": 50,
    "duration_seconds": None,  # numeric
    "upload_date": 20,
    "url": 500,
}


def sanitize_metadata(
    metadata: dict[str, Any] | None,
    *,
    safe_fields: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """Sanitize metadata for safe prompt/context usage."""
    if not metadata:
        return {}

    fields = safe_fields or DEFAULT_SAFE_FIELDS
    sanitized: dict[str, Any] = {}

    for field, max_length in fields.items():
        if field not in metadata:
            continue

        value = metadata[field]
        if isinstance(value, str):
            cleaned = "".join(char for char in value if ord(char) >= 32 or char in "\n\t\r")
            if max_length and len(cleaned) > max_length:
                cleaned = cleaned[:max_length] + "..."
            sanitized[field] = cleaned
            continue

        if isinstance(value, (int, float)):
            sanitized[field] = value
            continue

        str_value = str(value)
        cleaned = "".join(char for char in str_value if ord(char) >= 32 or char in "\n\t\r")
        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        sanitized[field] = cleaned

    return sanitized


def build_media_context(transcription: str, metadata: dict[str, Any], url: str) -> str:
    """Build enriched LLM context from transcription + metadata."""
    context_parts = []

    if metadata.get("source"):
        source_info = f"User shared a {metadata['source']} video"
        if metadata.get("title"):
            source_info += f": '{metadata['title']}'"
        if metadata.get("uploader"):
            source_info += f" by {metadata['uploader']}"
        if metadata.get("duration_seconds"):
            duration = metadata["duration_seconds"]
            source_info += f" (Duration: {duration:.1f}s"
            if metadata.get("speedup_factor"):
                source_info += f", processed at {metadata['speedup_factor']}x speed"
            source_info += ")"
        context_parts.append(source_info)
    else:
        context_parts.append(f"User shared a video from: {url}")

    if (transcription or "").strip():
        context_parts.append("The following is the audio transcription:")
        context_parts.append(transcription)
    else:
        context_parts.append("No audio transcription was available.")

    return "\n\n".join(context_parts)
