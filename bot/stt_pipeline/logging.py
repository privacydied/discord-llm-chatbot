"""Logging helpers for STT pipeline."""

from __future__ import annotations

from typing import Any


def transcript_preview(text: str, *, limit: int = 60) -> str:
    """Return a bounded preview for transcript logs."""
    value = text or ""
    if len(value) > limit:
        return value[:limit] + "..."
    return value


def log_stt_job_complete(
    *,
    logger: Any,
    url: str,
    transcript_text: str,
    url_limit: int = 80,
    preview_limit: int = 60,
) -> None:
    """Emit canonical STT completion breadcrumb."""
    preview = transcript_preview(transcript_text, limit=preview_limit)
    logger.info(
        "stt.job.complete url=%s chars=%d preview=%s",
        (url or "none")[:url_limit],
        len(transcript_text or ""),
        repr(preview),
    )
