"""Helpers for YouTube transcript-first STT path."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def build_youtube_transcript_result(
    *,
    url: str,
    transcript_text: str,
    title: Optional[str],
    uploader: Optional[str],
    duration_s: Optional[float],
    cache_hit: bool,
    source: Optional[str],
    language: Optional[str],
    timestamp_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """Build canonical STT result payload for transcript-first YouTube path."""
    now_iso = timestamp_iso or datetime.now(timezone.utc).isoformat()
    dur = float(duration_s or 0.0)
    return {
        "transcription": transcript_text,
        "partial": False,
        "abort_reason": "",
        "metadata": {
            "source": "youtube",
            "url": url,
            "title": title,
            "uploader": uploader,
            "upload_date": "",
            "original_duration_s": dur,
            "processed_duration_s": dur,
            "speedup_factor": 1.0,
            "cache_hit": bool(cache_hit),
            "timestamp": now_iso,
            "demux_fallback": False,
            "transcription_source": source,
            "transcription_language": language,
        },
    }
