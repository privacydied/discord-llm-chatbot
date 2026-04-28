"""Helpers for canonical STT payload shaping."""

from __future__ import annotations

from typing import Any, Dict


def build_url_transcript_result(
    *,
    transcript: Any,
    download: Any,
    pre: Any,
    atempo_factor: float,
) -> Dict[str, Any]:
    """Build canonical STT payload for yt-dlp/ffmpeg/whisper URL path."""
    metadata = download.metadata
    return {
        "transcription": transcript.text,
        "partial": transcript.aborted,
        "abort_reason": transcript.abort_reason or "",
        "confidence": transcript.confidence,
        "confidence_status": transcript.confidence_status,
        "language": transcript.language_detected,
        "language_confidence": transcript.language_confidence,
        "metadata": {
            "source": metadata.source_type,
            "url": metadata.url,
            "title": metadata.title,
            "uploader": metadata.uploader,
            "upload_date": metadata.upload_date,
            "original_duration_s": metadata.duration_seconds,
            "processed_duration_s": pre.duration_out,
            "speedup_factor": atempo_factor if pre.atempo_applied else 1.0,
            "cache_hit": download.cache_hit or transcript.cache_hit,
            "timestamp": download.timestamp.isoformat(),
            "demux_fallback": bool(getattr(download, "demux_fallback", False)),
        },
    }
