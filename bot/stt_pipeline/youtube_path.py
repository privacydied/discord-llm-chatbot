"""Helpers for YouTube transcript-first STT path."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .logging import log_stt_job_complete

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def build_youtube_transcript_result(
    *,
    url: str,
    transcript_text: str,
    title: str | None,
    uploader: str | None,
    duration_s: float | None,
    cache_hit: bool,
    source: str | None,
    language: str | None,
    timestamp_iso: str | None = None,
) -> dict[str, Any]:
    """Build canonical STT result payload for transcript-first YouTube path."""
    now_iso = timestamp_iso or datetime.now(UTC).isoformat()
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


async def try_youtube_transcript_first(
    *,
    url: str,
    force_refresh: bool,
    resolver: Callable[..., Awaitable[Any]],
    logger: Any,
) -> dict[str, Any] | None:
    """Resolve transcript-first payload for YouTube URLs, fail-open on resolver errors.

    Misses are logged at INFO with their cost: this stage can spend tens of seconds
    on an HTML fetch plus a yt-dlp caption probe before giving up, and when it only
    logged on success that time was invisible in the log -- it read as dead air
    between stt.job.start and the next stage. [REH]
    """
    started = time.monotonic()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    try:
        yt = await resolver(url, force_refresh=force_refresh)
    except Exception as exc:
        logger.info(
            "stt.youtube_transcript.miss reason=error elapsed_ms=%d url=%s err=%s",
            _elapsed_ms(),
            url[:120] if url else "none",
            exc,
        )
        return None

    if not (yt and getattr(yt, "text", "")):
        logger.info(
            "stt.youtube_transcript.miss reason=%s elapsed_ms=%d url=%s",
            "no_transcript" if yt else "unresolved",
            _elapsed_ms(),
            url[:120] if url else "none",
        )
        return None

    result = build_youtube_transcript_result(
        url=url,
        transcript_text=yt.text,
        title=yt.title,
        uploader=yt.uploader,
        duration_s=yt.duration_s,
        cache_hit=bool(yt.cache_hit),
        source=yt.source,
        language=yt.language,
    )
    logger.info(
        "stt.youtube_transcript.ok video_id=%s lang=%s source=%s chars=%d cache_hit=%s elapsed_ms=%d",
        yt.video_id,
        yt.language or "unknown",
        yt.source,
        len(yt.text),
        str(bool(yt.cache_hit)).lower(),
        _elapsed_ms(),
    )
    log_stt_job_complete(
        logger=logger,
        url=url,
        transcript_text=yt.text,
    )
    return result
