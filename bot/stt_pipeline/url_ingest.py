"""URL ingest helpers for STT pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


async def fetch_url_audio_with_span(
    *,
    url: str,
    force_refresh: bool,
    fetcher: Callable[..., Awaitable[Any]],
    spans: Any,
) -> Any:
    """Fetch URL audio while emitting canonical yt-dlp span markers."""
    spans.start("yt-dlp")
    try:
        result = await fetcher(url, force_refresh=force_refresh)
        spans.end("yt-dlp", ok=True)
        return result
    except Exception:
        spans.end("yt-dlp", ok=False, reason="error")
        raise
