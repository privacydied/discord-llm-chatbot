"""URL ingest helpers for STT pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..exceptions import InferenceError
from .runtime import ensure_stt_manager_ready


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


async def ensure_manager_ready_or_raise(
    *,
    manager: Any,
    job: Any,
    ensure_ready: Callable[[Any], Awaitable[bool]] = ensure_stt_manager_ready,
) -> None:
    """Ensure STT manager readiness or raise canonical user-facing error."""
    ready = await ensure_ready(manager)
    if ready:
        return
    exc = InferenceError("STT engine not available")
    await job.finish_failure(exc)
    raise exc


async def fetch_url_audio_or_raise(
    *,
    url: str,
    force_refresh: bool,
    fetcher: Callable[..., Awaitable[Any]],
    spans: Any,
    job: Any,
    ingest_error_type: Any,
) -> Any:
    """Fetch URL audio and convert configured ingest errors to InferenceError."""
    try:
        return await fetch_url_audio_with_span(
            url=url,
            force_refresh=force_refresh,
            fetcher=fetcher,
            spans=spans,
        )
    except ingest_error_type as exc:
        await job.finish_failure(exc)
        raise InferenceError(str(exc)) from exc
