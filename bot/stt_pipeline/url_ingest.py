"""URL ingest helpers for STT pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bot.exceptions import InferenceError

from .runtime import ensure_stt_manager_ready

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


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


async def prepare_url_download_for_stt(
    *,
    url: str,
    force_refresh: bool,
    manager: Any,
    job: Any,
    spans: Any,
    ram_guard: Any,
    fetcher: Callable[..., Awaitable[Any]],
    ingest_error_type: Any,
    ensure_ready_or_raise: Callable[..., Awaitable[None]] = ensure_manager_ready_or_raise,
    fetch_or_raise: Callable[..., Awaitable[Any]] = fetch_url_audio_or_raise,
) -> Any:
    """Run canonical URL download preparation for STT URL entrypoint."""
    await ensure_ready_or_raise(
        manager=manager,
        job=job,
    )
    download = await fetch_or_raise(
        url=url,
        force_refresh=force_refresh,
        fetcher=fetcher,
        spans=spans,
        job=job,
        ingest_error_type=ingest_error_type,
    )
    job.register_download(download)
    ram_guard.check("yt-dlp")
    return download
