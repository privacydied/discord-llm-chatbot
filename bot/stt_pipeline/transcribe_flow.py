"""Shared STT transcribe flow helpers."""

from __future__ import annotations

import asyncio
import contextlib
import random
from typing import TYPE_CHECKING, Any

from .spec_select import select_initial_model_spec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# A pure ffmpeg preprocessing timeout is host contention, not a decode
# failure: a production timeout (budget 37.6s) was reproduced against the
# exact same file and ffmpeg command afterwards and it decoded in 0.55s once
# the host was quiet. Retrying the whole attempt from scratch recovers those
# cases instead of failing the job on one slow host moment. [REH][CMV]
PREPROCESS_TIMEOUT_RETRY_MAX_ATTEMPTS = 2
PREPROCESS_TIMEOUT_RETRY_BACKOFF_BASE_S = 2.0
_CAUSE_CHAIN_WALK_LIMIT = 4


def _is_pure_ffmpeg_timeout(error: BaseException) -> bool:
    """True when the failure bottoms out in a pure ffmpeg wall-clock timeout.

    Walks `__cause__` (bounded) because STT_MULTIMODAL_FALLBACK_ENABLED=false
    re-wraps the original timeout behind a generic user-facing message. [REH]
    """
    current: BaseException | None = error
    for _ in range(_CAUSE_CHAIN_WALK_LIMIT):
        if current is None:
            return False
        if "timed out" in str(current).lower():
            return True
        current = current.__cause__
    return False


async def _discard_failed_pre(pre: Any) -> None:
    """Release the ffmpeg process/temp file for an attempt abandoned after a
    timeout retry -- job.register_pre() only tracks the latest attempt, so an
    earlier attempt's resources would otherwise leak until the job closes. [RM]
    """
    stream = getattr(pre, "stream", None)
    if stream is None:
        return
    with contextlib.suppress(Exception):
        await stream.abort()
    temp_path = getattr(stream, "_temp_path", None)
    if temp_path is not None:
        with contextlib.suppress(Exception):
            temp_path.unlink(missing_ok=True)


async def _wait_before_retry(attempt: int, error: BaseException, logger: Any) -> None:
    backoff_s = PREPROCESS_TIMEOUT_RETRY_BACKOFF_BASE_S * (attempt + 1) + random.uniform(0, 1.0)  # nosec B311
    logger.warning(
        "stt.preprocess_transcribe_timeout_retry attempt=%d backoff_s=%.1f error=%s",
        attempt + 1,
        backoff_s,
        str(error)[:80],
    )
    await asyncio.sleep(backoff_s)


async def _run_single_attempt(
    *,
    source_path: Any,
    spans: Any,
    download: Any,
    voice_note: bool,
    ram_guard: Any,
    job: Any,
    manager: Any,
    logger: Any,
    downgrade_threshold_s: float,
    preprocess_audio_with_retry: Callable[..., Awaitable[Any]],
    run_whisper_with_fallback: Callable[..., Awaitable[Any]],
    language: str | None,
) -> tuple[Any, Any]:
    """One preprocess + model-select + whisper attempt; raises on failure."""
    pre = await preprocess_audio_with_retry(
        source_path=source_path,
        spans=spans,
        download=download,
        voice_note=voice_note,
        ram_guard=ram_guard,
    )
    job.register_pre(pre)
    ram_guard.check("pre-stage")

    spec = select_initial_model_spec(
        manager=manager,
        duration_in_s=pre.duration_in,
        downgrade_threshold_s=downgrade_threshold_s,
        logger=logger,
    )
    transcript = await run_whisper_with_fallback(pre, spans, spec, ram_guard, job=job, language=language)
    return pre, transcript


async def preprocess_and_transcribe(
    *,
    source_path: Any,
    spans: Any,
    download: Any,
    voice_note: bool,
    ram_guard: Any,
    job: Any,
    manager: Any,
    logger: Any,
    downgrade_threshold_s: float = 120.0,
    preprocess_audio_with_retry: Callable[..., Awaitable[Any]],
    run_whisper_with_fallback: Callable[..., Awaitable[Any]],
    language: str | None = None,
) -> tuple[Any, Any]:
    """Run preprocess + model selection + whisper for one STT source.

    Retries the whole attempt, bounded with backoff, when the failure is a
    pure ffmpeg timeout -- see `_is_pure_ffmpeg_timeout`. Any other error
    (a real codec/data failure) propagates on the first attempt so Tier 2
    extraction upstream can still handle it. [REH]
    """
    attempt_kwargs = {
        "source_path": source_path,
        "spans": spans,
        "download": download,
        "voice_note": voice_note,
        "ram_guard": ram_guard,
        "job": job,
        "manager": manager,
        "logger": logger,
        "downgrade_threshold_s": downgrade_threshold_s,
        "preprocess_audio_with_retry": preprocess_audio_with_retry,
        "run_whisper_with_fallback": run_whisper_with_fallback,
        "language": language,
    }
    for attempt in range(PREPROCESS_TIMEOUT_RETRY_MAX_ATTEMPTS):
        is_last_attempt = attempt == PREPROCESS_TIMEOUT_RETRY_MAX_ATTEMPTS - 1
        try:
            return await _run_single_attempt(**attempt_kwargs)
        except Exception as exc:
            if is_last_attempt or not _is_pure_ffmpeg_timeout(exc):
                raise
            if job.pre is not None:
                await _discard_failed_pre(job.pre)
            await _wait_before_retry(attempt, exc, logger)
    msg = "unreachable: loop above always returns or raises"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover
