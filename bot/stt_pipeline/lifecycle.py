"""Lifecycle helpers for STT jobs."""

from __future__ import annotations

from typing import Any


async def abort_job_stream_if_present(
    *,
    job: Any,
    logger: Any,
    debug_message: str,
) -> None:
    """Abort a job's active stream if available, swallowing abort errors."""
    pre = getattr(job, "pre", None)
    stream = getattr(pre, "stream", None) if pre is not None else None
    if stream is None:
        return
    try:
        await stream.abort()
    except Exception:
        logger.debug(debug_message, exc_info=True)


async def abort_and_finish_failure(
    *,
    job: Any,
    logger: Any,
    exc: BaseException,
    abort_debug_message: str,
) -> None:
    """Abort stream if present, then record failure on the job."""
    await abort_job_stream_if_present(
        job=job,
        logger=logger,
        debug_message=abort_debug_message,
    )
    await job.finish_failure(exc)
