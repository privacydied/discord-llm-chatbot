"""Helpers for STT stitch-stage composition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def run_stitch_stage(
    *,
    spans: Any,
    pre: Any,
    transcript: Any,
    build_result: Callable[[], Any],
    log_summary: Callable[..., None] | None = None,
) -> Any:
    """Run canonical stitch-stage bookkeeping and return built payload."""
    spans.start("stitch")
    result = build_result()
    spans.end("stitch", ok=True)
    if log_summary is not None:
        log_summary(spans, pre, transcript, cache_hit=transcript.cache_hit)
    return result
