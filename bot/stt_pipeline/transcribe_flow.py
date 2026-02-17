"""Shared STT transcribe flow helpers."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Tuple

from .spec_select import select_initial_model_spec


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
) -> Tuple[Any, Any]:
    """Run preprocess + model selection + whisper for one STT source."""
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
    transcript = await run_whisper_with_fallback(
        pre, spans, spec, ram_guard, job=job
    )
    return pre, transcript
