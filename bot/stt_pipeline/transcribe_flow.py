"""Shared STT transcribe flow helpers."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Tuple


async def preprocess_and_transcribe(
    *,
    source_path: Any,
    spans: Any,
    download: Any,
    voice_note: bool,
    ram_guard: Any,
    job: Any,
    preprocess_audio_with_retry: Callable[..., Awaitable[Any]],
    select_model_spec: Callable[[float], Any],
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

    spec = select_model_spec(pre.duration_in)
    transcript = await run_whisper_with_fallback(
        pre, spans, spec, ram_guard, job=job
    )
    return pre, transcript
