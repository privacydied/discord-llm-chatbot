from types import SimpleNamespace

import pytest

from bot.stt_pipeline.transcribe_flow import preprocess_and_transcribe


@pytest.mark.asyncio
async def test_preprocess_and_transcribe_success() -> None:
    spans = object()
    download = object()
    ram_guard_calls = []
    select_calls = []

    class _Job:
        def __init__(self) -> None:
            self.pre = None

        def register_pre(self, pre) -> None:
            self.pre = pre

    class _Guard:
        def check(self, stage: str) -> None:
            ram_guard_calls.append(stage)

    async def _preprocess(**kwargs):
        assert kwargs["spans"] is spans
        assert kwargs["download"] is download
        assert kwargs["voice_note"] is True
        return SimpleNamespace(duration_in=42.5)

    def _select_model_spec(duration_in_s: float):
        select_calls.append(duration_in_s)
        return "spec"

    async def _run_whisper(pre, spans_arg, spec, ram_guard, job=None):
        assert pre.duration_in == 42.5
        assert spans_arg is spans
        assert spec == "spec"
        assert job is test_job
        return SimpleNamespace(text="hello")

    test_job = _Job()
    pre, transcript = await preprocess_and_transcribe(
        source_path="/tmp/a.wav",
        spans=spans,
        download=download,
        voice_note=True,
        ram_guard=_Guard(),
        job=test_job,
        preprocess_audio_with_retry=_preprocess,
        select_model_spec=_select_model_spec,
        run_whisper_with_fallback=_run_whisper,
    )

    assert pre.duration_in == 42.5
    assert transcript.text == "hello"
    assert test_job.pre is pre
    assert ram_guard_calls == ["pre-stage"]
    assert select_calls == [42.5]


@pytest.mark.asyncio
async def test_preprocess_and_transcribe_propagates_preprocess_error() -> None:
    class _Job:
        def register_pre(self, _pre) -> None:
            raise AssertionError("register_pre should not be called")

    class _Guard:
        def check(self, _stage: str) -> None:
            raise AssertionError("guard should not be called")

    async def _preprocess(**_kwargs):
        raise RuntimeError("pre failed")

    async def _run_whisper(*_args, **_kwargs):
        raise AssertionError("whisper should not run")

    with pytest.raises(RuntimeError, match="pre failed"):
        await preprocess_and_transcribe(
            source_path="/tmp/a.wav",
            spans=object(),
            download=None,
            voice_note=False,
            ram_guard=_Guard(),
            job=_Job(),
            preprocess_audio_with_retry=_preprocess,
            select_model_spec=lambda _dur: "spec",
            run_whisper_with_fallback=_run_whisper,
        )
