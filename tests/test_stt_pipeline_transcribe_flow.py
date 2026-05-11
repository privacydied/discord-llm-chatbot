from types import SimpleNamespace

import pytest

from bot.stt_pipeline.transcribe_flow import preprocess_and_transcribe


@pytest.mark.asyncio
async def test_preprocess_and_transcribe_success() -> None:
    spans = object()
    download = object()
    ram_guard_calls = []

    class _Job:
        def __init__(self) -> None:
            self.pre = None

        def register_pre(self, pre) -> None:
            self.pre = pre

    class _Guard:
        def check(self, stage: str) -> None:
            ram_guard_calls.append(stage)

    class _Logger:
        def __init__(self) -> None:
            self.info_calls = []

        def info(self, msg, *args):
            self.info_calls.append((msg, args))

    async def _preprocess(**kwargs):
        assert kwargs["spans"] is spans
        assert kwargs["download"] is download
        assert kwargs["voice_note"] is True
        return SimpleNamespace(duration_in=42.5)

    base_spec = SimpleNamespace(size="base")
    manager = SimpleNamespace(
        default_spec=base_spec,
        downgrade_spec=lambda _spec: None,
    )
    log = _Logger()

    async def _run_whisper(pre, spans_arg, spec, ram_guard, job=None, language=None):
        assert pre.duration_in == 42.5
        assert spans_arg is spans
        assert spec is base_spec
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
        manager=manager,
        logger=log,
        downgrade_threshold_s=120.0,
        preprocess_audio_with_retry=_preprocess,
        run_whisper_with_fallback=_run_whisper,
    )

    assert pre.duration_in == 42.5
    assert transcript.text == "hello"
    assert test_job.pre is pre
    assert ram_guard_calls == ["pre-stage"]
    assert log.info_calls == []


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
            manager=SimpleNamespace(
                default_spec=SimpleNamespace(size="base"),
                downgrade_spec=lambda _spec: None,
            ),
            logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
            preprocess_audio_with_retry=_preprocess,
            run_whisper_with_fallback=_run_whisper,
        )
