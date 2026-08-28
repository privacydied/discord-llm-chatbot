from types import SimpleNamespace
from typing import Never

import pytest

from bot.exceptions import InferenceError
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

        def info(self, msg, *args) -> None:
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
            msg = "register_pre should not be called"
            raise AssertionError(msg)

    class _Guard:
        def check(self, _stage: str) -> None:
            msg = "guard should not be called"
            raise AssertionError(msg)

    async def _preprocess(**_kwargs) -> Never:
        msg = "pre failed"
        raise RuntimeError(msg)

    async def _run_whisper(*_args, **_kwargs) -> Never:
        msg = "whisper should not run"
        raise AssertionError(msg)

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


class _RetryStream:
    def __init__(self) -> None:
        self.aborted = False
        self._temp_path = None

    async def abort(self) -> None:
        self.aborted = True


class _RetryJob:
    def __init__(self) -> None:
        self.pre = None

    def register_pre(self, pre) -> None:
        self.pre = pre


class _RetryLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def info(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, msg, *args) -> None:
        self.warnings.append(msg % args if args else msg)


async def _fast_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_preprocess_and_transcribe_retries_pure_ffmpeg_timeout(monkeypatch) -> None:
    monkeypatch.setattr("bot.stt_pipeline.transcribe_flow.asyncio.sleep", _fast_sleep)

    streams = [_RetryStream(), _RetryStream()]
    attempts = {"pre": 0, "whisper": 0}

    async def _preprocess(**_kwargs):
        pre = SimpleNamespace(duration_in=10.0, stream=streams[attempts["pre"]])
        attempts["pre"] += 1
        return pre

    async def _run_whisper(pre, *_args, **_kwargs):
        idx = attempts["whisper"]
        attempts["whisper"] += 1
        if idx == 0:
            cause = InferenceError("Audio preprocessing timed out")
            raise InferenceError("Audio transcription is temporarily unavailable.") from cause
        return SimpleNamespace(text="recovered")

    job = _RetryJob()
    log = _RetryLogger()
    manager = SimpleNamespace(default_spec=SimpleNamespace(size="base"), downgrade_spec=lambda _s: None)

    pre, transcript = await preprocess_and_transcribe(
        source_path="/tmp/a.mp4",
        spans=object(),
        download=None,
        voice_note=False,
        ram_guard=SimpleNamespace(check=lambda _stage: None),
        job=job,
        manager=manager,
        logger=log,
        preprocess_audio_with_retry=_preprocess,
        run_whisper_with_fallback=_run_whisper,
    )

    assert transcript.text == "recovered"
    assert attempts["pre"] == 2
    assert streams[0].aborted is True  # first (failed) attempt's ffmpeg process released
    assert streams[1].aborted is False  # second (successful) attempt left alone
    assert job.pre is pre
    assert len(log.warnings) == 1


@pytest.mark.asyncio
async def test_preprocess_and_transcribe_exhausts_timeout_retries(monkeypatch) -> None:
    monkeypatch.setattr("bot.stt_pipeline.transcribe_flow.asyncio.sleep", _fast_sleep)

    async def _preprocess(**_kwargs):
        return SimpleNamespace(duration_in=10.0, stream=_RetryStream())

    async def _run_whisper(*_args, **_kwargs) -> Never:
        cause = InferenceError("Audio preprocessing timed out")
        raise InferenceError("Audio transcription is temporarily unavailable.") from cause

    manager = SimpleNamespace(default_spec=SimpleNamespace(size="base"), downgrade_spec=lambda _s: None)

    with pytest.raises(InferenceError, match="temporarily unavailable"):
        await preprocess_and_transcribe(
            source_path="/tmp/a.mp4",
            spans=object(),
            download=None,
            voice_note=False,
            ram_guard=SimpleNamespace(check=lambda _stage: None),
            job=_RetryJob(),
            manager=manager,
            logger=_RetryLogger(),
            preprocess_audio_with_retry=_preprocess,
            run_whisper_with_fallback=_run_whisper,
        )


def test_is_pure_ffmpeg_timeout_walks_cause_chain() -> None:
    from bot.stt_pipeline.transcribe_flow import _is_pure_ffmpeg_timeout

    cause = InferenceError("Audio preprocessing timed out")
    wrapped = InferenceError("Audio transcription is temporarily unavailable.")
    wrapped.__cause__ = cause

    assert _is_pure_ffmpeg_timeout(wrapped) is True
    assert _is_pure_ffmpeg_timeout(RuntimeError("no audio stream")) is False
