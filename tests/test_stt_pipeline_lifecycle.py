from types import SimpleNamespace

import pytest

from bot.stt_pipeline.lifecycle import abort_job_stream_if_present


class _Logger:
    def __init__(self) -> None:
        self.debug_calls = []

    def debug(self, msg, *args, **kwargs):
        self.debug_calls.append((msg, args, kwargs))


class _StreamOK:
    def __init__(self) -> None:
        self.aborted = False

    async def abort(self) -> None:
        self.aborted = True


class _StreamFail:
    async def abort(self) -> None:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_abort_job_stream_if_present_aborts_when_stream_exists() -> None:
    logger = _Logger()
    stream = _StreamOK()
    job = SimpleNamespace(pre=SimpleNamespace(stream=stream))

    await abort_job_stream_if_present(
        job=job,
        logger=logger,
        debug_message="debug msg",
    )

    assert stream.aborted is True
    assert logger.debug_calls == []


@pytest.mark.asyncio
async def test_abort_job_stream_if_present_logs_and_swallows_abort_errors() -> None:
    logger = _Logger()
    job = SimpleNamespace(pre=SimpleNamespace(stream=_StreamFail()))

    await abort_job_stream_if_present(
        job=job,
        logger=logger,
        debug_message="debug msg",
    )

    assert len(logger.debug_calls) == 1
    msg, _args, kwargs = logger.debug_calls[0]
    assert msg == "debug msg"
    assert kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_abort_job_stream_if_present_noops_without_stream() -> None:
    logger = _Logger()
    job = SimpleNamespace(pre=None)

    await abort_job_stream_if_present(
        job=job,
        logger=logger,
        debug_message="debug msg",
    )

    assert logger.debug_calls == []
