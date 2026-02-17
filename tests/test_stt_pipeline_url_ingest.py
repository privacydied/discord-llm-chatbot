import pytest

from bot.exceptions import InferenceError
from bot.stt_pipeline.url_ingest import (
    ensure_manager_ready_or_raise,
    fetch_url_audio_or_raise,
    fetch_url_audio_with_span,
)


class _Spans:
    def __init__(self) -> None:
        self.calls = []

    def start(self, stage: str) -> None:
        self.calls.append(("start", stage, {}))

    def end(self, stage: str, **kwargs) -> None:
        self.calls.append(("end", stage, kwargs))


class _Job:
    def __init__(self) -> None:
        self.failures = []

    async def finish_failure(self, exc: Exception) -> None:
        self.failures.append(exc)


@pytest.mark.asyncio
async def test_fetch_url_audio_with_span_success() -> None:
    spans = _Spans()

    async def _fetcher(url: str, force_refresh: bool = False):
        assert url == "https://x.com/user/status/1"
        assert force_refresh is True
        return {"ok": True}

    result = await fetch_url_audio_with_span(
        url="https://x.com/user/status/1",
        force_refresh=True,
        fetcher=_fetcher,
        spans=spans,
    )

    assert result == {"ok": True}
    assert spans.calls == [
        ("start", "yt-dlp", {}),
        ("end", "yt-dlp", {"ok": True}),
    ]


@pytest.mark.asyncio
async def test_fetch_url_audio_with_span_error_marks_span_and_reraises() -> None:
    spans = _Spans()

    async def _fetcher(url: str, force_refresh: bool = False):
        raise RuntimeError("fetch failed")

    with pytest.raises(RuntimeError, match="fetch failed"):
        await fetch_url_audio_with_span(
            url="https://x.com/user/status/1",
            force_refresh=False,
            fetcher=_fetcher,
            spans=spans,
        )

    assert spans.calls == [
        ("start", "yt-dlp", {}),
        ("end", "yt-dlp", {"ok": False, "reason": "error"}),
    ]


@pytest.mark.asyncio
async def test_ensure_manager_ready_or_raise_passes_when_ready() -> None:
    job = _Job()

    async def _ensure_ready(_manager) -> bool:
        return True

    await ensure_manager_ready_or_raise(
        manager=object(),
        job=job,
        ensure_ready=_ensure_ready,
    )

    assert job.failures == []


@pytest.mark.asyncio
async def test_ensure_manager_ready_or_raise_raises_inference_error_when_not_ready() -> None:
    job = _Job()

    async def _ensure_ready(_manager) -> bool:
        return False

    with pytest.raises(InferenceError, match="STT engine not available"):
        await ensure_manager_ready_or_raise(
            manager=object(),
            job=job,
            ensure_ready=_ensure_ready,
        )

    assert len(job.failures) == 1
    assert isinstance(job.failures[0], InferenceError)


@pytest.mark.asyncio
async def test_fetch_url_audio_or_raise_converts_ingest_error() -> None:
    job = _Job()
    spans = _Spans()

    class _IngestErr(Exception):
        pass

    async def _fetcher(url: str, force_refresh: bool = False):
        raise _IngestErr("download failed")

    with pytest.raises(InferenceError, match="download failed"):
        await fetch_url_audio_or_raise(
            url="https://x.com/user/status/1",
            force_refresh=False,
            fetcher=_fetcher,
            spans=spans,
            job=job,
            ingest_error_type=_IngestErr,
        )

    assert len(job.failures) == 1
    assert str(job.failures[0]) == "download failed"
    assert spans.calls == [
        ("start", "yt-dlp", {}),
        ("end", "yt-dlp", {"ok": False, "reason": "error"}),
    ]


@pytest.mark.asyncio
async def test_fetch_url_audio_or_raise_passthrough_non_ingest_error() -> None:
    job = _Job()
    spans = _Spans()

    class _IngestErr(Exception):
        pass

    async def _fetcher(url: str, force_refresh: bool = False):
        raise RuntimeError("other failure")

    with pytest.raises(RuntimeError, match="other failure"):
        await fetch_url_audio_or_raise(
            url="https://x.com/user/status/1",
            force_refresh=False,
            fetcher=_fetcher,
            spans=spans,
            job=job,
            ingest_error_type=_IngestErr,
        )

    assert job.failures == []
    assert spans.calls == [
        ("start", "yt-dlp", {}),
        ("end", "yt-dlp", {"ok": False, "reason": "error"}),
    ]
