from typing import Never

import pytest

from bot.exceptions import InferenceError
from bot.stt_pipeline.url_ingest import (
    ensure_manager_ready_or_raise,
    fetch_url_audio_or_raise,
    fetch_url_audio_with_span,
    prepare_url_download_for_stt,
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
        self.downloads = []

    async def finish_failure(self, exc: Exception) -> None:
        self.failures.append(exc)

    def register_download(self, download) -> None:
        self.downloads.append(download)


class _Guard:
    def __init__(self) -> None:
        self.calls = []

    def check(self, stage: str) -> None:
        self.calls.append(stage)


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

    async def _fetcher(url: str, force_refresh: bool = False) -> Never:
        msg = "fetch failed"
        raise RuntimeError(msg)

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

    async def _fetcher(url: str, force_refresh: bool = False) -> Never:
        msg = "download failed"
        raise _IngestErr(msg)

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

    async def _fetcher(url: str, force_refresh: bool = False) -> Never:
        msg = "other failure"
        raise RuntimeError(msg)

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


@pytest.mark.asyncio
async def test_prepare_url_download_for_stt_success() -> None:
    job = _Job()
    guard = _Guard()
    spans = _Spans()
    manager = object()
    order = []

    async def _ensure_ready_or_raise(**kwargs) -> None:
        assert kwargs["manager"] is manager
        assert kwargs["job"] is job
        order.append("ensure")

    async def _fetch_or_raise(**kwargs):
        assert kwargs["url"] == "https://x.com/user/status/1"
        assert kwargs["force_refresh"] is True
        assert kwargs["spans"] is spans
        assert kwargs["job"] is job
        order.append("fetch")
        return {"ok": True}

    result = await prepare_url_download_for_stt(
        url="https://x.com/user/status/1",
        force_refresh=True,
        manager=manager,
        job=job,
        spans=spans,
        ram_guard=guard,
        fetcher=lambda *_args, **_kwargs: None,
        ingest_error_type=RuntimeError,
        ensure_ready_or_raise=_ensure_ready_or_raise,
        fetch_or_raise=_fetch_or_raise,
    )

    assert result == {"ok": True}
    assert order == ["ensure", "fetch"]
    assert job.downloads == [{"ok": True}]
    assert guard.calls == ["yt-dlp"]


@pytest.mark.asyncio
async def test_prepare_url_download_for_stt_stops_on_ready_error() -> None:
    job = _Job()
    guard = _Guard()
    spans = _Spans()

    async def _ensure_ready_or_raise(**kwargs) -> Never:
        msg = "not ready"
        raise RuntimeError(msg)

    async def _fetch_or_raise(**kwargs) -> Never:
        msg = "fetch should not run"
        raise AssertionError(msg)

    with pytest.raises(RuntimeError, match="not ready"):
        await prepare_url_download_for_stt(
            url="https://x.com/user/status/1",
            force_refresh=False,
            manager=object(),
            job=job,
            spans=spans,
            ram_guard=guard,
            fetcher=lambda *_args, **_kwargs: None,
            ingest_error_type=RuntimeError,
            ensure_ready_or_raise=_ensure_ready_or_raise,
            fetch_or_raise=_fetch_or_raise,
        )

    assert job.downloads == []
    assert guard.calls == []


@pytest.mark.asyncio
async def test_prepare_url_download_for_stt_stops_on_fetch_error() -> None:
    job = _Job()
    guard = _Guard()
    spans = _Spans()

    async def _ensure_ready_or_raise(**kwargs) -> None:
        return None

    async def _fetch_or_raise(**kwargs) -> Never:
        msg = "fetch failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="fetch failed"):
        await prepare_url_download_for_stt(
            url="https://x.com/user/status/1",
            force_refresh=False,
            manager=object(),
            job=job,
            spans=spans,
            ram_guard=guard,
            fetcher=lambda *_args, **_kwargs: None,
            ingest_error_type=RuntimeError,
            ensure_ready_or_raise=_ensure_ready_or_raise,
            fetch_or_raise=_fetch_or_raise,
        )

    assert job.downloads == []
    assert guard.calls == []
