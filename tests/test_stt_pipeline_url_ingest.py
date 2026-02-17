import pytest

from bot.stt_pipeline.url_ingest import fetch_url_audio_with_span


class _Spans:
    def __init__(self) -> None:
        self.calls = []

    def start(self, stage: str) -> None:
        self.calls.append(("start", stage, {}))

    def end(self, stage: str, **kwargs) -> None:
        self.calls.append(("end", stage, kwargs))


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
