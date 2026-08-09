from types import SimpleNamespace
from typing import Never

import pytest

from bot.stt_pipeline.youtube_path import (
    build_youtube_transcript_result,
    try_youtube_transcript_first,
)


def test_build_youtube_transcript_result_shapes_metadata() -> None:
    result = build_youtube_transcript_result(
        url="https://www.youtube.com/watch?v=abc",
        transcript_text="hello world",
        title="Video title",
        uploader="Uploader",
        duration_s=42.5,
        cache_hit=True,
        source="ytdlp_automatic_captions",
        language="en",
        timestamp_iso="2026-02-17T00:00:00+00:00",
    )

    assert result["transcription"] == "hello world"
    assert result["partial"] is False
    assert result["metadata"]["source"] == "youtube"
    assert result["metadata"]["url"] == "https://www.youtube.com/watch?v=abc"
    assert result["metadata"]["title"] == "Video title"
    assert result["metadata"]["uploader"] == "Uploader"
    assert result["metadata"]["original_duration_s"] == 42.5
    assert result["metadata"]["processed_duration_s"] == 42.5
    assert result["metadata"]["cache_hit"] is True
    assert result["metadata"]["transcription_source"] == "ytdlp_automatic_captions"
    assert result["metadata"]["transcription_language"] == "en"


def test_build_youtube_transcript_result_defaults_duration() -> None:
    result = build_youtube_transcript_result(
        url="https://www.youtube.com/watch?v=abc",
        transcript_text="hello world",
        title=None,
        uploader=None,
        duration_s=None,
        cache_hit=False,
        source=None,
        language=None,
        timestamp_iso="2026-02-17T00:00:00+00:00",
    )

    assert result["metadata"]["original_duration_s"] == 0.0
    assert result["metadata"]["processed_duration_s"] == 0.0


class _Logger:
    def __init__(self) -> None:
        self.info_calls = []
        self.debug_calls = []

    def info(self, msg, *args) -> None:
        self.info_calls.append((msg, args))

    def debug(self, msg, *args) -> None:
        self.debug_calls.append((msg, args))


@pytest.mark.asyncio
async def test_try_youtube_transcript_first_success() -> None:
    logger = _Logger()

    async def _resolver(_url: str, force_refresh: bool = False):
        assert force_refresh is False
        return SimpleNamespace(
            video_id="abc123",
            text="hello world",
            title="Video title",
            uploader="Uploader",
            duration_s=42.5,
            cache_hit=True,
            source="ytdlp_automatic_captions",
            language="en",
        )

    result = await try_youtube_transcript_first(
        url="https://www.youtube.com/watch?v=abc123",
        force_refresh=False,
        resolver=_resolver,
        logger=logger,
    )

    assert result is not None
    assert result["transcription"] == "hello world"
    assert result["metadata"]["transcription_source"] == "ytdlp_automatic_captions"
    assert result["metadata"]["transcription_language"] == "en"
    assert len(logger.info_calls) == 2
    assert logger.info_calls[0][0].startswith("stt.youtube_transcript.ok")
    assert logger.info_calls[1][0] == "stt.job.complete url=%s chars=%d preview=%s"
    assert logger.debug_calls == []


@pytest.mark.asyncio
async def test_try_youtube_transcript_first_fail_open_on_error() -> None:
    logger = _Logger()

    async def _resolver(_url: str, force_refresh: bool = False) -> Never:
        msg = "boom"
        raise RuntimeError(msg)

    result = await try_youtube_transcript_first(
        url="https://www.youtube.com/watch?v=abc123",
        force_refresh=True,
        resolver=_resolver,
        logger=logger,
    )

    assert result is None
    # A resolver blow-up still fails open, but it is logged at INFO with its cost:
    # this stage can burn tens of seconds of the item budget, and at DEBUG that
    # time was invisible in production logs. [REH]
    assert len(logger.info_calls) == 1
    assert logger.info_calls[0][0].startswith("stt.youtube_transcript.miss reason=error")
    assert logger.debug_calls == []


@pytest.mark.asyncio
async def test_try_youtube_transcript_first_ignores_empty_text() -> None:
    logger = _Logger()

    async def _resolver(_url: str, force_refresh: bool = False):
        return SimpleNamespace(text="")

    result = await try_youtube_transcript_first(
        url="https://www.youtube.com/watch?v=abc123",
        force_refresh=False,
        resolver=_resolver,
        logger=logger,
    )

    assert result is None
    assert len(logger.info_calls) == 1
    assert logger.info_calls[0][0].startswith("stt.youtube_transcript.miss reason=%s")
    assert logger.info_calls[0][1][0] == "no_transcript"
    assert logger.debug_calls == []
