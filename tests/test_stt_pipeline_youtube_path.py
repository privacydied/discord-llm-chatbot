from bot.stt_pipeline.youtube_path import build_youtube_transcript_result


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
