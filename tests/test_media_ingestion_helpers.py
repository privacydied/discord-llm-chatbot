from bot.media_ingestion_helpers import build_media_context, sanitize_metadata


def test_sanitize_metadata_keeps_only_safe_fields() -> None:
    raw = {
        "title": "Video Title",
        "uploader": "Uploader",
        "source": "youtube",
        "duration_seconds": 120.5,
        "upload_date": "2026-01-01",
        "url": "https://youtube.com/watch?v=abc",
        "ignored": "drop me",
    }
    out = sanitize_metadata(raw)
    assert out["title"] == "Video Title"
    assert out["uploader"] == "Uploader"
    assert out["source"] == "youtube"
    assert out["duration_seconds"] == 120.5
    assert out["upload_date"] == "2026-01-01"
    assert out["url"] == "https://youtube.com/watch?v=abc"
    assert "ignored" not in out


def test_sanitize_metadata_removes_control_chars_but_preserves_whitespace() -> None:
    raw = {
        "title": "Test\x00Video\x01Title\x1f",
        "uploader": "Channel\nName\tWith\rWhitespace",
    }
    out = sanitize_metadata(raw)
    assert out["title"] == "TestVideoTitle"
    assert out["uploader"] == "Channel\nName\tWith\rWhitespace"


def test_sanitize_metadata_applies_length_limits() -> None:
    raw = {"title": "A" * 300, "uploader": "B" * 150, "url": "C" * 700}
    out = sanitize_metadata(raw)
    assert len(out["title"]) <= 203
    assert out["title"].endswith("...")
    assert len(out["uploader"]) <= 103
    assert out["uploader"].endswith("...")
    assert len(out["url"]) <= 503
    assert out["url"].endswith("...")


def test_build_media_context_with_full_metadata() -> None:
    out = build_media_context(
        "This is the transcript.",
        {
            "source": "youtube",
            "title": "Test Video",
            "uploader": "Test Channel",
            "duration_seconds": 120.5,
            "speedup_factor": 1.5,
        },
        "https://youtube.com/watch?v=abc",
    )
    assert "youtube video" in out.lower()
    assert "Test Video" in out
    assert "Test Channel" in out
    assert "120.5s" in out
    assert "1.5x speed" in out
    assert "This is the transcript." in out


def test_build_media_context_with_minimal_metadata() -> None:
    out = build_media_context(
        "Transcript text",
        {},
        "https://youtube.com/watch?v=abc",
    )
    assert "User shared a video from: https://youtube.com/watch?v=abc" in out
    assert "Transcript text" in out


def test_build_media_context_no_transcription_note() -> None:
    out = build_media_context(
        "",
        {"source": "youtube", "title": "No Audio"},
        "https://youtube.com/watch?v=abc",
    )
    assert "No audio transcription was available." in out
