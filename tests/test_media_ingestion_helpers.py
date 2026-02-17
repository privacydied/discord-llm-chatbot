from bot.media_ingestion_helpers import sanitize_metadata


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
