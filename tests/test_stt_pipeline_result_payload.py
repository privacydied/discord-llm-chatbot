from datetime import datetime, timezone
from types import SimpleNamespace

from bot.stt_pipeline.result_payload import build_url_transcript_result


def test_build_url_transcript_result_shapes_metadata() -> None:
    transcript = SimpleNamespace(
        text="hello world",
        aborted=True,
        abort_reason="chunk_limit",
        cache_hit=False,
        confidence=None,
        confidence_status="unknown",
        language_detected=None,
        language_confidence=None,
    )
    metadata = SimpleNamespace(
        source_type="twitter",
        url="https://x.com/user/status/1",
        title="clip",
        uploader="alice",
        upload_date="20260217",
        duration_seconds=12.34,
    )
    download = SimpleNamespace(
        metadata=metadata,
        cache_hit=True,
        timestamp=datetime(2026, 2, 17, tzinfo=timezone.utc),
        demux_fallback=True,
    )
    pre = SimpleNamespace(duration_out=10.0, atempo_applied=True)

    result = build_url_transcript_result(
        transcript=transcript,
        download=download,
        pre=pre,
        atempo_factor=1.25,
    )

    assert result["transcription"] == "hello world"
    assert result["partial"] is True
    assert result["abort_reason"] == "chunk_limit"
    assert result["metadata"]["source"] == "twitter"
    assert result["metadata"]["url"] == "https://x.com/user/status/1"
    assert result["metadata"]["title"] == "clip"
    assert result["metadata"]["uploader"] == "alice"
    assert result["metadata"]["upload_date"] == "20260217"
    assert result["metadata"]["original_duration_s"] == 12.34
    assert result["metadata"]["processed_duration_s"] == 10.0
    assert result["metadata"]["speedup_factor"] == 1.25
    assert result["metadata"]["cache_hit"] is True
    assert result["metadata"]["timestamp"] == "2026-02-17T00:00:00+00:00"
    assert result["metadata"]["demux_fallback"] is True


def test_build_url_transcript_result_defaults_abort_reason_and_demux() -> None:
    transcript = SimpleNamespace(
        text="hello world",
        aborted=False,
        abort_reason=None,
        cache_hit=True,
        confidence=None,
        confidence_status="unknown",
        language_detected=None,
        language_confidence=None,
    )
    metadata = SimpleNamespace(
        source_type="youtube",
        url="https://www.youtube.com/watch?v=abc",
        title="video",
        uploader="bob",
        upload_date="",
        duration_seconds=42.5,
    )
    # Intentionally omit demux_fallback to verify default False.
    download = SimpleNamespace(
        metadata=metadata,
        cache_hit=False,
        timestamp=datetime(2026, 2, 17, tzinfo=timezone.utc),
    )
    pre = SimpleNamespace(duration_out=42.5, atempo_applied=False)

    result = build_url_transcript_result(
        transcript=transcript,
        download=download,
        pre=pre,
        atempo_factor=1.25,
    )

    assert result["partial"] is False
    assert result["abort_reason"] == ""
    assert result["metadata"]["speedup_factor"] == 1.0
    assert result["metadata"]["cache_hit"] is True
    assert result["metadata"]["demux_fallback"] is False
