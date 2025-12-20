"""
Tests for video ingestion utilities and URL-based transcription helpers.
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import pytest

from bot.exceptions import InferenceError
from bot.hear import hear_infer_from_url
from bot.video_ingest import (
    DownloadedAudio,
    VideoIngestError,
    VideoIngestionManager,
    VideoMetadata,
)


# ---------------------------------------------------------------------------
# Video ingestion manager helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory for testing."""
    return tmp_path / "cache"


@pytest.fixture
def manager(temp_cache_dir: Path) -> VideoIngestionManager:
    """Create a VideoIngestionManager that uses an isolated cache directory."""
    with patch("bot.video_ingest.CACHE_DIR", temp_cache_dir):
        return VideoIngestionManager()


def test_compute_download_key(manager: VideoIngestionManager) -> None:
    """Download keys should be deterministic and incorporate identity details."""
    key1 = manager._compute_download_key(
        "https://example.com/video/audio.m4a",
        "140",
        12345,
        original_url="https://youtube.com/watch?v=test123",
    )
    key2 = manager._compute_download_key(
        "https://example.com/video/audio.m4a",
        "140",
        12345,
        original_url="https://youtube.com/watch?v=test123",
    )
    different = manager._compute_download_key(
        "https://example.com/other/audio.m4a",
        "251",
        54321,
        original_url="https://youtube.com/watch?v=test456",
    )

    assert key1 == key2
    assert key1 != different
    assert "-v" in key1  # Video identity hash should be appended


def test_supported_url_detection(manager: VideoIngestionManager) -> None:
    """Supported URL detection should align with configured patterns."""
    supported = [
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.tiktok.com/@user/video/123456789",
        "https://x.com/someuser/status/1234567890",
    ]
    unsupported = [
        "not-a-url",
        "ftp://example.com/video.mp4",
    ]

    for url in supported:
        assert manager._is_supported_url(url), f"Expected support for {url}"

    for url in unsupported:
        assert not manager._is_supported_url(url), f"Expected no support for {url}"


def test_source_type_detection(manager: VideoIngestionManager) -> None:
    """Source type detection should map common domains correctly."""
    assert manager._get_source_type("https://youtube.com/watch?v=test") == "youtube"
    assert manager._get_source_type("https://youtu.be/test") == "youtube"
    assert manager._get_source_type("https://www.tiktok.com/@user/video/1") == "tiktok"


def test_cache_entry_validation(manager: VideoIngestionManager, tmp_path: Path) -> None:
    """Cache entries should only resolve when the referenced file exists."""
    cache_key = "test-key"
    cached_file = tmp_path / "audio.wav"

    # Missing file should return None
    manager._index[cache_key] = {
        "raw_path": str(cached_file),
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    assert manager._get_cache_entry(cache_key) is None

    # Create the file to make the cache entry valid
    manager._index[cache_key] = {
        "raw_path": str(cached_file),
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    cached_file.touch()
    entry, path = manager._get_cache_entry(cache_key) or ({}, None)

    assert path == cached_file
    assert entry["raw_path"] == str(cached_file)


@pytest.mark.asyncio
async def test_fetch_and_prepare_url_audio_invalid(
    manager: VideoIngestionManager,
) -> None:
    """An unsupported URL should raise a VideoIngestError."""
    with pytest.raises(VideoIngestError):
        await manager.fetch_and_prepare_url_audio("invalid")


# ---------------------------------------------------------------------------
# hear_infer_from_url helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_hear(monkeypatch, tmp_path: Path) -> Tuple[SimpleNamespace, DownloadedAudio]:
    """
    Stub expensive helpers in bot.hear to make hear_infer_from_url deterministic.
    Returns the module reference and the DownloadedAudio instance used by fetch stub.
    """
    import bot.hear as hear

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"data")

    metadata = VideoMetadata(
        url="https://youtube.com/watch?v=test",
        title="Test Video",
        duration_seconds=120.0,
        uploader="Test User",
        upload_date="20240101",
        source_type="youtube",
    )
    download = DownloadedAudio(
        raw_path=audio_path,
        metadata=metadata,
        download_key="key",
        format_id="140",
        resolved_url="https://example.com/audio.m4a",
        content_length=4,
        cache_hit=False,
        ext="m4a",
        timestamp=datetime.now(timezone.utc),
        demux_fallback=False,
    )

    # Lightweight RAM guard and job implementations
    class DummyGuard:
        def __init__(self, *_args, **_kwargs):
            pass

        def check(self, *_args, **_kwargs):
            return None

    class DummyJob:
        def __init__(self, *_args, **_kwargs):
            self.download = None
            self.pre = None

        def register_download(self, download_obj):
            self.download = download_obj

        def register_pre(self, pre):
            self.pre = pre

        async def finish_success(self, payload):
            return payload

        async def finish_failure(self, exc):
            raise exc

        async def close(self):
            return None

    monkeypatch.setattr(hear, "STTRAMGuard", DummyGuard)
    monkeypatch.setattr(hear, "STTJob", DummyJob)

    spec = SimpleNamespace(size="base", compute_type="int8")
    monkeypatch.setattr(
        hear,
        "stt_manager",
        SimpleNamespace(
            is_available=lambda: True,
            default_spec=spec,
            downgrade_spec=lambda _spec: None,
            cpu_threads=2,
        ),
    )

    async def fake_preprocess(
        source_path, spans, download=None, voice_note=False, ram_guard=None
    ):
        return SimpleNamespace(
            duration_in=120.0,
            duration_out=80.0,
            atempo_applied=True,
            cache_hit=False,
            stream=None,
        )

    async def fake_run_whisper(pre, spans, spec_obj, ram_guard, job=None):
        return SimpleNamespace(
            text="This is the transcribed text",
            aborted=False,
            abort_reason="",
            cache_hit=False,
            model_spec=spec,
            chunks=[],
        )

    async def fake_fetch(url: str, force_refresh: bool = False):
        return download

    monkeypatch.setattr(hear, "_preprocess_audio", fake_preprocess)
    monkeypatch.setattr(hear, "_run_whisper", fake_run_whisper)
    monkeypatch.setattr(hear, "fetch_and_prepare_url_audio", fake_fetch)

    return hear, download


@pytest.mark.asyncio
async def test_hear_infer_from_url_success(stub_hear) -> None:
    """hear_infer_from_url should return transcription and metadata when dependencies succeed."""
    _, download = stub_hear
    result = await hear_infer_from_url(download.metadata.url)

    assert result["transcription"] == "This is the transcribed text"
    assert result["metadata"]["source"] == "youtube"
    assert result["metadata"]["title"] == "Test Video"
    assert result["metadata"]["cache_hit"] is False


@pytest.mark.asyncio
async def test_hear_infer_from_url_stt_unavailable(monkeypatch, stub_hear) -> None:
    """If STT is unavailable, the helper should raise a user-facing error."""
    hear, _ = stub_hear
    monkeypatch.setattr(
        hear,
        "stt_manager",
        SimpleNamespace(is_available=lambda: False, cpu_threads=2),
    )

    with pytest.raises(InferenceError, match="STT engine not available"):
        await hear_infer_from_url("https://youtube.com/watch?v=test")


@pytest.mark.asyncio
async def test_hear_infer_from_url_video_ingest_error(monkeypatch, stub_hear) -> None:
    """Video ingestion errors should surface as InferenceError for callers."""
    hear, _ = stub_hear

    async def failing_fetch(url: str, force_refresh: bool = False):
        raise VideoIngestError("download failed")

    monkeypatch.setattr(hear, "fetch_and_prepare_url_audio", failing_fetch)

    with pytest.raises(InferenceError, match="download failed"):
        await hear_infer_from_url("https://youtube.com/watch?v=test")


if __name__ == "__main__":  # pragma: no cover - convenience for local runs
    pytest.main([__file__, "-q"])
