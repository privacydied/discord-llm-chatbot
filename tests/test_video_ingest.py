"""Tests for video ingestion utilities and URL-based transcription helpers."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Never
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
        "cached_at": datetime.now(UTC).isoformat(),
    }
    assert manager._get_cache_entry(cache_key) is None

    # Create the file to make the cache entry valid
    manager._index[cache_key] = {
        "raw_path": str(cached_file),
        "cached_at": datetime.now(UTC).isoformat(),
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


def test_instagram_alias_urls_are_supported_for_stt(
    manager: VideoIngestionManager,
) -> None:
    """Instagram mirror hosts should match the existing Instagram STT URL patterns."""
    assert manager._is_supported_url("https://www.kkinstagram.com/reel/DWjQyv_Dt4k/")
    assert manager._is_supported_url("https://d.vxinstagram.com/reel/DWjQyv_Dt4k/")
    assert manager._is_supported_url("https://www.instagram.com/reel/DWjQyv_Dt4k/")
    assert manager._is_supported_url("https://d.vxinstagram.com/offload/DVMPNJtE50x/0.mp4")


def test_instagram_alias_urls_canonicalize_to_instagram_before_ytdlp() -> None:
    """Kkinstagram mirror URLs should be passed to yt-dlp as instagram.com URLs."""
    expected = "https://www.instagram.com/reel/DWjQyv_Dt4k/"

    assert VideoIngestionManager._canonicalize_instagram_url_for_ytdlp("https://www.kkinstagram.com/reel/DWjQyv_Dt4k/") == expected
    assert VideoIngestionManager._canonicalize_instagram_url_for_ytdlp("https://d.vxinstagram.com/reel/DWjQyv_Dt4k/") == "https://d.vxinstagram.com/reel/DWjQyv_Dt4k/"
    assert VideoIngestionManager._canonicalize_instagram_url_for_ytdlp(expected) == expected


def test_instagram_alias_canonicalization_preserves_safe_query() -> None:
    """Canonicalization should preserve useful query params and drop fragments."""
    assert (
        VideoIngestionManager._canonicalize_instagram_url_for_ytdlp("https://www.kkinstagram.com/p/ABC123/?img_index=2&utm_source=ig_web_copy_link#frag")
        == "https://www.instagram.com/p/ABC123/?img_index=2&utm_source=ig_web_copy_link"
    )


def test_instagram_alias_unsupported_paths_and_random_hosts_rejected(
    manager: VideoIngestionManager,
) -> None:
    """Alias support should not promote unrelated paths or random hosts into STT."""
    assert not manager._is_supported_url("https://www.kkinstagram.com/accounts/login/")
    assert not manager._is_supported_url("https://d.vxinstagram.com/explore/tags/guitar/")
    assert not manager._is_supported_url("https://random.example.com/reel/DWjQyv_Dt4k/")
    assert VideoIngestionManager._canonicalize_instagram_url_for_ytdlp("https://www.kkinstagram.com/accounts/login/") == "https://www.kkinstagram.com/accounts/login/"


@pytest.mark.asyncio
async def test_fetch_uses_canonical_instagram_url_for_ytdlp(manager: VideoIngestionManager, tmp_path: Path, monkeypatch) -> None:
    """fetch_and_prepare_url_audio should invoke yt-dlp probe/download with canonical kkinstagram URL."""
    seen = {"probe": None, "download": None}
    canonical = "https://www.instagram.com/reel/DWjQyv_Dt4k/"

    async def fake_probe(url: str, timeout_s: float):
        seen["probe"] = url
        return {
            "id": "DWjQyv_Dt4k",
            "extractor_key": "Instagram",
            "title": "Instagram Reel",
            "duration": 12,
            "uploader": "tester",
            "upload_date": "20260507",
            "webpage_url": canonical,
            "formats": [
                {
                    "format_id": "audio",
                    "ext": "m4a",
                    "vcodec": "none",
                    "acodec": "mp4a",
                    "url": "https://cdn.example.com/audio.m4a",
                    "filesize": 4,
                },
            ],
        }

    async def fake_download(source_url, format_id, ext, output_dir, timeout_s):
        seen["download"] = source_url
        output = Path(output_dir) / f"DWjQyv_Dt4k.{ext}"
        output.write_bytes(b"data")
        return output

    monkeypatch.setattr(manager, "_probe_metadata", fake_probe)
    monkeypatch.setattr(manager, "_download_audio", fake_download)

    result = await manager.fetch_and_prepare_url_audio(
        "https://www.kkinstagram.com/reel/DWjQyv_Dt4k/",
        force_refresh=True,
    )

    assert seen == {"probe": canonical, "download": canonical}
    assert result.metadata.url == "https://www.kkinstagram.com/reel/DWjQyv_Dt4k/"
    assert result.metadata.source_type == "instagram"


@pytest.mark.asyncio
async def test_long_youtube_bypasses_audio_duration_cap(
    manager: VideoIngestionManager, tmp_path: Path, monkeypatch,
) -> None:
    """A long YouTube (e.g. 45m) must NOT be rejected by the 10m audio cap.

    YouTube has transcript-first paths (summarize CLI, caption resolver in
    hear.py) that run before this gate, and is streamable/low-risk, so the
    flat MAX_DURATION_SECONDS ceiling does not apply to it. [REH][IV]
    """
    url = "https://www.youtube.com/watch?v=longvideo123"
    youtube_id = "longvideo123"

    async def fake_probe(url_arg: str, timeout_s: float):
        return {
            "id": youtube_id,
            "extractor_key": "Youtube",
            "title": "A Very Long YouTube Video",
            "duration": 2700,  # 45 minutes
            "uploader": "tester",
            "upload_date": "20260507",
            "webpage_url": url,
            "formats": [
                {
                    "format_id": "audio",
                    "ext": "m4a",
                    "vcodec": "none",
                    "acodec": "mp4a",
                    "url": "https://cdn.example.com/audio.m4a",
                    "filesize": 4,
                },
            ],
        }

    downloaded = {}

    async def fake_download(source_url, format_id, ext, output_dir, timeout_s):
        downloaded["url"] = source_url
        output = Path(output_dir) / f"{youtube_id}.{ext}"
        output.write_bytes(b"data")
        return output

    monkeypatch.setattr(manager, "_probe_metadata", fake_probe)
    monkeypatch.setattr(manager, "_download_audio", fake_download)

    # Must not raise VideoIngestError despite 45m > 10m default cap.
    result = await manager.fetch_and_prepare_url_audio(url, force_refresh=True)
    assert result.metadata.source_type == "youtube"
    assert downloaded.get("url") == url


@pytest.mark.asyncio
async def test_long_non_youtube_still_rejected_by_duration_cap(
    manager: VideoIngestionManager, tmp_path: Path, monkeypatch,
) -> None:
    """Non-YouTube (TikTok) long videos keep the flat 10m audio cap. [REH][IV]"""
    url = "https://www.tiktok.com/@user/video/longtiktok123"

    async def fake_probe(url_arg: str, timeout_s: float):
        return {
            "id": "longtiktok123",
            "extractor_key": "TikTok",
            "title": "A Very Long TikTok",
            "duration": 2700,  # 45 minutes
            "uploader": "tester",
            "upload_date": "20260507",
            "webpage_url": url,
            "formats": [
                {
                    "format_id": "audio",
                    "ext": "m4a",
                    "vcodec": "none",
                    "acodec": "mp4a",
                    "url": "https://cdn.example.com/audio.m4a",
                    "filesize": 4,
                },
            ],
        }

    monkeypatch.setattr(manager, "_probe_metadata", fake_probe)

    with pytest.raises(VideoIngestError, match="Video too long"):
        await manager.fetch_and_prepare_url_audio(url, force_refresh=True)


@pytest.mark.asyncio
async def test_youtube_duration_cap_disabled_via_env(
    manager: VideoIngestionManager, tmp_path: Path, monkeypatch,
) -> None:
    """VIDEO_MAX_DURATION_YOUTUBE=0 disables the ceiling entirely for YouTube."""
    # Patch the module global in place (no reload — reload leaks module state
    # into sibling tests in the same session). [REH]
    monkeypatch.setattr(
        "bot.video_ingest.MAX_DURATION_SECONDS_YOUTUBE", 0
    )
    monkeypatch.setattr(manager, "_probe_metadata", lambda u, t: _yt_probe_coro(7200, u))
    downloaded = {}
    async def _fake_dl_async(su, fi, ex, od, ts):
        return _fake_dl(downloaded, su, od, "yt0")
    monkeypatch.setattr(
        manager,
        "_download_audio",
        _fake_dl_async,
    )

    result = await manager.fetch_and_prepare_url_audio(
        "https://www.youtube.com/watch?v=yt0", force_refresh=True
    )
    assert result.metadata.source_type == "youtube"
    assert downloaded.get("url") == "https://www.youtube.com/watch?v=yt0"


def _yt_probe_coro(duration: float, url_arg: str):
    async def _probe():
        return {
            "id": "yt0",
            "extractor_key": "Youtube",
            "title": "Long",
            "duration": duration,
            "uploader": "tester",
            "upload_date": "20260507",
            "webpage_url": url_arg,
            "formats": [
                {
                    "format_id": "audio",
                    "ext": "m4a",
                    "vcodec": "none",
                    "acodec": "mp4a",
                    "url": "https://cdn.example.com/audio.m4a",
                    "filesize": 4,
                },
            ],
        }

    return _probe()


def _fake_dl(seen: dict, source_url: str, output_dir: Path, vid: str):
    seen["url"] = source_url
    output = Path(output_dir) / f"{vid}.m4a"
    output.write_bytes(b"data")
    return output


@pytest.mark.asyncio
async def test_vxinstagram_reel_resolves_to_direct_media_without_instagram_ytdlp(manager: VideoIngestionManager, tmp_path: Path, monkeypatch) -> None:
    """d.vxinstagram reel pages should use their og:video direct MP4 to avoid Instagram login walls."""
    page_url = "https://d.vxinstagram.com/reel/DVMPNJtE50x/"
    media_url = "https://d.vxinstagram.com/offload/DVMPNJtE50x/0.mp4"
    seen = {"probe": False, "resolved": None, "direct_download": None}

    async def fake_resolve(url_arg: str, timeout_s: float):
        seen["resolved"] = url_arg
        return media_url

    async def fake_probe(url_arg: str, timeout_s: float) -> Never:
        seen["probe"] = True
        msg = "yt-dlp probe should not run for resolved vxinstagram media"
        raise AssertionError(msg)

    async def fake_download_direct(media_url_arg: str, ext: str, timeout_s: float):
        seen["direct_download"] = media_url_arg
        output = tmp_path / f"direct.{ext}"
        output.write_bytes(b"data")
        return output, 4

    monkeypatch.setattr(manager, "_resolve_vxinstagram_direct_media_url", fake_resolve)
    monkeypatch.setattr(manager, "_probe_metadata", fake_probe)
    monkeypatch.setattr(manager, "_download_direct_media", fake_download_direct)

    result = await manager.fetch_and_prepare_url_audio(page_url, force_refresh=True)

    assert seen == {"probe": False, "resolved": page_url, "direct_download": media_url}
    assert result.metadata.url == page_url
    assert result.metadata.source_type == "instagram"
    assert result.resolved_url == media_url
    assert result.format_id == "direct"
    assert result.ext == "mp4"
    assert result.demux_fallback is True


def test_vxinstagram_og_video_extraction_is_strict() -> None:
    html = """
    <meta property="og:video" content="https://d.vxinstagram.com/offload/DVMPNJtE50x/0.mp4" />
    """
    assert VideoIngestionManager._extract_vxinstagram_direct_media_url(html) == "https://d.vxinstagram.com/offload/DVMPNJtE50x/0.mp4"
    assert VideoIngestionManager._extract_vxinstagram_direct_media_url('<meta property="og:video" content="https://evil.example/offload/x.mp4" />') is None


@pytest.mark.asyncio
async def test_vxinstagram_media_capability_short_circuits_ytdlp_probe(
    monkeypatch,
) -> None:
    from bot.media_capability import MediaCapabilityDetector

    async def fail_create_subprocess_exec(*_args, **_kwargs) -> Never:
        msg = "yt-dlp probe should not run for vxinstagram media pages"
        raise AssertionError(msg)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_create_subprocess_exec)

    capable, reason = await MediaCapabilityDetector()._probe_url_lightweight("https://d.vxinstagram.com/reel/DVMPNJtE50x/")

    assert capable is True
    assert reason == "vxinstagram media page"


@pytest.mark.asyncio
async def test_vxinstagram_offload_mp4_uses_direct_media_fallback(manager: VideoIngestionManager, tmp_path: Path, monkeypatch) -> None:
    """d.vxinstagram.com/offload MP4 links should enter existing direct media STT ingestion."""
    url = "https://d.vxinstagram.com/offload/DVMPNJtE50x/0.mp4"
    seen = {"metadata_probe": None, "direct_download": None}

    async def fake_probe(url_arg: str, timeout_s: float) -> Never:
        seen["metadata_probe"] = url_arg
        msg = "metadata unavailable"
        raise VideoIngestError(msg)

    async def fake_download_direct(media_url: str, ext: str, timeout_s: float):
        seen["direct_download"] = media_url
        output = tmp_path / f"direct.{ext}"
        output.write_bytes(b"data")
        return output, 4

    monkeypatch.setattr(manager, "_probe_metadata", fake_probe)
    monkeypatch.setattr(manager, "_download_direct_media", fake_download_direct)

    result = await manager.fetch_and_prepare_url_audio(url, force_refresh=True)

    assert seen == {"metadata_probe": url, "direct_download": url}
    assert result.metadata.url == url
    assert result.metadata.source_type == "instagram"
    assert result.format_id == "direct"
    assert result.ext == "mp4"
    assert result.demux_fallback is True


# ---------------------------------------------------------------------------
# hear_infer_from_url helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_hear(monkeypatch, tmp_path: Path) -> tuple[SimpleNamespace, DownloadedAudio]:
    """Stub expensive helpers in bot.hear to make hear_infer_from_url deterministic.
    Returns the module reference and the DownloadedAudio instance used by fetch stub.
    """
    from bot import hear

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
        timestamp=datetime.now(UTC),
        demux_fallback=False,
    )

    # Lightweight RAM guard and job implementations
    class DummyGuard:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def check(self, *_args, **_kwargs) -> None:
            return None

    class DummyJob:
        def __init__(self, *_args, **_kwargs) -> None:
            self.download = None
            self.pre = None

        def register_download(self, download_obj) -> None:
            self.download = download_obj

        def register_pre(self, pre) -> None:
            self.pre = pre

        async def finish_success(self, payload):
            return payload

        async def finish_failure(self, exc) -> Never:
            raise exc

        async def close(self) -> None:
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

    async def fake_preprocess(source_path, spans, download=None, voice_note=False, ram_guard=None):
        return SimpleNamespace(
            source_path=source_path,
            duration_in=120.0,
            duration_out=80.0,
            atempo_applied=True,
            cache_hit=False,
            stream=None,
        )

    async def fake_run_whisper(pre, spans, spec_obj, ram_guard, job=None, language=None):
        return SimpleNamespace(
            text="This is the transcribed text",
            aborted=False,
            abort_reason="",
            cache_hit=False,
            model_spec=spec,
            chunks=[],
            confidence=None,
            confidence_status="unknown",
            language_detected=None,
            language_confidence=None,
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

    async def failing_fetch(url: str, force_refresh: bool = False) -> Never:
        msg = "download failed"
        raise VideoIngestError(msg)

    monkeypatch.setattr(hear, "fetch_and_prepare_url_audio", failing_fetch)

    with pytest.raises(InferenceError, match="download failed"):
        await hear_infer_from_url("https://youtube.com/watch?v=test")


if __name__ == "__main__":  # pragma: no cover - convenience for local runs
    pytest.main([__file__, "-q"])
