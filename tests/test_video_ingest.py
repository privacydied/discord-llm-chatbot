"""
Tests for video ingestion and URL-based audio processing.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from bot.video_ingest import (
    VideoIngestionManager,
    VideoMetadata,
    DownloadedAudio,
    ProcessedAudio,
    VideoIngestError,
)
from bot.hear import hear_infer_from_url
from bot.exceptions import InferenceError


class TestVideoIngestionManager:
    """Test cases for VideoIngestionManager."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Create VideoIngestionManager with temporary cache."""
        with patch("bot.video_ingest.CACHE_DIR", temp_cache_dir):
            return VideoIngestionManager()

    def test_cache_key_generation(self, manager):
        """Test cache key generation is deterministic."""
        url1 = "https://youtube.com/watch?v=test123"
        url2 = "https://youtube.com/watch?v=test456"

        key1a = manager._hash_resolved_url(url1)
        key1b = manager._hash_resolved_url(url1)
        key2 = manager._hash_resolved_url(url2)

        assert key1a == key1b  # Same URL should produce same key
        assert key1a != key2  # Different URLs should produce different keys
        assert len(key1a) == 16  # Should be 16 characters (truncated SHA256)

    def test_supported_url_detection(self, manager):
        """Test URL pattern matching."""
        supported_urls = [
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=test123",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://tiktok.com/@user/video/123456789",
            "https://www.tiktok.com/@user/video/123456789",
            "https://vm.tiktok.com/abc123",
            "https://vimeo.com/123456",
            "https://instagram.com/reel/abc123def",
            "https://www.facebook.com/user/videos/1234567890/",
        ]

        unsupported_urls = [
            "not-a-url",
            "https://example.com",
            "ftp://youtube.com/watch?v=test",
        ]

        for url in supported_urls:
            assert manager._is_supported_url(url), f"Should support: {url}"

        for url in unsupported_urls:
            assert not manager._is_supported_url(url), f"Should not support: {url}"

    def test_source_type_detection(self, manager):
        """Test source type detection from URLs."""
        youtube_urls = [
            "https://youtube.com/watch?v=test",
            "https://www.youtube.com/watch?v=test",
            "https://youtu.be/test",
        ]

        tiktok_urls = [
            "https://tiktok.com/@user/video/123",
            "https://www.tiktok.com/@user/video/123",
            "https://vm.tiktok.com/abc123",
        ]

        for url in youtube_urls:
            assert manager._get_source_type(url) == "youtube"

        for url in tiktok_urls:
            assert manager._get_source_type(url) == "tiktok"

    def test_cache_index_setup(self, manager):
        """Test cache index initialization."""
        assert manager.cache_index_path.exists()

        with open(manager.cache_index_path, "r") as f:
            index = json.load(f)

        assert isinstance(index, dict)
        assert len(index) == 0  # Should start empty

    @pytest.mark.asyncio
    async def test_unsupported_url_error(self, manager):
        """Test error handling for unsupported URLs."""
        unsupported_url = "https://example.com/123456"

        with pytest.raises(VideoIngestError, match="Unsupported URL format"):
            await manager.fetch_and_prepare_url_audio(unsupported_url)

    @pytest.mark.asyncio
    async def test_ytdlp_probe_success(self, manager):
        """Test successful yt-dlp metadata probe parsing."""
        url = "https://youtube.com/watch?v=test123"
        payload = {
            "id": "test123",
            "title": "Test Video Title",
            "duration": 120.5,
            "uploader": "Test Uploader",
            "upload_date": "20240101",
            "extractor_key": "youtube",
            "webpage_url": url,
            "formats": [],
        }

        with patch.object(
            manager,
            "_run_subprocess",
            new=AsyncMock(return_value=(json.dumps(payload).encode(), b"")),
        ):
            metadata = await manager._probe_metadata(url, timeout_s=1.0)

        assert metadata["id"] == "test123"
        assert metadata["title"] == "Test Video Title"
        assert float(metadata["duration"]) == 120.5
        assert metadata["uploader"] == "Test Uploader"

    @pytest.mark.asyncio
    async def test_ytdlp_probe_failure(self, manager):
        """Test yt-dlp metadata probe failure handling."""
        url = "https://youtube.com/watch?v=invalid"

        with patch.object(
            manager,
            "_run_subprocess",
            new=AsyncMock(side_effect=VideoIngestError("yt-dlp metadata probe failed: nope")),
        ):
            with pytest.raises(VideoIngestError, match="yt-dlp metadata probe failed"):
                await manager._probe_metadata(url, timeout_s=1.0)

    @pytest.mark.asyncio
    async def test_ytdlp_download_audio_command(self, manager):
        """Test yt-dlp download orchestration emits expected command shape."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            fake_download = output_dir / "test.m4a"
            fake_download.write_bytes(b"dummy")

            mock_run = AsyncMock(return_value=(str(fake_download).encode(), b""))
            with patch.object(manager, "_run_subprocess", new=mock_run):
                path = await manager._download_audio(
                    source_url="https://example.com/audio",
                    format_id="140",
                    ext="m4a",
                    output_dir=output_dir,
                    timeout_s=1.0,
                )

            assert path.exists()
            assert path.suffix == ".m4a"
            assert mock_run.call_count == 1
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "yt-dlp"
            assert "--format" in cmd
            assert "140" in cmd

    def test_cache_entry_validation(self, manager):
        """Test cache entry validation logic."""
        cache_key = "test123"
        raw_path = manager.cache_dir / "test.m4a"
        entry = {
            "raw_path": str(raw_path),
            "content_length": 123,
            "format_id": "140",
            "ext": "m4a",
            "source_url": "https://youtube.com/watch?v=test",
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "demux_fallback": False,
        }

        manager._index = {cache_key: entry}
        manager._save_cache_index()

        # Missing artifact -> None
        assert manager._get_cache_entry(cache_key) is None

        # _get_cache_entry purges missing artifacts from the index, so restore it.
        manager._index = {cache_key: entry}
        manager._save_cache_index()

        raw_path.touch()
        found = manager._get_cache_entry(cache_key)
        assert found is not None
        found_entry, found_path = found
        assert found_entry["format_id"] == "140"
        assert found_path == raw_path


class TestHearInferFromUrl:
    """Test cases for hear_infer_from_url function."""

    @pytest.mark.asyncio
    @patch("bot.hear.fetch_and_prepare_url_audio")
    async def test_successful_transcription(self, mock_fetch):
        """Test successful URL transcription."""
        from bot.stt import ModelSpec

        stt_stub = Mock()
        stt_stub.ensure_ready = AsyncMock(return_value=True)
        stt_stub.default_spec = ModelSpec(size="base", compute_type="int8")
        stt_stub.downgrade_spec = Mock(return_value=None)
        stt_stub.cpu_threads = 2

        mock_metadata = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            title="Test Video",
            duration_seconds=120.0,
            uploader="Test User",
            upload_date="20240101",
            source_type="youtube",
        )

        mock_download = DownloadedAudio(
            raw_path=Path("/tmp/test.wav"),
            metadata=mock_metadata,
            download_key="abc123",
            format_id="140",
            resolved_url="https://example.com/audio",
            content_length=123,
            cache_hit=False,
            ext="m4a",
            timestamp=datetime.now(timezone.utc),
        )
        mock_fetch.return_value = mock_download

        pre = Mock()
        pre.duration_in = 10.0
        pre.duration_out = 9.0
        pre.atempo_applied = False
        stream = AsyncMock()
        stream.finalize = AsyncMock()
        stream.abort = AsyncMock()
        pre.stream = stream

        transcript = Mock()
        transcript.text = "This is the transcribed text"
        transcript.cache_hit = False
        transcript.aborted = False
        transcript.abort_reason = None
        transcript.model_spec = stt_stub.default_spec

        with (
            patch("bot.hear.stt_manager", stt_stub),
            patch("bot.hear._preprocess_audio", new=AsyncMock(return_value=pre)),
            patch("bot.hear._run_whisper", new=AsyncMock(return_value=transcript)),
        ):
            result = await hear_infer_from_url("https://youtube.com/watch?v=test")

        assert result["transcription"] == "This is the transcribed text"
        assert result["metadata"]["source"] == "youtube"
        assert result["metadata"]["title"] == "Test Video"
        assert not result["metadata"]["cache_hit"]

        mock_fetch.assert_called_once_with(
            "https://youtube.com/watch?v=test", force_refresh=False
        )

    @pytest.mark.asyncio
    async def test_stt_unavailable_error(self):
        """Test error when STT is not available."""
        stt_stub = Mock()
        stt_stub.ensure_ready = AsyncMock(return_value=False)

        with patch("bot.hear.stt_manager", stt_stub):
            with pytest.raises(InferenceError, match="STT engine not available"):
                await hear_infer_from_url("https://youtube.com/watch?v=test")

    @pytest.mark.asyncio
    @patch("bot.hear.fetch_and_prepare_url_audio")
    async def test_video_ingest_error_passthrough(self, mock_fetch):
        """Test VideoIngestError is surfaced as an InferenceError with the same message."""
        from bot.stt import ModelSpec

        stt_stub = Mock()
        stt_stub.ensure_ready = AsyncMock(return_value=True)
        stt_stub.default_spec = ModelSpec(size="base", compute_type="int8")
        stt_stub.downgrade_spec = Mock(return_value=None)
        stt_stub.cpu_threads = 2

        mock_fetch.side_effect = VideoIngestError("Unsupported URL format: https://example.com")
        with patch("bot.hear.stt_manager", stt_stub):
            with pytest.raises(InferenceError, match="Unsupported URL format"):
                await hear_infer_from_url("https://youtube.com/watch?v=test")


class TestVideoCommands:
    """Test cases for Discord video commands."""

    @pytest.fixture
    def mock_bot(self):
        """Create mock Discord bot."""
        bot = Mock()
        bot.user = Mock()
        bot.user.id = 12345
        return bot

    @pytest.fixture
    def mock_ctx(self):
        """Create mock Discord context."""
        ctx = Mock()
        ctx.author = Mock()
        ctx.author.id = 67890
        ctx.guild = Mock()
        ctx.guild.id = 11111
        ctx.message = Mock()
        ctx.reply = AsyncMock()
        ctx.typing = AsyncMock().__aenter__ = AsyncMock()
        ctx.typing().__aexit__ = AsyncMock()
        return ctx

    @pytest.mark.asyncio
    async def test_url_extraction_from_message(self):
        """Test URL extraction from Discord message content."""
        from bot.commands.video_commands import VideoCommands

        video_commands = VideoCommands(Mock())

        test_cases = [
            (
                "Check out this video: https://youtube.com/watch?v=test123",
                "https://youtube.com/watch?v=test123",
            ),
            ("https://youtu.be/abc123 is amazing!", "https://youtu.be/abc123"),
            (
                "Look at https://tiktok.com/@user/video/123456789",
                "https://tiktok.com/@user/video/123456789",
            ),
            ("No video URL here", None),
            ("https://vimeo.com/123456 unsupported", None),
        ]

        for content, expected in test_cases:
            result = video_commands._extract_url_from_message(content)
            assert result == expected

    def test_url_type_detection(self):
        """Test URL type detection."""
        from bot.commands.video_commands import VideoCommands

        video_commands = VideoCommands(Mock())

        youtube_urls = [
            "https://youtube.com/watch?v=test",
            "https://youtu.be/test",
        ]

        tiktok_urls = [
            "https://tiktok.com/@user/video/123",
            "https://vm.tiktok.com/abc123",
        ]

        for url in youtube_urls:
            assert video_commands._get_url_type(url) == "YouTube"

        for url in tiktok_urls:
            assert video_commands._get_url_type(url) == "TikTok"


@pytest.mark.integration
class TestVideoIngestionIntegration:
    """Integration tests for the complete video ingestion pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("test_videos").exists(), reason="Test videos not available"
    )
    async def test_full_pipeline_youtube(self):
        """Test complete pipeline with real YouTube video (if available)."""
        # This would require a real short test video
        # Skip in CI/CD environments
        pass

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("test_videos").exists(), reason="Test videos not available"
    )
    async def test_full_pipeline_tiktok(self):
        """Test complete pipeline with real TikTok video (if available)."""
        # This would require a real short test video
        # Skip in CI/CD environments
        pass

    @pytest.mark.asyncio
    async def test_cache_behavior(self):
        """Test caching behavior across multiple requests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            with patch("bot.video_ingest.CACHE_DIR", cache_dir):
                manager = VideoIngestionManager()

                url = "https://youtube.com/watch?v=test"
                metadata = {
                    "id": "testid",
                    "title": "Test Video",
                    "duration": 60.0,
                    "uploader": "Test User",
                    "upload_date": "20240101",
                    "extractor_key": "youtube",
                    "webpage_url": url,
                    "formats": [
                        {
                            "format_id": "140",
                            "ext": "m4a",
                            "acodec": "aac",
                            "vcodec": "none",
                            "abr": 64,
                            "url": "https://example.com/audio.m4a",
                            "filesize": 123,
                        }
                    ],
                }

                async def _fake_download(*_args, **_kwargs) -> Path:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
                        Path(tmp.name).write_bytes(b"dummy")
                        return Path(tmp.name)

                with (
                    patch.object(manager, "_probe_metadata", new=AsyncMock(return_value=metadata)),
                    patch.object(manager, "_download_audio", new=AsyncMock(side_effect=_fake_download)) as mock_download,
                ):
                    result1 = await manager.fetch_and_prepare_url_audio(url)
                    assert not result1.cache_hit

                    result2 = await manager.fetch_and_prepare_url_audio(url)
                    assert result2.cache_hit
                    assert mock_download.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
