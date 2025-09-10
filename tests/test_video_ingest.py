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

        key1a = manager._get_cache_key(url1)
        key1b = manager._get_cache_key(url1)
        key2 = manager._get_cache_key(url2)

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
        ]

        unsupported_urls = [
            "https://vimeo.com/123456",
            "https://facebook.com/video/123",
            "https://instagram.com/p/abc123",
            "not-a-url",
            "https://example.com",
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
        unsupported_url = "https://vimeo.com/123456"

        with pytest.raises(VideoIngestError, match="Unsupported URL format"):
            await manager.fetch_and_prepare_url_audio(unsupported_url)

    @pytest.mark.asyncio
    @patch("bot.video_ingest.asyncio.create_subprocess_exec")
    async def test_ytdlp_download_success(self, mock_subprocess, manager):
        """Test successful yt-dlp download."""
        # Mock successful yt-dlp process
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (
            b"/tmp/test.wav\nTest Video Title\n120.5\nTest Uploader\n20240101\n",
            b"",
        )
        mock_subprocess.return_value = mock_proc

        url = "https://youtube.com/watch?v=test123"

        with patch.object(Path, "exists", return_value=True):
            metadata, filepath = await manager._download_with_ytdlp(url, Path("/tmp"))

        assert metadata.url == url
        assert metadata.title == "Test Video Title"
        assert metadata.duration_seconds == 120.5
        assert metadata.uploader == "Test Uploader"
        assert metadata.source_type == "youtube"
        assert filepath == Path("/tmp/test.wav")

    @pytest.mark.asyncio
    @patch("bot.video_ingest.asyncio.create_subprocess_exec")
    async def test_ytdlp_download_failure(self, mock_subprocess, manager):
        """Test yt-dlp download failure handling."""
        # Mock failed yt-dlp process
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"Video not available")
        mock_subprocess.return_value = mock_proc

        url = "https://youtube.com/watch?v=invalid"

        with pytest.raises(VideoIngestError, match="yt-dlp download failed"):
            await manager._download_with_ytdlp(url, Path("/tmp"))

    @pytest.mark.asyncio
    @patch("bot.video_ingest.asyncio.create_subprocess_exec")
    async def test_audio_processing_success(self, mock_subprocess, manager):
        """Test successful audio processing with ffmpeg."""
        # Mock successful ffmpeg process
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")
        mock_subprocess.return_value = mock_proc

        input_path = Path("/tmp/input.wav")

        with patch.object(Path, "exists", return_value=True):
            result_path = await manager._process_audio(input_path, speedup=1.5)

        expected_path = input_path.with_suffix(".processed.wav")
        assert result_path == expected_path

        # Verify ffmpeg command
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "ffmpeg" in args
        assert "-ar" in args and "16000" in args  # 16kHz
        assert "-ac" in args and "1" in args  # Mono
        assert "atempo=1.5" in " ".join(args)  # Speedup

    def test_cache_entry_validation(self, manager):
        """Test cache entry validation logic."""
        # Create mock cache entry
        cache_key = "test123"
        cache_entry = {
            "url": "https://youtube.com/watch?v=test",
            "title": "Test Video",
            "duration_seconds": 120.0,
            "uploader": "Test User",
            "upload_date": "20240101",
            "source_type": "youtube",
            "processed_path": str(manager.cache_dir / "test.wav"),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "speedup_factor": 1.5,
        }

        # Write to cache index
        with open(manager.cache_index_path, "w") as f:
            json.dump({cache_key: cache_entry}, f)

        # Test with missing file
        result = manager._get_cached_entry(cache_key)
        assert result is None  # Should return None if file doesn't exist

        # Test with existing file
        cache_file = Path(cache_entry["processed_path"])
        cache_file.touch()  # Create empty file

        result = manager._get_cached_entry(cache_key)
        assert result == cache_entry


class TestHearInferFromUrl:
    """Test cases for hear_infer_from_url function."""

    @pytest.mark.asyncio
    @patch("bot.hear.stt_manager")
    @patch("bot.hear.fetch_and_prepare_url_audio")
    async def test_successful_transcription(self, mock_fetch, mock_stt):
        """Test successful URL transcription."""
        # Mock STT manager
        mock_stt.is_available.return_value = True
        mock_stt.transcribe_async.return_value = "This is the transcribed text"

        # Mock processed audio
        mock_metadata = VideoMetadata(
            url="https://youtube.com/watch?v=test",
            title="Test Video",
            duration_seconds=120.0,
            uploader="Test User",
            upload_date="20240101",
            source_type="youtube",
        )

        mock_processed = ProcessedAudio(
            audio_path=Path("/tmp/test.wav"),
            metadata=mock_metadata,
            processed_duration_seconds=80.0,
            speedup_factor=1.5,
            cache_hit=False,
            timestamp=datetime.now(timezone.utc),
        )

        mock_fetch.return_value = mock_processed

        # Test the function
        result = await hear_infer_from_url("https://youtube.com/watch?v=test")

        assert result["transcription"] == "This is the transcribed text"
        assert result["metadata"]["source"] == "youtube"
        assert result["metadata"]["title"] == "Test Video"
        assert not result["metadata"]["cache_hit"]

        # Verify calls
        mock_fetch.assert_called_once_with(
            "https://youtube.com/watch?v=test", 1.5, False
        )
        mock_stt.transcribe_async.assert_called_once_with(Path("/tmp/test.wav"))

    @pytest.mark.asyncio
    @patch("bot.hear.stt_manager")
    async def test_stt_unavailable_error(self, mock_stt):
        """Test error when STT is not available."""
        mock_stt.is_available.return_value = False

        with pytest.raises(InferenceError, match="STT engine not available"):
            await hear_infer_from_url("https://youtube.com/watch?v=test")

    @pytest.mark.asyncio
    @patch("bot.hear.stt_manager")
    @patch("bot.hear.fetch_and_prepare_url_audio")
    async def test_user_friendly_error_messages(self, mock_fetch, mock_stt):
        """Test user-friendly error message conversion."""
        mock_stt.is_available.return_value = True

        test_cases = [
            ("unsupported url format", "This URL is not supported"),
            ("video too long", "This video is too long to process"),
            ("download failed", "Could not download the video"),
            ("audio processing failed", "Could not process the audio"),
            ("unknown error", "Video transcription failed"),
        ]

        for error_input, expected_output in test_cases:
            mock_fetch.side_effect = Exception(error_input)

            with pytest.raises(InferenceError, match=expected_output):
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

                # Mock successful processing
                with (
                    patch.object(manager, "_download_with_ytdlp") as mock_download,
                    patch.object(manager, "_process_audio") as mock_process,
                ):
                    # Setup mocks
                    mock_metadata = VideoMetadata(
                        url="https://youtube.com/watch?v=test",
                        title="Test Video",
                        duration_seconds=60.0,
                        uploader="Test User",
                        upload_date="20240101",
                        source_type="youtube",
                    )

                    mock_download.return_value = (mock_metadata, Path("/tmp/raw.wav"))
                    mock_process.return_value = Path("/tmp/processed.wav")

                    # Mock file operations
                    with (
                        patch.object(Path, "exists", return_value=True),
                        patch.object(Path, "rename"),
                        patch.object(Path, "touch"),
                    ):
                        # First request - should download
                        result1 = await manager.fetch_and_prepare_url_audio(
                            "https://youtube.com/watch?v=test"
                        )
                        assert not result1.cache_hit

                        # Second request - should use cache
                        result2 = await manager.fetch_and_prepare_url_audio(
                            "https://youtube.com/watch?v=test"
                        )
                        assert result2.cache_hit

                        # Verify download was only called once
                        assert mock_download.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
