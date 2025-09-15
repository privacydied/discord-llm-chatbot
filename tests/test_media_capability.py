"""
Tests for media capability detection system.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest

from bot.media_capability import (
    MediaCapabilityDetector,
    ProbeResult,
    is_media_capable_url,
    is_twitter_video_url,
)


class TestMediaCapabilityDetector:
    """Test suite for MediaCapabilityDetector."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def detector(self, temp_cache_dir):
        """Create detector instance with temporary cache."""
        with patch("bot.media_capability.CACHE_DIR", temp_cache_dir):
            detector = MediaCapabilityDetector()
            return detector

    def test_init(self, detector, temp_cache_dir):
        """Test detector initialization."""
        assert detector.cache_dir == temp_cache_dir
        assert detector.cache_file == temp_cache_dir / "probe_cache.json"
        assert isinstance(detector._cache, dict)

    def test_get_cache_key(self, detector):
        """Test cache key generation."""
        url1 = "https://youtube.com/watch?v=test123"
        url2 = "https://youtube.com/watch?v=test456"

        key1 = detector._get_cache_key(url1)
        key2 = detector._get_cache_key(url2)

        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert len(key1) == 16  # SHA256 truncated to 16 chars
        assert len(key2) == 16
        assert key1 != key2  # Different URLs should have different keys

        # Same URL should produce same key
        key1_again = detector._get_cache_key(url1)
        assert key1 == key1_again

    def test_is_whitelisted_domain(self, detector):
        """Test domain whitelist checking."""
        # Whitelisted domains
        assert detector._is_whitelisted_domain("https://youtube.com/watch?v=test")
        assert detector._is_whitelisted_domain("https://www.youtube.com/watch?v=test")
        assert detector._is_whitelisted_domain("https://youtu.be/test")
        assert detector._is_whitelisted_domain("https://tiktok.com/@user/video/123")
        assert detector._is_whitelisted_domain("https://twitter.com/user/status/123")
        assert detector._is_whitelisted_domain("https://x.com/user/status/123")

        # Non-whitelisted domains
        assert not detector._is_whitelisted_domain("https://example.com/video")
        assert not detector._is_whitelisted_domain("https://vimeo.com/123")
        assert not detector._is_whitelisted_domain("https://facebook.com/video")

        # Invalid URLs
        assert not detector._is_whitelisted_domain("not-a-url")
        assert not detector._is_whitelisted_domain("")

    def test_is_cache_valid(self, detector):
        """Test cache validity checking."""
        current_time = time.time()

        # Valid cache entry (recent)
        valid_entry = {"timestamp": current_time - 100}  # 100 seconds ago
        assert detector._is_cache_valid(valid_entry)

        # Invalid cache entry (old)
        invalid_entry = {
            "timestamp": current_time - 400
        }  # 400 seconds ago (> 300s TTL)
        assert not detector._is_cache_valid(invalid_entry)

        # Entry without timestamp
        no_timestamp_entry = {"some_data": "value"}
        assert not detector._is_cache_valid(no_timestamp_entry)

    @pytest.mark.asyncio
    async def test_probe_url_lightweight_success(self, detector):
        """Test successful URL probing."""
        url = "https://youtube.com/watch?v=test123"

        # Mock successful yt-dlp response
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"Test Video Title\n120.5\n", b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=mock_proc):
                is_capable, reason = await detector._probe_url_lightweight(url)

        assert is_capable is True
        assert reason == "media available"

    @pytest.mark.asyncio
    async def test_probe_url_lightweight_failure(self, detector):
        """Test failed URL probing."""
        url = "https://example.com/not-a-video"

        # Mock failed yt-dlp response
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"ERROR: Unsupported URL")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=mock_proc):
                is_capable, reason = await detector._probe_url_lightweight(url)

        assert is_capable is False
        assert "unsupported url format" in reason

    @pytest.mark.asyncio
    async def test_probe_url_lightweight_timeout(self, detector):
        """Test URL probing timeout."""
        url = "https://youtube.com/watch?v=test123"

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            is_capable, reason = await detector._probe_url_lightweight(url)

        assert is_capable is False
        assert reason == "probe timeout"

    @pytest.mark.asyncio
    async def test_probe_url_lightweight_ytdlp_not_found(self, detector):
        """Test handling when yt-dlp is not installed."""
        url = "https://youtube.com/watch?v=test123"

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            is_capable, reason = await detector._probe_url_lightweight(url)

        assert is_capable is False
        assert reason == "yt-dlp not available"

    @pytest.mark.asyncio
    async def test_is_media_capable_domain_not_whitelisted(self, detector):
        """Test capability check for non-whitelisted domain."""
        url = "https://example.com/video"

        result = await detector.is_media_capable(url)

        assert result.is_media_capable is False
        assert result.reason == "domain not whitelisted"
        assert result.cached is False
        assert result.probe_duration_ms == 0

    @pytest.mark.asyncio
    async def test_is_media_capable_cache_hit(self, detector):
        """Test capability check with cache hit."""
        url = "https://youtube.com/watch?v=test123"
        cache_key = detector._get_cache_key(url)

        # Pre-populate cache
        detector._cache[cache_key] = {
            "url": url,
            "is_media_capable": True,
            "reason": "media available",
            "probe_duration_ms": 150.0,
            "timestamp": time.time(),
        }

        result = await detector.is_media_capable(url)

        assert result.is_media_capable is True
        assert result.reason == "media available"
        assert result.cached is True
        assert result.probe_duration_ms == 150.0

    @pytest.mark.asyncio
    async def test_is_media_capable_cache_expired(self, detector):
        """Test capability check with expired cache entry."""
        url = "https://youtube.com/watch?v=test123"
        cache_key = detector._get_cache_key(url)

        # Pre-populate cache with expired entry
        detector._cache[cache_key] = {
            "url": url,
            "is_media_capable": True,
            "reason": "media available",
            "probe_duration_ms": 150.0,
            "timestamp": time.time() - 400,  # Expired (> 300s TTL)
        }

        # Mock fresh probe
        with patch.object(
            detector, "_probe_url_lightweight", return_value=(True, "media available")
        ):
            result = await detector.is_media_capable(url)

        assert result.is_media_capable is True
        assert result.reason == "media available"
        assert result.cached is False
        assert cache_key not in detector._cache  # Expired entry should be removed

    @pytest.mark.asyncio
    async def test_is_media_capable_fresh_probe(self, detector):
        """Test capability check with fresh probe."""
        url = "https://youtube.com/watch?v=test123"

        with patch.object(
            detector, "_probe_url_lightweight", return_value=(True, "media available")
        ):
            result = await detector.is_media_capable(url)

        assert result.is_media_capable is True
        assert result.reason == "media available"
        assert result.cached is False
        assert result.probe_duration_ms is not None
        assert result.probe_duration_ms > 0

        # Check that result was cached
        cache_key = detector._get_cache_key(url)
        assert cache_key in detector._cache
        cached_entry = detector._cache[cache_key]
        assert cached_entry["is_media_capable"] is True
        assert cached_entry["reason"] == "media available"

    @pytest.mark.asyncio
    async def test_is_twitter_video_present_non_twitter_url(self, detector):
        """Test Twitter video detection for non-Twitter URL."""
        url = "https://youtube.com/watch?v=test123"

        result = await detector.is_twitter_video_present(url)

        assert result.is_media_capable is False
        assert result.reason == "not a twitter/x url"

    @pytest.mark.asyncio
    async def test_is_twitter_video_present_success(self, detector):
        """Test successful Twitter video detection."""
        url = "https://twitter.com/user/status/123"

        with patch.object(
            detector,
            "is_media_capable",
            return_value=ProbeResult(
                is_media_capable=True,
                reason="media available",
                cached=False,
                probe_duration_ms=100.0,
            ),
        ):
            result = await detector.is_twitter_video_present(url)

        assert result.is_media_capable is True
        assert result.reason == "twitter video detected via probe"
        assert result.cached is False
        assert result.probe_duration_ms == 100.0

    @pytest.mark.asyncio
    async def test_is_twitter_video_present_no_video(self, detector):
        """Test Twitter video detection when no video is present."""
        url = "https://twitter.com/user/status/123"

        with patch.object(
            detector,
            "is_media_capable",
            return_value=ProbeResult(
                is_media_capable=False,
                reason="no video found",
                cached=False,
                probe_duration_ms=50.0,
            ),
        ):
            result = await detector.is_twitter_video_present(url)

        assert result.is_media_capable is False
        assert "no twitter video found" in result.reason
        assert result.cached is False
        assert result.probe_duration_ms == 50.0

    def test_cleanup_expired_cache(self, detector):
        """Test cleanup of expired cache entries."""
        current_time = time.time()

        # Add mix of valid and expired entries
        detector._cache = {
            "valid1": {"timestamp": current_time - 100},  # Valid
            "valid2": {"timestamp": current_time - 200},  # Valid
            "expired1": {"timestamp": current_time - 400},  # Expired
            "expired2": {"timestamp": current_time - 500},  # Expired
            "no_timestamp": {"some_data": "value"},  # No timestamp (should be removed)
        }

        detector.cleanup_expired_cache()

        # Only valid entries should remain
        assert len(detector._cache) == 2
        assert "valid1" in detector._cache
        assert "valid2" in detector._cache
        assert "expired1" not in detector._cache
        assert "expired2" not in detector._cache
        assert "no_timestamp" not in detector._cache

    def test_get_cache_stats(self, detector):
        """Test cache statistics generation."""
        current_time = time.time()

        detector._cache = {
            "valid1": {"timestamp": current_time - 100},  # Valid
            "valid2": {"timestamp": current_time - 200},  # Valid
            "expired1": {"timestamp": current_time - 400},  # Expired
            "no_timestamp": {"some_data": "value"},  # No timestamp (expired)
        }

        stats = detector.get_cache_stats()

        assert stats["total_entries"] == 4
        assert stats["valid_entries"] == 2
        assert stats["expired_entries"] == 2
        assert stats["cache_hit_potential"] == 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_is_media_capable_url(self):
        """Test is_media_capable_url convenience function."""
        url = "https://youtube.com/watch?v=test123"

        with patch("bot.media_capability.media_detector") as mock_detector:
            mock_detector.is_media_capable.return_value = ProbeResult(
                is_media_capable=True, reason="media available"
            )

            result = await is_media_capable_url(url)

            mock_detector.is_media_capable.assert_called_once_with(url)
            assert result.is_media_capable is True
            assert result.reason == "media available"

    @pytest.mark.asyncio
    async def test_is_twitter_video_url(self):
        """Test is_twitter_video_url convenience function."""
        url = "https://twitter.com/user/status/123"

        with patch("bot.media_capability.media_detector") as mock_detector:
            mock_detector.is_twitter_video_present.return_value = ProbeResult(
                is_media_capable=True, reason="twitter video detected"
            )

            result = await is_twitter_video_url(url)

            mock_detector.is_twitter_video_present.assert_called_once_with(url)
            assert result.is_media_capable is True
            assert result.reason == "twitter video detected"


@pytest.mark.integration
class TestMediaCapabilityIntegration:
    """Integration tests requiring actual yt-dlp."""

    @pytest.mark.asyncio
    async def test_real_youtube_url(self):
        """Test with real YouTube URL (requires yt-dlp)."""
        # Skip if yt-dlp not available
        try:
            import subprocess

            subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("yt-dlp not available")

        detector = MediaCapabilityDetector()

        # Use a known stable YouTube video (YouTube's own channel)
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll (stable test video)

        result = await detector.is_media_capable(url)

        # Should detect as media-capable
        assert result.is_media_capable is True
        assert "media available" in result.reason
        assert result.probe_duration_ms is not None
        assert result.probe_duration_ms > 0

    @pytest.mark.asyncio
    async def test_real_invalid_url(self):
        """Test with real invalid URL (requires yt-dlp)."""
        # Skip if yt-dlp not available
        try:
            import subprocess

            subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("yt-dlp not available")

        detector = MediaCapabilityDetector()

        # Use a URL that looks like YouTube but isn't valid
        url = "https://www.youtube.com/watch?v=invalid123456789"

        result = await detector.is_media_capable(url)

        # Should not detect as media-capable
        assert result.is_media_capable is False
        assert result.probe_duration_ms is not None
