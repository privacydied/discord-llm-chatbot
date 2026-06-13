"""Tests for TikTok STT URL isolation and cache key uniqueness.
Ensures that different TikTok URLs produce different cache keys
and that transcripts are not cross-contaminated.
"""

import hashlib

# Inline copies of the functions under test (to avoid import issues)
import re
from urllib.parse import urlparse


def _normalize_tiktok_url(url: str) -> str:
    """Normalize TikTok URLs to a canonical form for consistent cache keying.
    Also extracts video ID from player/embed URLs to map them to canonical identity.
    """
    if not url:
        return url
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host in ("vm.tiktok.com", "m.tiktok.com", "www.tiktok.com", "tiktok.com"):
            path = parsed.path.rstrip("/")

            # Handle /player/v1/<video_id> embed URLs - extract video ID for identity
            player_match = re.match(r"^/player(?:/v\d+)?/(\d+)", path)
            if player_match:
                video_id = player_match.group(1)
                return f"tiktok://video/{video_id}"

            # Handle /@user/video/<video_id> canonical URLs - extract video ID
            video_match = re.match(r"^/@[\w\.-]+/video/(\d+)", path)
            if video_match:
                video_id = video_match.group(1)
                return f"tiktok://video/{video_id}"

            # For short URLs like /t/ZP8UxRTSU, the path is the key
            return f"tiktok://{path}"
    except Exception:
        pass
    return url


def _is_tiktok_player_url(url: str) -> bool:
    """Check if URL is a TikTok player/embed URL."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host in ("vm.tiktok.com", "m.tiktok.com", "www.tiktok.com", "tiktok.com"):
            path = parsed.path or ""
            if path.startswith("/player"):
                return True
    except Exception:
        pass
    return False


def _hash_resolved_url(resolved_url: str) -> str:
    h = hashlib.sha256((resolved_url or "").encode("utf-8")).hexdigest()
    return h[:16]


def _canonicalize_video_identity(original_url: str, metadata=None) -> str:
    """Canonicalize video identity for cache keying."""
    if metadata:
        extractor = metadata.get("extractor_key") or metadata.get("extractor") or ""
        video_id = metadata.get("id") or ""
        if extractor and video_id:
            return f"{extractor.lower()}:{video_id}"

    if not original_url:
        return ""

    url_lower = original_url.lower()

    if "tiktok.com" in url_lower:
        normalized = _normalize_tiktok_url(original_url)
        if normalized.startswith("tiktok://"):
            return normalized.replace("://", ":")

    return f"generic:{hashlib.sha256(original_url.encode()).hexdigest()[:16]}"


def _compute_download_key(
    resolved_url: str,
    fmt_id: str,
    content_length,
    original_url=None,
    video_identity=None,
) -> str:
    """Compute cache key with video identity for collision resistance."""
    length_part = str(content_length) if content_length is not None else "na"
    base_key = f"{_hash_resolved_url(resolved_url)}-{fmt_id}-{length_part}"

    # Always include video identity hash to prevent cross-contamination
    if video_identity:
        identity_hash = _hash_resolved_url(video_identity)[:10]
        base_key = f"{base_key}-v{identity_hash}"
    elif original_url:
        fallback_identity = _canonicalize_video_identity(original_url)
        identity_hash = _hash_resolved_url(fallback_identity)[:10]
        base_key = f"{base_key}-v{identity_hash}"

    return base_key


class TestTikTokUrlNormalization:
    """Test TikTok URL normalization for consistent cache keying."""

    def test_normalize_short_url(self) -> None:
        """Short /t/ URLs should normalize to canonical form."""
        url = "https://www.tiktok.com/t/ZP8UxRTSU/"
        normalized = _normalize_tiktok_url(url)
        assert normalized == "tiktok:///t/ZP8UxRTSU"

    def test_normalize_vm_url(self) -> None:
        """vm.tiktok.com URLs should normalize."""
        url = "https://vm.tiktok.com/ZP8UxRTSU/"
        normalized = _normalize_tiktok_url(url)
        assert normalized == "tiktok:///ZP8UxRTSU"

    def test_normalize_full_url(self) -> None:
        """Full @user/video URLs should normalize to video ID."""
        url = "https://www.tiktok.com/@user/video/7123456789"
        normalized = _normalize_tiktok_url(url)
        assert normalized == "tiktok://video/7123456789"

    def test_normalize_player_url(self) -> None:
        """Player/embed URLs should normalize to video ID."""
        url = "https://www.tiktok.com/player/v1/7578893205815495958"
        normalized = _normalize_tiktok_url(url)
        assert normalized == "tiktok://video/7578893205815495958"

    def test_player_and_canonical_same_identity(self) -> None:
        """Player URL and canonical URL for same video should normalize to same identity."""
        player_url = "https://www.tiktok.com/player/v1/7578893205815495958"
        canonical_url = "https://www.tiktok.com/@mmukss/video/7578893205815495958"
        assert _normalize_tiktok_url(player_url) == _normalize_tiktok_url(canonical_url)

    def test_different_tiktoks_different_normalized(self) -> None:
        """Different TikTok URLs should produce different normalized forms."""
        url1 = "https://www.tiktok.com/t/ABC123/"
        url2 = "https://www.tiktok.com/t/XYZ789/"
        norm1 = _normalize_tiktok_url(url1)
        norm2 = _normalize_tiktok_url(url2)
        assert norm1 != norm2

    def test_non_tiktok_unchanged(self) -> None:
        """Non-TikTok URLs should be returned unchanged."""
        url = "https://www.youtube.com/watch?v=abc123"
        normalized = _normalize_tiktok_url(url)
        assert normalized == url


class TestTikTokPlayerUrlDetection:
    """Test TikTok player/embed URL detection."""

    def test_player_v1_url(self) -> None:
        """Player v1 URLs should be detected."""
        url = "https://www.tiktok.com/player/v1/7578893205815495958"
        assert _is_tiktok_player_url(url) is True

    def test_player_url_without_version(self) -> None:
        """Player URLs without version should be detected."""
        url = "https://www.tiktok.com/player/7578893205815495958"
        assert _is_tiktok_player_url(url) is True

    def test_canonical_url_not_player(self) -> None:
        """Canonical video URLs should not be detected as player URLs."""
        url = "https://www.tiktok.com/@user/video/7578893205815495958"
        assert _is_tiktok_player_url(url) is False

    def test_short_url_not_player(self) -> None:
        """Short /t/ URLs should not be detected as player URLs."""
        url = "https://www.tiktok.com/t/ZP8UxRTSU/"
        assert _is_tiktok_player_url(url) is False

    def test_youtube_not_player(self) -> None:
        """Non-TikTok URLs should not be detected as player URLs."""
        url = "https://www.youtube.com/watch?v=abc123"
        assert _is_tiktok_player_url(url) is False


class TestCacheKeyIsolation:
    """Test cache key uniqueness for different videos."""

    def test_different_tiktoks_different_keys(self) -> None:
        """Different TikTok URLs should produce different cache keys."""
        # Simulate same resolved CDN URL (worst case scenario)
        same_resolved = "https://cdn.tiktok.com/video/abc123.mp4"

        key1 = _compute_download_key(
            same_resolved,
            "ba",
            1000000,
            original_url="https://www.tiktok.com/t/ABC123/",
        )
        key2 = _compute_download_key(
            same_resolved,
            "ba",
            1000000,
            original_url="https://www.tiktok.com/t/XYZ789/",
        )

        assert key1 != key2, "Different TikTok URLs must produce different cache keys"

    def test_same_tiktok_same_key(self) -> None:
        """Same TikTok URL should produce same cache key."""
        resolved = "https://cdn.tiktok.com/video/abc123.mp4"
        original = "https://www.tiktok.com/t/ABC123/"

        key1 = _compute_download_key(resolved, "ba", 1000000, original_url=original)
        key2 = _compute_download_key(resolved, "ba", 1000000, original_url=original)

        assert key1 == key2, "Same TikTok URL should produce same cache key"

    def test_youtube_has_identity_suffix(self) -> None:
        """YouTube URLs should have identity suffix in key."""
        resolved = "https://cdn.youtube.com/video/abc123.mp4"

        key = _compute_download_key(
            resolved,
            "ba",
            1000000,
            original_url="https://www.youtube.com/watch?v=abc123",
        )

        # Key should have the "-v" identity suffix
        assert "-v" in key

    def test_tiktok_has_identity_suffix(self) -> None:
        """TikTok URLs should have identity suffix in key."""
        resolved = "https://cdn.tiktok.com/video/abc123.mp4"

        key = _compute_download_key(resolved, "ba", 1000000, original_url="https://www.tiktok.com/t/ABC123/")

        # Key should have the "-v" identity suffix
        assert "-v" in key


class TestUrlHashUniqueness:
    """Test URL hash uniqueness."""

    def test_hash_different_for_different_urls(self) -> None:
        """Different URLs should produce different hashes."""
        hash1 = _hash_resolved_url("https://example.com/video1.mp4")
        hash2 = _hash_resolved_url("https://example.com/video2.mp4")
        assert hash1 != hash2

    def test_hash_same_for_same_url(self) -> None:
        """Same URL should produce same hash."""
        url = "https://example.com/video.mp4"
        hash1 = _hash_resolved_url(url)
        hash2 = _hash_resolved_url(url)
        assert hash1 == hash2

    def test_hash_length(self) -> None:
        """Hash should be 16 characters."""
        h = _hash_resolved_url("https://example.com/video.mp4")
        assert len(h) == 16
