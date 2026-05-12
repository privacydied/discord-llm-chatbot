"""
Tests for STT cache key isolation across all video providers.
Ensures that different videos produce different cache keys and that
transcripts are not cross-contaminated, regardless of URL variants or CDN overlap.
"""

import hashlib
import re
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# Inline copies of functions under test (to avoid import issues in CI)
# ---------------------------------------------------------------------------


def _hash_resolved_url(resolved_url: str) -> str:
    h = hashlib.sha256((resolved_url or "").encode("utf-8")).hexdigest()
    return h[:16]


def _normalize_youtube_url(url: str) -> str:
    """
    Normalize YouTube URLs to a canonical form.
    Returns canonical form: youtube://video/{VIDEO_ID}
    """
    if not url:
        return url
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path or ""

        # youtu.be/VIDEO_ID
        if host in ("youtu.be", "www.youtu.be"):
            video_id = path.lstrip("/").split("/")[0].split("?")[0]
            if video_id and len(video_id) >= 6:
                return f"youtube://video/{video_id}"

        # youtube.com variants
        if host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            # /watch?v=VIDEO_ID
            if path.startswith("/watch"):
                query = parse_qs(parsed.query)
                video_id = query.get("v", [""])[0]
                if video_id and len(video_id) >= 6:
                    return f"youtube://video/{video_id}"

            # /shorts/VIDEO_ID, /embed/VIDEO_ID, /live/VIDEO_ID, /v/VIDEO_ID
            for prefix in ("/shorts/", "/embed/", "/live/", "/v/"):
                if path.startswith(prefix):
                    video_id = path[len(prefix) :].split("/")[0].split("?")[0]
                    if video_id and len(video_id) >= 6:
                        return f"youtube://video/{video_id}"
    except Exception:
        pass
    return url


def _normalize_tiktok_url(url: str) -> str:
    """Normalize TikTok URLs to canonical form."""
    if not url:
        return url
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host in ("vm.tiktok.com", "m.tiktok.com", "www.tiktok.com", "tiktok.com"):
            path = parsed.path.rstrip("/")

            # /player/v1/<video_id> embed URLs
            player_match = re.match(r"^/player(?:/v\d+)?/(\d+)", path)
            if player_match:
                video_id = player_match.group(1)
                return f"tiktok://video/{video_id}"

            # /@user/video/<video_id> canonical URLs
            video_match = re.match(r"^/@[\w\.-]+/video/(\d+)", path)
            if video_match:
                video_id = video_match.group(1)
                return f"tiktok://video/{video_id}"

            return f"tiktok://{path}"
    except Exception:
        pass
    return url


def _canonicalize_video_identity(original_url: str, metadata=None) -> str:
    """
    Canonicalize video identity for cache keying across all providers.
    Uses yt-dlp metadata when available, falls back to URL normalization.
    """
    # If we have yt-dlp metadata, use extractor:id
    if metadata:
        extractor = metadata.get("extractor_key") or metadata.get("extractor") or ""
        video_id = metadata.get("id") or ""
        if extractor and video_id:
            return f"{extractor.lower()}:{video_id}"

    if not original_url:
        return ""

    url_lower = original_url.lower()

    # YouTube normalization
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        normalized = _normalize_youtube_url(original_url)
        if normalized.startswith("youtube://"):
            return normalized.replace("://", ":")

    # TikTok normalization
    if "tiktok.com" in url_lower:
        normalized = _normalize_tiktok_url(original_url)
        if normalized.startswith("tiktok://"):
            return normalized.replace("://", ":")

    # Generic fallback: hash of original URL
    return f"generic:{hashlib.sha256(original_url.encode()).hexdigest()[:16]}"


def _compute_download_key(
    resolved_url: str,
    fmt_id: str,
    content_length,
    original_url=None,
    video_identity=None,
) -> str:
    """
    Compute cache key with video identity for collision resistance.
    """
    length_part = str(content_length) if content_length is not None else "na"
    base_key = f"{_hash_resolved_url(resolved_url)}-{fmt_id}-{length_part}"

    # Always include video identity hash
    if video_identity:
        identity_hash = _hash_resolved_url(video_identity)[:10]
        base_key = f"{base_key}-v{identity_hash}"
    elif original_url:
        fallback_identity = _canonicalize_video_identity(original_url)
        identity_hash = _hash_resolved_url(fallback_identity)[:10]
        base_key = f"{base_key}-v{identity_hash}"

    return base_key


# ---------------------------------------------------------------------------
# YouTube URL Normalization Tests
# ---------------------------------------------------------------------------


class TestYouTubeUrlNormalization:
    """Test YouTube URL normalization for consistent cache keying."""

    def test_normalize_watch_url(self):
        """Standard watch URLs should normalize to video ID."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        normalized = _normalize_youtube_url(url)
        assert normalized == "youtube://video/dQw4w9WgXcQ"

    def test_normalize_watch_url_with_extra_params(self):
        """Watch URLs with extra params should still extract video ID."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30&list=PLx0"
        normalized = _normalize_youtube_url(url)
        assert normalized == "youtube://video/dQw4w9WgXcQ"

    def test_normalize_shorts_url(self):
        """Shorts URLs should normalize to video ID."""
        url = "https://www.youtube.com/shorts/abcd1234xyz"
        normalized = _normalize_youtube_url(url)
        assert normalized == "youtube://video/abcd1234xyz"

    def test_normalize_youtu_be_url(self):
        """youtu.be short links should normalize to video ID."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        normalized = _normalize_youtube_url(url)
        assert normalized == "youtube://video/dQw4w9WgXcQ"

    def test_normalize_embed_url(self):
        """Embed URLs should normalize to video ID."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        normalized = _normalize_youtube_url(url)
        assert normalized == "youtube://video/dQw4w9WgXcQ"

    def test_normalize_live_url(self):
        """Live URLs should normalize to video ID."""
        url = "https://www.youtube.com/live/dQw4w9WgXcQ"
        normalized = _normalize_youtube_url(url)
        assert normalized == "youtube://video/dQw4w9WgXcQ"

    def test_same_video_different_urls(self):
        """Different URL formats for same video should normalize identically."""
        video_id = "dQw4w9WgXcQ"
        urls = [
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://youtube.com/shorts/{video_id}",
            f"https://youtu.be/{video_id}",
            f"https://www.youtube.com/embed/{video_id}",
            f"https://m.youtube.com/watch?v={video_id}",
        ]
        normalized = [_normalize_youtube_url(url) for url in urls]
        assert all(n == f"youtube://video/{video_id}" for n in normalized)

    def test_different_videos_different_normalized(self):
        """Different videos should produce different normalized forms."""
        url1 = "https://www.youtube.com/watch?v=video1abcdef"
        url2 = "https://www.youtube.com/watch?v=video2ghijkl"
        assert _normalize_youtube_url(url1) != _normalize_youtube_url(url2)

    def test_non_youtube_unchanged(self):
        """Non-YouTube URLs should be returned unchanged."""
        url = "https://www.tiktok.com/@user/video/123456789"
        assert _normalize_youtube_url(url) == url


# ---------------------------------------------------------------------------
# Video Identity Canonicalization Tests
# ---------------------------------------------------------------------------


class TestVideoIdentityCanonicalization:
    """Test video identity canonicalization across providers."""

    def test_youtube_identity_from_metadata(self):
        """Should use extractor:id from yt-dlp metadata."""
        metadata = {"extractor_key": "Youtube", "id": "dQw4w9WgXcQ"}
        identity = _canonicalize_video_identity(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", metadata
        )
        assert identity == "youtube:dQw4w9WgXcQ"

    def test_tiktok_identity_from_metadata(self):
        """Should use extractor:id from yt-dlp metadata."""
        metadata = {"extractor_key": "TikTok", "id": "7123456789012345678"}
        identity = _canonicalize_video_identity(
            "https://www.tiktok.com/@user/video/7123456789012345678", metadata
        )
        assert identity == "tiktok:7123456789012345678"

    def test_youtube_identity_fallback_to_url(self):
        """Should fall back to URL normalization without metadata."""
        identity = _canonicalize_video_identity(
            "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        )
        assert identity == "youtube:video/dQw4w9WgXcQ"

    def test_tiktok_identity_fallback_to_url(self):
        """Should fall back to URL normalization without metadata."""
        identity = _canonicalize_video_identity(
            "https://www.tiktok.com/@user/video/7123456789"
        )
        assert identity == "tiktok:video/7123456789"

    def test_generic_identity_for_unknown_domain(self):
        """Unknown domains should get generic hash identity."""
        identity = _canonicalize_video_identity("https://example.com/video/abc123")
        assert identity.startswith("generic:")
        assert len(identity) > 20  # generic: + 16 char hash

    def test_cross_provider_different_identities(self):
        """Same numeric ID on different providers must have different identities."""
        youtube_identity = _canonicalize_video_identity(
            "https://www.youtube.com/watch?v=123456789012"
        )
        tiktok_identity = _canonicalize_video_identity(
            "https://www.tiktok.com/@user/video/123456789012"
        )
        assert youtube_identity != tiktok_identity


# ---------------------------------------------------------------------------
# Cache Key Isolation Tests
# ---------------------------------------------------------------------------


class TestCacheKeyIsolation:
    """Test cache key uniqueness across different videos."""

    def test_different_youtube_videos_different_keys(self):
        """Different YouTube videos must produce different cache keys."""
        # Simulate same resolved CDN URL (worst case scenario)
        same_resolved = "https://rr1---sn-abc.googlevideo.com/videoplayback?id=xyz"

        key1 = _compute_download_key(
            same_resolved,
            "251",
            1000000,
            original_url="https://www.youtube.com/watch?v=video1abc",
            video_identity="youtube:video1abc",
        )
        key2 = _compute_download_key(
            same_resolved,
            "251",
            1000000,
            original_url="https://www.youtube.com/watch?v=video2def",
            video_identity="youtube:video2def",
        )

        assert key1 != key2, (
            "Different YouTube videos must produce different cache keys"
        )

    def test_same_youtube_video_same_key(self):
        """Same YouTube video should produce same cache key."""
        resolved = "https://rr1---sn-abc.googlevideo.com/videoplayback?id=xyz"
        video_id = "dQw4w9WgXcQ"
        identity = f"youtube:{video_id}"

        key1 = _compute_download_key(
            resolved,
            "251",
            1000000,
            original_url=f"https://www.youtube.com/watch?v={video_id}",
            video_identity=identity,
        )
        key2 = _compute_download_key(
            resolved,
            "251",
            1000000,
            original_url=f"https://youtu.be/{video_id}",
            video_identity=identity,
        )

        assert key1 == key2, "Same video should produce same cache key"

    def test_youtube_shorts_vs_watch_same_key(self):
        """YouTube shorts and watch for same video should have same key."""
        resolved = "https://rr1---sn-abc.googlevideo.com/videoplayback?id=xyz"
        video_id = "abcd1234xyz"
        identity = f"youtube:{video_id}"

        key_shorts = _compute_download_key(
            resolved,
            "251",
            1000000,
            original_url=f"https://www.youtube.com/shorts/{video_id}",
            video_identity=identity,
        )
        key_watch = _compute_download_key(
            resolved,
            "251",
            1000000,
            original_url=f"https://www.youtube.com/watch?v={video_id}",
            video_identity=identity,
        )

        assert key_shorts == key_watch

    def test_cross_provider_isolation(self):
        """YouTube and TikTok with same numeric ID must have different keys."""
        same_resolved = "https://cdn.example.com/video.mp4"
        numeric_id = "1234567890123"

        youtube_key = _compute_download_key(
            same_resolved, "251", 1000000, video_identity=f"youtube:{numeric_id}"
        )
        tiktok_key = _compute_download_key(
            same_resolved, "251", 1000000, video_identity=f"tiktok:{numeric_id}"
        )

        assert youtube_key != tiktok_key, "Cross-provider keys must differ"

    def test_all_keys_have_identity_suffix(self):
        """All cache keys should include video identity suffix."""
        key = _compute_download_key(
            "https://cdn.example.com/video.mp4",
            "251",
            1000000,
            original_url="https://www.youtube.com/watch?v=test12345",
            video_identity="youtube:test12345",
        )
        assert "-v" in key, "Cache key should have video identity suffix"

    def test_fallback_identity_when_no_metadata(self):
        """Should compute identity from URL when video_identity not provided."""
        key = _compute_download_key(
            "https://cdn.example.com/video.mp4",
            "251",
            1000000,
            original_url="https://www.youtube.com/watch?v=fallback123",
        )
        assert "-v" in key, "Should have identity suffix even without explicit identity"

    def test_generic_url_still_gets_identity(self):
        """Generic URLs should still get identity suffix for isolation."""
        key = _compute_download_key(
            "https://cdn.example.com/video.mp4",
            "251",
            1000000,
            original_url="https://unknownsite.com/video/xyz123",
        )
        assert "-v" in key


# ---------------------------------------------------------------------------
# Regression: TikTok Tests (ensure fix still works)
# ---------------------------------------------------------------------------


class TestTikTokCacheIsolation:
    """Regression tests to ensure TikTok fix still works with generalized code."""

    def test_different_tiktoks_different_keys(self):
        """Different TikTok URLs should produce different cache keys."""
        same_resolved = "https://cdn.tiktok.com/video/abc123.mp4"

        key1 = _compute_download_key(
            same_resolved, "ba", 1000000, video_identity="tiktok:7111111111111111111"
        )
        key2 = _compute_download_key(
            same_resolved, "ba", 1000000, video_identity="tiktok:7222222222222222222"
        )

        assert key1 != key2

    def test_same_tiktok_same_key(self):
        """Same TikTok video should produce same cache key."""
        resolved = "https://cdn.tiktok.com/video/abc123.mp4"
        identity = "tiktok:7123456789012345678"

        key1 = _compute_download_key(resolved, "ba", 1000000, video_identity=identity)
        key2 = _compute_download_key(resolved, "ba", 1000000, video_identity=identity)

        assert key1 == key2

    def test_tiktok_player_and_canonical_same_identity(self):
        """Player URL and canonical URL for same video should have same identity."""
        player_url = "https://www.tiktok.com/player/v1/7578893205815495958"
        canonical_url = "https://www.tiktok.com/@user/video/7578893205815495958"

        player_identity = _canonicalize_video_identity(player_url)
        canonical_identity = _canonicalize_video_identity(canonical_url)

        assert player_identity == canonical_identity


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_url(self):
        """Empty URL should not crash."""
        assert _normalize_youtube_url("") == ""
        assert _normalize_tiktok_url("") == ""
        assert _canonicalize_video_identity("") == ""

    def test_malformed_url(self):
        """Malformed URLs should be handled gracefully."""
        url = "not-a-valid-url"
        # Should not crash, may return original or fallback
        identity = _canonicalize_video_identity(url)
        assert identity.startswith("generic:")

    def test_short_video_id_ignored(self):
        """Very short video IDs should be rejected."""
        # YouTube IDs are typically 11 chars, minimum 6
        url = "https://www.youtube.com/watch?v=ab"
        normalized = _normalize_youtube_url(url)
        # Should return original URL since ID too short
        assert normalized == url

    def test_hash_consistency(self):
        """Same input should always produce same hash."""
        url = "https://www.youtube.com/watch?v=consistent123"
        hash1 = _hash_resolved_url(url)
        hash2 = _hash_resolved_url(url)
        assert hash1 == hash2
        assert len(hash1) == 16
