"""Tests for bot/utils/url_dedup.py — URL deduplication utility.

Phase 17: HTTP/search/X reductions — prevent duplicate URL processing.
"""

from bot.utils.url_dedup import deduplicate_urls


class TestDeduplicateUrls:
    def test_empty_list(self):
        assert deduplicate_urls([]) == []

    def test_single_url(self):
        urls = ["https://example.com"]
        assert deduplicate_urls(urls) == ["https://example.com"]

    def test_exact_duplicates(self):
        urls = ["https://example.com", "https://example.com", "https://other.com"]
        result = deduplicate_urls(urls)
        assert len(result) == 2
        assert "https://example.com" in result
        assert "https://other.com" in result

    def test_case_normalization(self):
        urls = ["https://EXAMPLE.COM/path", "https://example.com/path"]
        result = deduplicate_urls(urls)
        # Both normalize to same logical URL, keep first occurrence
        assert len(result) == 1

    def test_tracking_params_stripped(self):
        urls = [
            "https://example.com/path?utm_source=t&ref=abc",
            "https://example.com/path",
        ]
        result = deduplicate_urls(urls)
        assert len(result) == 1

    def test_different_paths_preserved(self):
        urls = [
            "https://example.com/a",
            "https://example.com/b",
        ]
        result = deduplicate_urls(urls)
        assert len(result) == 2

    def test_preserve_order(self):
        urls = [
            "https://z.com",
            "https://a.com",
            "https://b.com",
        ]
        result = deduplicate_urls(urls)
        assert result == urls

    def test_mixed_protocols(self):
        urls = ["http://example.com", "https://example.com"]
        # Different protocols = different safe keys after normalization
        # The dedup normalizes to lower but keeps protocol
        result = deduplicate_urls(urls)
        # Both should exist since http !== https after normalization
        assert len(result) == 2
