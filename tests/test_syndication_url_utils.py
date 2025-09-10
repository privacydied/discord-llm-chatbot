"""
Unit tests for syndication URL utilities.
Tests for upgrade_pbs_to_orig function ensuring idempotent, safe URL upgrades.
"""

import pytest
from bot.syndication.url_utils import upgrade_pbs_to_orig


class TestUpgradePbsToOrig:
    """Test cases for upgrade_pbs_to_orig function."""

    def test_upgrade_pbs_basic(self):
        """upgrade_pbs_to_orig_basic: basic upgrade with format preservation."""
        input_url = "https://pbs.twimg.com/media/ABC123?format=jpg&name=small"
        expected = "https://pbs.twimg.com/media/ABC123?format=jpg&name=orig"
        result = upgrade_pbs_to_orig(input_url)
        assert result == expected

    def test_upgrade_pbs_preserve_format(self):
        """upgrade_pbs_preserve_format: preserve existing format param."""
        input_url = "https://pbs.twimg.com/media/ABC123?name=large&format=png"
        expected = "https://pbs.twimg.com/media/ABC123?name=orig&format=png"
        result = upgrade_pbs_to_orig(input_url)
        assert result == expected

    def test_upgrade_pbs_no_existing_format(self):
        """Don't add format if not present, only upgrade name param."""
        input_url = "https://pbs.twimg.com/media/ABC123?name=small"
        expected = "https://pbs.twimg.com/media/ABC123?name=orig"
        result = upgrade_pbs_to_orig(input_url)
        assert result == expected

    def test_upgrade_pbs_idempotent(self):
        """upgrade_pbs_idempotent: calling twice yields same result."""
        input_url = "https://pbs.twimg.com/media/ABC123?format=png&name=orig"
        expected = "https://pbs.twimg.com/media/ABC123?format=png&name=orig"
        result1 = upgrade_pbs_to_orig(input_url)
        result2 = upgrade_pbs_to_orig(result1)
        assert result1 == expected
        assert result2 == expected
        assert result1 == result2

    def test_upgrade_non_pbs_noop(self):
        """upgrade_non_pbs_noop: non-pbs URLs unchanged."""
        test_urls = [
            "https://example.com/img.jpg",
            "https://cdn.example.com/media/photo.png?size=large",
            "https://twitter.com/user/photo",
            "https://abs.twimg.com/media/photo.jpg",  # Different subdomain
        ]

        for url in test_urls:
            result = upgrade_pbs_to_orig(url)
            assert result == url, f"URL should be unchanged: {url}"

    def test_upgrade_pbs_no_query_params(self):
        """Handle pbs URLs with no existing query params."""
        input_url = "https://pbs.twimg.com/media/ABC123"
        expected = "https://pbs.twimg.com/media/ABC123?name=orig"
        result = upgrade_pbs_to_orig(input_url)
        assert result == expected

    def test_upgrade_pbs_complex_params(self):
        """Handle pbs URLs with multiple params, preserving all except name."""
        input_url = (
            "https://pbs.twimg.com/media/ABC123?format=webp&name=240x240&other=value"
        )
        expected = (
            "https://pbs.twimg.com/media/ABC123?format=webp&name=orig&other=value"
        )
        result = upgrade_pbs_to_orig(input_url)
        assert result == expected

    def test_upgrade_malformed_url_safety(self):
        """upgrade_pbs_* should never throw, even on malformed URLs."""
        malformed_urls = [
            "not-a-url",
            "://malformed",
            "",
            "None",  # String representation of None
        ]

        # URLs that look like pbs.twimg.com but are still valid enough to parse should get upgraded
        pbs_like_urls = [
            (
                "https://pbs.twimg.com/media/ABC[invalid",
                "https://pbs.twimg.com/media/ABC[invalid?name=orig",
            ),
        ]

        # Test truly malformed URLs that should return unchanged
        for url in malformed_urls:
            try:
                result = upgrade_pbs_to_orig(url)
                # Should return input unchanged for malformed URLs
                assert result == url
            except Exception as e:
                pytest.fail(f"upgrade_pbs_to_orig should never throw, got: {e}")

        # Test pbs.twimg.com URLs that are valid enough to upgrade
        for input_url, expected in pbs_like_urls:
            result = upgrade_pbs_to_orig(input_url)
            assert result == expected

    def test_upgrade_pbs_edge_cases(self):
        """Test edge cases like fragments, ports, etc."""
        test_cases = [
            # With fragment
            (
                "https://pbs.twimg.com/media/ABC123?name=small#section",
                "https://pbs.twimg.com/media/ABC123?name=orig#section",
            ),
            # With port
            (
                "https://pbs.twimg.com:443/media/ABC123?name=small",
                "https://pbs.twimg.com:443/media/ABC123?name=orig",
            ),
            # Different path structure
            (
                "https://pbs.twimg.com/card_img/ABC123?name=small&format=jpg",
                "https://pbs.twimg.com/card_img/ABC123?name=orig&format=jpg",
            ),
        ]

        for input_url, expected in test_cases:
            result = upgrade_pbs_to_orig(input_url)
            assert result == expected, f"Failed for: {input_url}"
