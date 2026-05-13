"""Tests for metrics label sanitization.

Phase 19: Metrics reductions — prevent high-cardinality label growth.
"""

import pytest
from bot.metrics.prometheus_metrics import sanitize_label_value, sanitize_labels


class TestSanitizeLabelValue:
    def test_short_string_passthrough(self):
        assert sanitize_label_value("ok") == "ok"
        assert sanitize_label_value("channel") == "channel"

    def test_url_sanitized(self):
        url = "https://example.com/very/long/path"
        result = sanitize_label_value(url)
        # URLs match the high-cardinality pattern and get replaced
        assert result == "__sanitized__"

    def test_ip_address_sanitized(self):
        result = sanitize_label_value("192.168.1.1")
        assert result == "__sanitized__"

    def test_snowflake_sanitized(self):
        # Discord snowflake: 17-19 digit number
        result = sanitize_label_value("123456789012345678")
        assert result == "__sanitized__"

    def test_long_value_truncated(self):
        long_val = "abcdefghij" * 20  # 200 chars
        result = sanitize_label_value(long_val)
        assert len(result) <= 128
        assert "..." in result

    def test_none_type_error(self):
        with pytest.raises(TypeError):
            sanitize_label_value(None)

    def test_underscores_and_spaces_preserved_if_no_pattern(self):
        result = sanitize_label_value("some-channel_name")
        assert result == "some-channel_name"


class TestSanitizeLabels:
    def test_none_returns_none(self):
        assert sanitize_labels(None) is None

    def test_empty_dict_returns_none(self):
        assert sanitize_labels({}) is None

    def test_sanitizes_all_values(self):
        labels = {
            "safe": "ok",
            "url": "https://example.com",
            "channel": "#general",
        }
        result = sanitize_labels(labels)
        assert result["safe"] == "ok"
        assert result["url"] == "__sanitized__"
        assert result["channel"] == "#general"

    def test_converts_non_string_values(self):
        labels = {"count": 42}
        result = sanitize_labels(labels)
        assert result["count"] == "42"
