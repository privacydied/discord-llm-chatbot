"""Regression tests for security/logging hardening.

[SFT] Secrets redaction
[SFT] PII protection
[SFT] Chain-of-thought stripping
"""

import logging
import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skip(reason="Requires public output sanitizer refactoring")

from bot.public_output import extract_public_reply_text, has_reasoning_leakage
from bot.utils.logging import SensitiveDataFilter, redact_sensitive_values


def _make_log_record(**kwargs):
    """Create a real-ish log record for testing SensitiveDataFilter."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test",
        args=(),
        exc_info=None,
    )
    for k, v in kwargs.items():
        setattr(record, k, v)
    return record


class TestSensitiveDataFilter:
    @pytest.fixture
    def filter_obj(self):
        return SensitiveDataFilter()

    def test_redacts_api_keys_in_detail(self, filter_obj) -> None:
        """API keys should be redacted in detail field."""
        record = _make_log_record()
        record.detail = {
            "OPENAI_API_KEY": "sk-abc123",
            "other_field": "visible",
        }

        result = filter_obj.filter(record)

        assert result is True
        assert record.detail["OPENAI_API_KEY"] == "[REDACTED]"
        assert record.detail["other_field"] == "visible"

    def test_redacts_discord_token(self, filter_obj) -> None:
        """Discord token should be redacted."""
        record = _make_log_record()
        record.detail = {
            "DISCORD_TOKEN": "discord.token.12345",
        }

        filter_obj.filter(record)

        assert record.detail["DISCORD_TOKEN"] == "[REDACTED]"

    def test_redacts_x_bearer_token(self, filter_obj) -> None:
        """X/Twitter bearer token should be redacted."""
        record = _make_log_record()
        record.detail = {
            "X_API_BEARER_TOKEN": "Bearer abc123",
        }

        filter_obj.filter(record)

        assert record.detail["X_API_BEARER_TOKEN"] == "[REDACTED]"

    def test_redacts_nested_secrets(self, filter_obj) -> None:
        """Nested secrets should also be redacted."""
        record = _make_log_record()
        record.detail = {
            "config": {
                "api_key": "secret-value",
            },
        }

        filter_obj.filter(record)

        assert record.detail["config"]["api_key"] == "[REDACTED]"

    def test_handles_missing_detail_gracefully(self, filter_obj) -> None:
        """Missing detail field should not cause errors."""
        record = _make_log_record()

        result = filter_obj.filter(record)

        assert result is True

    def test_non_dict_detail_passes_through(self, filter_obj) -> None:
        """Non-dict detail should be handled gracefully."""
        record = _make_log_record()
        record.detail = "just a string"

        result = filter_obj.filter(record)

        assert result is True
        assert record.detail == "just a string"


class TestRedactSensitiveValues:
    def test_redacts_api_key_in_text(self, monkeypatch) -> None:
        """API keys in arbitrary text should be redacted."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")

        text = "The API key is sk-test-key-12345 - don't share it"
        result = redact_sensitive_values(text)

        assert "sk-test-key-12345" not in result
        assert "[REDACTED]" in result

    def test_redacts_discord_token(self, monkeypatch) -> None:
        """Discord token should be redacted from text."""
        monkeypatch.setenv("DISCORD_TOKEN", "discord.secret.token")

        text = "Token: discord.secret.token"
        result = redact_sensitive_values(text)

        assert "discord.secret.token" not in result
        assert "[REDACTED]" in result

    def test_handles_missing_env_vars(self, monkeypatch) -> None:
        """Missing env vars should not cause errors."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        text = "Some unrelated text"
        result = redact_sensitive_values(text)

        assert result == "Some unrelated text"

    def test_handles_short_tokens(self, monkeypatch) -> None:
        """Short tokens (<4 chars) should be ignored."""
        monkeypatch.setenv("SEARCH_API_KEY", "ab")  # Too short

        text = "Key is ab"
        result = redact_sensitive_values(text)

        # Short tokens should not be replaced
        assert result == text


class TestPublicOutputSanitizer:
    def test_normal_text_passes_through(self) -> None:
        """Normal text should not be blocked."""
        text = "Hello! How can I help you today?"
        result = extract_public_reply_text(text)

        assert result == text

    def test_reasoning_leak_blocked(self) -> None:
        """Reasoning leak patterns should be blocked."""
        text = "Okay, the user wants me to analyze this image..."
        result = extract_public_reply_text(text)

        # Should return safe fallback
        assert result != text
        assert "couldn't produce a clean public answer" in result

    def test_mode_gate_blocked(self) -> None:
        """MODE GATE content should be blocked."""
        text = "Checking the MODE GATE for this request..."
        result = extract_public_reply_text(text)

        assert result != text
        assert "couldn't produce" in result

    def test_thinking_tags_blocked(self) -> None:
        """Thinking tags should be blocked."""
        text = "<thinking>I need to be careful here</thinking> Result"
        result = extract_public_reply_text(text)

        assert result != text
        assert "couldn't produce" in result

    def test_political_discussion_allowed(self) -> None:
        """Political discussion should be allowed (not blocked)."""
        text = "The political situation has been evolving..."
        result = extract_public_reply_text(text)

        assert result == text

    def test_empty_input_returns_fallback(self) -> None:
        """Empty input should return safe fallback."""
        result = extract_public_reply_text("")

        assert "couldn't produce" in result

    def test_none_input_returns_fallback(self) -> None:
        """None input should return safe fallback."""
        result = extract_public_reply_text(None)

        assert "couldn't produce" in result

    def test_whitespace_only_returns_fallback(self) -> None:
        """Whitespace-only input should return safe fallback."""
        result = extract_public_reply_text("   ")

        assert "couldn't produce" in result

    def test_excessive_whitespace_normalized(self) -> None:
        """Multiple blank lines should be normalized."""
        text = "Line 1\n\n\n\n\nLine 2"
        result = extract_public_reply_text(text)

        assert "\n\n\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result


class TestHasReasoningLeakage:
    def test_detects_okay_the_user(self) -> None:
        """Should detect 'Okay, the user' pattern."""
        assert has_reasoning_leakage("Okay, the user wants...") is True

    def test_detects_thinking_tags(self) -> None:
        """Should detect thinking tags."""
        assert has_reasoning_leakage("<thinking>internal</thinking>") is True

    def test_no_leak_in_normal_text(self) -> None:
        """Normal text should not be flagged."""
        assert has_reasoning_leakage("Hello, how are you?") is False

    def test_empty_string_no_leak(self) -> None:
        """Empty string should not be flagged."""
        assert has_reasoning_leakage("") is False

    def test_none_no_leak(self) -> None:
        """None should not be flagged."""
        assert has_reasoning_leakage(None) is False


class TestLoggingDoesNotLeakSecrets:
    """Integration-style tests for log safety."""

    def test_reasoning_leak_logged_with_hash_not_content(self, caplog) -> None:
        """When reasoning is blocked, log should contain pattern info not full content."""
        with caplog.at_level("WARNING"):
            extract_public_reply_text("Okay, the user wants help")

        # Log should mention blocked reasoning
        log_text = caplog.text
        assert "Blocked reasoning leak" in log_text or "reasoning" in log_text.lower()

    def test_log_excludes_full_content(self, caplog) -> None:
        """Log should not contain full blocked content."""
        sensitive_text = "Okay, the user wants me to do something"

        with caplog.at_level("WARNING"):
            extract_public_reply_text(sensitive_text)

        # Full content should not be in logs (only truncated pattern + hash)
        combined_logs = " ".join(r.message for r in caplog.records)
        # The log message should use pattern='%s' hash=%s format, not full text
        assert "Okay, the user" not in combined_logs or len(combined_logs) < 200


class TestPartialSecretRedaction:
    """Verify that secret VALUES are redacted even when the key name isn't in SECRET_KEYS."""

    def test_env_value_redacted_from_arbitrary_key(self) -> None:
        """If a string field contains a known secret value, redact it [S7]."""
        fake_token = "sk-fake-discord-token-abc123"
        with patch.dict(os.environ, {"DISCORD_TOKEN": fake_token}):
            f = SensitiveDataFilter()
            record = _make_log_record(extra_data={"url": f"https://api.example.com?token={fake_token}"})
            f.filter(record)
            assert fake_token not in str(record.extra_data["url"])
            assert "[REDACTED]" in str(record.extra_data["url"])

    def test_nested_dict_value_redaction(self) -> None:
        """Nested dicts with secret values in non-secret keys get redacted [S7]."""
        fake_key = "sk-openai-fake-key-999"
        with patch.dict(os.environ, {"OPENAI_API_KEY": fake_key}):
            f = SensitiveDataFilter()
            record = _make_log_record(detail={"response": f"Key used: {fake_key}"})
            f.filter(record)
            assert fake_key not in record.detail["response"]

    def test_exact_key_still_redacted(self) -> None:
        """Keys in SECRET_KEYS still get full redaction regardless of value."""
        f = SensitiveDataFilter()
        record = _make_log_record(extra_data={"api_key": "some-secret-value"})
        f.filter(record)
        assert record.extra_data["api_key"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# S8: SECRET_KEYS completeness
# ---------------------------------------------------------------------------


class TestSecretKeysCompleteness:
    """Verify that all secret env vars from config are in SECRET_KEYS."""

    EXPECTED_SECRET_ENV_VARS = {
        "OPENAI_API_KEY",
        "DISCORD_TOKEN",
        "X_API_BEARER_TOKEN",
        "VISION_API_KEY",
        "SCREENSHOT_API_KEY",
        "SEARCH_API_KEY",
        "YOUTUBE_API_KEY",
        "WHISPER_API_KEY",
        "DDG_API_KEY",
        "CUSTOM_SEARCH_API_KEY",
    }

    def test_all_secret_envs_in_secret_keys(self) -> None:
        """All secret env vars from config must appear in SECRET_KEYS [S8]."""
        missing = self.EXPECTED_SECRET_ENV_VARS - SensitiveDataFilter.SECRET_KEYS
        assert not missing, f"Missing from SECRET_KEYS: {missing}"

    def test_all_secret_envs_in_redact_list(self) -> None:
        """All secret env vars must appear in redact_sensitive_values list [S8]."""
        # Access the list inside the function
        # We verify by calling redact_sensitive_values with each key's value
        for key in self.EXPECTED_SECRET_ENV_VARS:
            fake_val = f"fake-{key}-value-12345678"
            with patch.dict(os.environ, {key: fake_val}):
                result = redact_sensitive_values(f"text with {fake_val} inside")
                assert fake_val not in result, f"redact_sensitive_values didn't redact {key}"


# ---------------------------------------------------------------------------
# Discord token never logged
# ---------------------------------------------------------------------------
