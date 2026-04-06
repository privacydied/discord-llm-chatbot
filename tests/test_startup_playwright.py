"""Tests for startup.py Playwright browser check with remote server awareness."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCheckPlaywrightBrowsersRemoteSkip:
    """When PW_SERVER_URL is set, check_playwright_browsers skips local checks."""

    def test_skips_local_check_when_remote_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "http://localhost:3006")
        from bot.core.startup import check_playwright_browsers

        mock_logger = MagicMock()
        check_playwright_browsers(mock_logger)

        # Should log about remote and return without trying local checks
        assert any(
            "remote server configured" in str(call).lower()
            for call in mock_logger.info.call_args_list
        )
        mock_logger.warning.assert_not_called()

    def test_checks_local_when_no_remote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PW_SERVER_URL", raising=False)
        from bot.core.startup import check_playwright_browsers

        mock_logger = MagicMock()

        # Mock the internal check so we don't actually invoke subprocess
        with patch("bot.core.startup._get_playwright_chromium_path", return_value="/fake/chrome"):
            check_playwright_browsers(mock_logger)

        assert mock_logger.info.called
        # Should mention checking for browser binaries
        assert any(
            "browser" in str(call).lower() and "checking" in str(call).lower()
            for call in mock_logger.info.call_args_list
        )
