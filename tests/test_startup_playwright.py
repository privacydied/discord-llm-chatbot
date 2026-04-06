"""Tests for startup.py Playwright browser check with remote server awareness."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from bot.config import ConfigurationError


class TestCheckPlaywrightBrowsersRemoteValidation:
    """When PW_SERVER_URL is set, validate the remote server is reachable."""

    def test_validates_and_succeeds_when_reachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "http://localhost:3006")
        from bot.core.startup import check_playwright_browsers

        mock_logger = MagicMock()

        with patch("bot.core.startup.socket.create_connection", return_value=MagicMock()):
            check_playwright_browsers(mock_logger)

        mock_logger.warning.assert_not_called()
        assert any("reachable" in str(call).lower() for call in mock_logger.info.call_args_list)

    def test_raises_when_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "http://localhost:3006")
        from bot.core.startup import check_playwright_browsers

        mock_logger = MagicMock()

        with patch("bot.core.startup.socket.create_connection", side_effect=ConnectionRefusedError("refused")):
            with pytest.raises(ConfigurationError, match="unreachable"):
                check_playwright_browsers(mock_logger)

    def test_checks_local_when_no_remote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PW_SERVER_URL", raising=False)
        from bot.core.startup import check_playwright_browsers

        mock_logger = MagicMock()

        with patch("bot.core.startup._get_playwright_chromium_path", return_value="/fake/chrome"):
            check_playwright_browsers(mock_logger)

        assert mock_logger.info.called
        assert any(
            "browser" in str(call).lower()
            and "checking" in str(call).lower()
            for call in mock_logger.info.call_args_list
        )
