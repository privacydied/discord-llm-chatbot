"""Tests for startup.py Playwright browser check with remote server awareness."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest



class TestCheckPlaywrightBrowsersRemoteValidation:
    """When PW_SERVER_URL is set, validate the remote server is reachable."""

    def test_validates_and_succeeds_when_reachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "http://localhost:3006")
        from bot.core.startup import check_playwright_browsers

        mock_logger = MagicMock()

        with patch(
            "bot.core.startup.socket.create_connection", return_value=MagicMock()
        ):
            check_playwright_browsers(mock_logger)

        mock_logger.warning.assert_not_called()
        assert any(
            "reachable" in str(call).lower() for call in mock_logger.info.call_args_list
        )

    def test_warns_when_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "http://localhost:3006")
        from bot.core.startup import check_playwright_browsers

        mock_logger = MagicMock()

        with patch(
            "bot.core.startup.socket.create_connection",
            side_effect=ConnectionRefusedError("refused"),
        ):
            check_playwright_browsers(mock_logger)

        # Production logs a warning but does NOT raise when unreachable
        mock_logger.warning.assert_called_once()
        assert "unreachable" in str(mock_logger.warning.call_args).lower()

    def test_checks_local_when_no_remote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PW_SERVER_URL", raising=False)
        from bot.core.startup import check_playwright_browsers

        mock_logger = MagicMock()

        with patch(
            "bot.core.startup._get_playwright_chromium_path",
            return_value="/fake/chrome",
        ):
            check_playwright_browsers(mock_logger)

        assert mock_logger.info.called
        assert any(
            "browser" in str(call).lower() and "checking" in str(call).lower()
            for call in mock_logger.info.call_args_list
        )
