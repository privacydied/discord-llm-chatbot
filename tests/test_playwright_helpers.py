"""Playwright remote-server routing via playwright_helpers.py"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright._impl._errors import Error as PlaywrightError


class TestPlaywrightHelpersConfig:
    def _run_isolated(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    def test_http_url_is_normalised_to_ws(self) -> None:
        r = self._run_isolated("import os; os.environ['PW_SERVER_URL']='http://localhost:3006'; from bot.utils.playwright_helpers import _pw_server_url; assert _pw_server_url() == 'ws://localhost:3006'")
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_ws_url_is_preserved(self) -> None:
        r = self._run_isolated("import os; os.environ['PW_SERVER_URL']='ws://localhost:3006'; from bot.utils.playwright_helpers import _pw_server_url; assert _pw_server_url() == 'ws://localhost:3006'")
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_empty_defaults_to_none(self) -> None:
        r = self._run_isolated("import os; os.environ.pop('PW_SERVER_URL', None); from bot.utils.playwright_helpers import _pw_server_url; assert _pw_server_url() is None")
        assert r.returncode == 0, f"stderr: {r.stderr}"


@pytest.mark.asyncio
class TestPlaywrightHelpersConnectBrowser:
    async def test_connect_browser_uses_remote_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "ws://localhost:9999")
        import bot.utils.playwright_helpers as pwh
        import importlib

        importlib.reload(pwh)

        mock_browser = MagicMock()
        mock_chromium = MagicMock()
        mock_chromium.connect = AsyncMock(return_value=mock_browser)

        result = await pwh.connect_browser(mock_chromium)

        assert result is mock_browser
        mock_chromium.connect.assert_called_once_with("ws://localhost:9999", timeout=30_000)

    async def test_connect_browser_raises_on_remote_failure(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "ws://bad-host:9999")
        caplog.set_level("WARNING")
        import bot.utils.playwright_helpers as pwh
        import importlib

        importlib.reload(pwh)

        mock_chromium = MagicMock()
        mock_chromium.connect = AsyncMock(side_effect=PlaywrightError("refused"))

        # Production code catches connection errors, logs warning, and returns None
        result = await pwh.connect_browser(mock_chromium)

        assert result is None
        assert any("unreachable" in rec.message.lower() or "refused" in rec.message for rec in caplog.records)

    async def test_connect_browser_returns_none_when_no_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PW_SERVER_URL", raising=False)
        import bot.utils.playwright_helpers as pwh
        import importlib

        importlib.reload(pwh)

        mock_chromium = MagicMock()
        result = await pwh.connect_browser(mock_chromium)

        assert result is None
        mock_chromium.connect.assert_not_called()
