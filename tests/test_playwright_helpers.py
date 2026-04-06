"""Tests for Playwright remote-server routing via playwright_helpers.py

All subprocess-based config tests (no import-reload flakiness).
All unit tests with proper mock objects.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Config-level tests (subprocess-isolated to avoid env import caching) ──


class TestPlaywrightHelpersConfig:
    """Verify PW_SERVER_URL is read from environment at import time."""

    def _run_isolated(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

    def test_pw_server_url_set(self) -> None:
        r = self._run_isolated(
            "import os; os.environ['PW_SERVER_URL']='http://localhost:3006'; "
            "from bot.utils.playwright_helpers import PW_SERVER_URL; "
            "assert PW_SERVER_URL == 'http://localhost:3006', f'got {PW_SERVER_URL}'"
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_pw_server_url_empty_defaults_to_none(self) -> None:
        r = self._run_isolated(
            "import os; os.environ['PW_SERVER_URL']='  '; "
            "from bot.utils.playwright_helpers import PW_SERVER_URL; "
            "assert PW_SERVER_URL is None, f'got {PW_SERVER_URL}'"
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_is_remote_flag_true_when_set(self) -> None:
        r = self._run_isolated(
            "import os; os.environ['PW_SERVER_URL']='http://localhost:3006'; "
            "from bot.utils.playwright_helpers import is_remote_playwright_configured; "
            "assert is_remote_playwright_configured() is True"
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_is_remote_flag_false_when_unset(self) -> None:
        r = self._run_isolated(
            "import os; os.environ.pop('PW_SERVER_URL', None); "
            "from bot.utils.playwright_helpers import is_remote_playwright_configured; "
            "assert is_remote_playwright_configured() is False"
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"


# ── connect_browser unit tests ─────────────────────────────────────────


@pytest.mark.asyncio
class TestPlaywrightHelpersConnectBrowser:
    """Verify connect_browser tries remote first, then local fallback."""

    async def test_connect_browser_uses_remote_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "http://localhost:9999")

        import bot.utils.playwright_helpers as pwh
        import importlib
        importlib.reload(pwh)

        mock_browser = MagicMock()
        mock_browser._is_remote = False

        chromium_mock = MagicMock()
        chromium_mock.connect_over_cdp = AsyncMock(return_value=mock_browser)
        chromium_mock.launch = AsyncMock()
        fake_pw = MagicMock()
        fake_pw.chromium = chromium_mock

        result = await pwh.connect_browser(fake_pw)

        assert result is mock_browser
        chromium_mock.connect_over_cdp.assert_called_once_with(
            "http://localhost:9999", timeout=30_000
        )
        chromium_mock.launch.assert_not_called()
        assert getattr(result, "_is_remote", False) is True

    async def test_connect_browser_falls_back_to_local_when_remote_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "http://bad-host:9999")

        import bot.utils.playwright_helpers as pwh
        import importlib
        importlib.reload(pwh)

        local_browser = MagicMock()
        local_browser._is_remote = False

        chromium_mock = MagicMock()
        chromium_mock.connect_over_cdp = AsyncMock(side_effect=ConnectionError("refused"))
        chromium_mock.launch = AsyncMock(return_value=local_browser)
        fake_pw = MagicMock()
        fake_pw.chromium = chromium_mock

        result = await pwh.connect_browser(fake_pw)

        assert result is local_browser
        chromium_mock.connect_over_cdp.assert_called_once()
        chromium_mock.launch.assert_called_once()
        assert getattr(result, "_is_remote", False) is False

    async def test_connect_browser_uses_local_when_no_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PW_SERVER_URL", raising=False)

        import bot.utils.playwright_helpers as pwh
        import importlib
        importlib.reload(pwh)

        local_browser = MagicMock()
        local_browser._is_remote = False

        chromium_mock = MagicMock()
        chromium_mock.connect_over_cdp = AsyncMock()
        chromium_mock.launch = AsyncMock(return_value=local_browser)
        fake_pw = MagicMock()
        fake_pw.chromium = chromium_mock

        result = await pwh.connect_browser(fake_pw)

        assert result is local_browser
        chromium_mock.connect_over_cdp.assert_not_called()
        chromium_mock.launch.assert_called_once()
        assert getattr(result, "_is_remote", False) is False

    async def test_connect_browser_returns_none_when_both_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PW_SERVER_URL", "http://bad-host:9999")

        import bot.utils.playwright_helpers as pwh
        import importlib
        importlib.reload(pwh)

        chromium_mock = MagicMock()
        chromium_mock.connect_over_cdp = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        chromium_mock.launch = AsyncMock(side_effect=RuntimeError("cannot launch"))
        fake_pw = MagicMock()
        fake_pw.chromium = chromium_mock

        result = await pwh.connect_browser(fake_pw)

        assert result is None
