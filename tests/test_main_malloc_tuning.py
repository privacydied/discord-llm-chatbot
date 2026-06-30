"""Tests for bot.main._tune_glibc_malloc_arenas (memory audit fix).

Verifies the glibc arena cap is applied via a live mallopt() call (works
regardless of when threads/imports happen, unlike the MALLOC_ARENA_MAX env
var which only helps if set before the process starts), respects an env
override, and never raises on non-Linux or non-glibc platforms.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import bot.main as main_module


class TestTuneGlibcMallocArenas:
    def test_noop_on_non_linux(self, monkeypatch) -> None:
        monkeypatch.setattr(main_module.sys, "platform", "darwin")
        with patch("ctypes.CDLL") as mock_cdll:
            main_module._tune_glibc_malloc_arenas()
        mock_cdll.assert_not_called()

    def test_calls_mallopt_with_default_cap_on_linux(self, monkeypatch) -> None:
        monkeypatch.setattr(main_module.sys, "platform", "linux")
        monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
        mock_libc = MagicMock()
        with patch("ctypes.CDLL", return_value=mock_libc) as mock_cdll:
            main_module._tune_glibc_malloc_arenas()
        mock_cdll.assert_called_once_with("libc.so.6")
        mock_libc.mallopt.assert_called_once_with(-8, 2)  # M_ARENA_MAX, default cap

    def test_respects_malloc_arena_max_env_override(self, monkeypatch) -> None:
        monkeypatch.setattr(main_module.sys, "platform", "linux")
        monkeypatch.setenv("MALLOC_ARENA_MAX", "4")
        mock_libc = MagicMock()
        with patch("ctypes.CDLL", return_value=mock_libc):
            main_module._tune_glibc_malloc_arenas()
        mock_libc.mallopt.assert_called_once_with(-8, 4)

    def test_never_raises_when_libc_unavailable(self, monkeypatch) -> None:
        monkeypatch.setattr(main_module.sys, "platform", "linux")
        with patch("ctypes.CDLL", side_effect=OSError("no libc here")):
            main_module._tune_glibc_malloc_arenas()  # must not raise

    def test_never_raises_when_mallopt_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(main_module.sys, "platform", "linux")
        mock_libc = MagicMock(spec=[])  # no mallopt attribute at all
        with patch("ctypes.CDLL", return_value=mock_libc):
            main_module._tune_glibc_malloc_arenas()  # must not raise
