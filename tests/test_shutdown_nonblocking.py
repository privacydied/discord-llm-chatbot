from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import bot.shutdown as shutdown_mod
from bot.shutdown import GracefulShutdown


@pytest.mark.asyncio
async def test_save_all_data_runs_profile_flush_off_event_loop(monkeypatch) -> None:
    bot = MagicMock()
    bot.is_closed.return_value = False
    shutdown = GracefulShutdown(bot)

    calls = []

    def fake_save_all_profiles() -> bool:
        calls.append("user")
        return True

    def fake_save_all_server_profiles() -> bool:
        calls.append("server")
        return True

    async def fake_to_thread(func, *args, **kwargs):
        calls.append("to_thread")
        return func(*args, **kwargs)

    monkeypatch.setattr(shutdown_mod, "save_all_profiles", fake_save_all_profiles)
    monkeypatch.setattr(shutdown_mod, "save_all_server_profiles", fake_save_all_server_profiles)
    monkeypatch.setattr(shutdown_mod.asyncio, "to_thread", fake_to_thread)

    await shutdown._save_all_data()

    assert calls == ["to_thread", "user", "server"]
