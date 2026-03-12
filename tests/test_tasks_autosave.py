import pytest
from unittest.mock import AsyncMock, Mock

import bot.tasks as task_module


@pytest.mark.asyncio
async def test_persist_profiles_nonblocking_uses_to_thread(monkeypatch):
    called = {}

    def fake_sync():
        called["sync_called"] = True
        return True, False

    async def fake_to_thread(func, *args, **kwargs):
        called["func"] = func
        return func(*args, **kwargs)

    monkeypatch.setattr(task_module, "_persist_profiles_sync", fake_sync)
    monkeypatch.setattr(task_module.asyncio, "to_thread", fake_to_thread)

    result = await task_module._persist_profiles_nonblocking()

    assert called["func"] is fake_sync
    assert called["sync_called"] is True
    assert result == (True, False)


@pytest.mark.asyncio
async def test_setup_memory_save_task_awaits_nonblocking_persist(monkeypatch):
    persist_mock = AsyncMock(return_value=(True, True))

    monkeypatch.setattr(
        task_module,
        "load_config",
        lambda: {"PROFILE_AUTOSAVE_INTERVAL": 10},
    )
    monkeypatch.setattr(task_module, "_persist_profiles_nonblocking", persist_mock)

    loop = task_module.setup_memory_save_task(Mock())
    await loop.coro()

    persist_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_task_manager_profile_autosave_awaits_nonblocking_persist(monkeypatch):
    persist_mock = AsyncMock(return_value=(True, True))

    monkeypatch.setattr(
        task_module,
        "load_config",
        lambda: {"PROFILE_AUTOSAVE_INTERVAL": 10},
    )
    monkeypatch.setattr(task_module, "_persist_profiles_nonblocking", persist_mock)
    monkeypatch.setattr(task_module.tasks.Loop, "start", lambda self, *a, **k: None)

    manager = task_module.TaskManager(Mock())
    await manager._start_profile_autosave()

    loop = manager.tasks["profile_autosave"]
    await loop.coro()

    persist_mock.assert_awaited_once()
