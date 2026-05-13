import pytest
from unittest.mock import AsyncMock

from bot.memory import service as memory_service


@pytest.mark.asyncio
async def test_get_memory_service_initializes_global_without_unboundlocal(monkeypatch):
    monkeypatch.setattr(memory_service, "_memory_service", None)
    monkeypatch.setattr(memory_service.CuratedMemoryService, "start", AsyncMock(return_value=None))

    bot = object()
    service = await memory_service.get_memory_service(bot=bot)

    assert service is memory_service._memory_service
    assert service.bot is bot
    memory_service.CuratedMemoryService.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_memory_service_reuses_existing_instance(monkeypatch):
    monkeypatch.setattr(memory_service, "_memory_service", None)
    monkeypatch.setattr(memory_service.CuratedMemoryService, "start", AsyncMock(return_value=None))

    first_bot = object()
    first = await memory_service.get_memory_service(bot=first_bot)
    second_bot = object()
    second = await memory_service.get_memory_service(bot=second_bot)

    assert first is second
    assert second.bot is second_bot
    memory_service.CuratedMemoryService.start.assert_awaited_once()
