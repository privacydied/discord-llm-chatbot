from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.ext import commands

from bot.commands.memory_extended_cmds import ExtendedMemoryCommands


@pytest.fixture
def mock_bot():
    bot = MagicMock()
    bot.owner_ids = {12345}
    return bot


@pytest.fixture
def mock_ctx(mock_bot):
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.id = 12345
    ctx.author.guild_permissions.administrator = True
    ctx.guild = MagicMock()
    ctx.guild.id = 54321
    ctx.send = AsyncMock()
    ctx.bot = mock_bot
    return ctx


@pytest.fixture
def memory_cog(mock_bot):
    return ExtendedMemoryCommands(mock_bot)


def test_memory_status_command_is_registered():
    assert isinstance(
        ExtendedMemoryCommands.__dict__["memory_status"], commands.Command
    )


@pytest.mark.asyncio
async def test_memory_status_command_sends_embed(memory_cog, mock_ctx, monkeypatch):
    fake_service = MagicMock()
    fake_service.enabled = True
    fake_service.queue = MagicMock()
    fake_service.queue._queue = [1, 2, 3]
    fake_service.semantic_store = MagicMock()
    fake_service.semantic_store._collection = object()
    fake_service.store = MagicMock()
    fake_service.store._conn = object()

    monkeypatch.setattr(
        "bot.commands.memory_extended_cmds.get_memory_service",
        AsyncMock(return_value=fake_service),
    )

    await ExtendedMemoryCommands.__dict__["memory_status"].callback(
        memory_cog, mock_ctx
    )

    mock_ctx.send.assert_called_once()
    embed = mock_ctx.send.call_args.kwargs["embed"]
    assert isinstance(embed, discord.Embed)
    assert embed.title == "Memory Service Status"
    assert any(
        field.name == "Enabled" and field.value == "True" for field in embed.fields
    )
