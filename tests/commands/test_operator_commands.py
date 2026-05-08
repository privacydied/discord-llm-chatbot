from __future__ import annotations

import sys
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.commands.operator_commands import OperatorCommands


@pytest.fixture
def mock_bot():
    bot = MagicMock()
    bot.config = {
        "TEXT_BACKEND": "openrouter",
        "rag_enabled": True,
    }
    bot.add_cog = AsyncMock()
    bot._boot_time = 1_700_000_000.0
    bot._active_long_running_tasks = {}
    bot._user_queues = {}
    bot.tts_manager = None
    bot.web_extraction_service = None
    return bot


@pytest.fixture
def operator_cog(mock_bot):
    return OperatorCommands(mock_bot)


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.guild = MagicMock()
    ctx.guild.id = 123
    ctx.reply = AsyncMock()
    ctx.send = AsyncMock()
    ctx.author = MagicMock()
    ctx.author.guild_permissions.administrator = True
    return ctx


def test_operator_commands_are_registered(operator_cog):
    command_names = {command.name for command in operator_cog.get_commands()}
    assert {"help", "status", "feature"}.issubset(command_names)


@pytest.mark.asyncio
async def test_help_command_returns_capability_embed(operator_cog, mock_ctx):
    await operator_cog.help_command.callback(operator_cog, mock_ctx)

    mock_ctx.reply.assert_called_once()
    embed = mock_ctx.reply.call_args.kwargs["embed"]
    assert isinstance(embed, discord.Embed)
    assert embed.title == "🤖 Bot Capability Card"
    assert "Chat" in embed.fields[0].value
    assert "!search" in embed.fields[1].value


@pytest.mark.asyncio
async def test_status_command_reports_core_health_fields(
    operator_cog, mock_ctx, monkeypatch
):
    fake_psutil = SimpleNamespace(
        Process=lambda: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=128 * 1024 * 1024)
        )
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(
        "bot.commands.operator_commands.stt_manager",
        SimpleNamespace(available=True, default_spec=SimpleNamespace(size="small")),
    )
    monkeypatch.setattr(
        "bot.commands.operator_commands.get_server_feature_toggles",
        lambda guild_id: {
            "stt": True,
            "tts": True,
            "vision": True,
            "image_generation": True,
            "web_extraction": True,
            "x_twitter_extraction": True,
            "rag": True,
        },
    )

    tts_manager = SimpleNamespace(
        get_status=lambda: {"available": True, "engine": "edge"},
        _file_cache={"a": object(), "b": object()},
        _cache_max=8,
    )
    operator_cog.bot.tts_manager = tts_manager
    operator_cog.bot.web_extraction_service = SimpleNamespace(_tier_b_available=True)
    operator_cog.bot._active_long_running_tasks = {1: object()}
    queue = Queue()
    queue.put("queued")
    operator_cog.bot._user_queues = {42: queue}

    await operator_cog.status_command.callback(operator_cog, mock_ctx)

    mock_ctx.reply.assert_called_once()
    embed = mock_ctx.reply.call_args.kwargs["embed"]
    fields = {field.name: field.value for field in embed.fields}
    assert fields["Uptime"].endswith("s") or "m" in fields["Uptime"]
    assert fields["Active backend"] == "openrouter"
    assert "enabled" in fields["RAG"]
    assert "loaded=True" in fields["STT"]
    assert "cache=2/8" in fields["TTS"]
    assert "configured=" in fields["Playwright"]
    assert "backpressure=yes" in fields["Queue / backpressure"]


@pytest.mark.asyncio
async def test_status_command_gracefully_handles_missing_optional_fields(
    mock_bot, monkeypatch
):
    cog = OperatorCommands(mock_bot)
    ctx = MagicMock()
    ctx.guild = None
    ctx.reply = AsyncMock()

    monkeypatch.setattr(
        "bot.commands.operator_commands.stt_manager",
        SimpleNamespace(available=False, default_spec=None),
    )
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    monkeypatch.setattr(
        "bot.commands.operator_commands.get_server_feature_toggles",
        lambda guild_id: {
            "stt": False,
            "tts": False,
            "vision": False,
            "image_generation": False,
            "web_extraction": False,
            "x_twitter_extraction": False,
            "rag": False,
        },
    )

    await cog.status_command.callback(cog, ctx)

    embed = ctx.reply.call_args.kwargs["embed"]
    fields = {field.name: field.value for field in embed.fields}
    assert fields["RAG"]
    assert "manager=missing" in fields["TTS"]
    assert "configured=" in fields["Playwright"]
