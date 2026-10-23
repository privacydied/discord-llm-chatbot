"""NotSoBot-style ".edit <prompt>" command [CA][REH]."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.action import BotAction
from bot.router_components import conversational_edit as ce
from bot.router_components.conversational_edit import (
    ResolvedEditImage,
    is_dot_edit_command,
    parse_dot_edit_command,
    resolve_edit_source_image,
)

# -- parser ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "prompt"),
    [
        (".edit make her hair green", "make her hair green"),
        (".EDIT: make her hair green", "make her hair green"),
        ("  .edit   give him a hat  ", "give him a hat"),
        (".edit", ""),
        (".edit -seed 3 neon outfit", "-seed 3 neon outfit"),
    ],
)
def test_parses_dot_edit(text, prompt):
    assert parse_dot_edit_command(text).prompt == prompt
    assert is_dot_edit_command(text)


@pytest.mark.parametrize("text", [".editor is broken", "edit this", ".", "...", "lol .edit this", "!edit x", ""])
def test_ignores_non_commands(text):
    assert parse_dot_edit_command(text) is None
    assert not is_dot_edit_command(text)


def test_custom_prefix():
    assert parse_dot_edit_command(";edit make it blue", ";").prompt == "make it blue"
    assert parse_dot_edit_command(".edit make it blue", ";") is None


# -- resolver channel lookback --------------------------------------------------


def _history(messages):
    def history(limit, before):
        async def gen():
            for m in messages[:limit]:
                yield m

        return gen()

    return history


async def test_lookback_finds_newest_channel_image(monkeypatch):
    with_image = SimpleNamespace(name="img")
    monkeypatch.setattr(ce, "collect_image_urls_from_message", lambda m: ["ref"] if m is with_image else [])
    download = AsyncMock(return_value=ResolvedEditImage(data=b"x", content_type="image/png", source="channel"))
    monkeypatch.setattr(ce, "_download_ref", download)
    message = SimpleNamespace(
        content=".edit x",
        reference=None,
        channel=SimpleNamespace(history=_history([SimpleNamespace(name="text"), with_image])),
    )
    resolved = await resolve_edit_source_image(message, 25, channel_lookback=10)
    assert resolved.source == "channel"
    assert download.await_args.args[1] == "channel"


async def test_no_lookback_by_default(monkeypatch):
    monkeypatch.setattr(ce, "collect_image_urls_from_message", lambda m: [])
    history = MagicMock()
    message = SimpleNamespace(content="edit x", reference=None, channel=SimpleNamespace(history=history))
    assert await resolve_edit_source_image(message, 25) is None
    history.assert_not_called()


# -- router -----------------------------------------------------------------


def _router(cfg=None, orchestrator=True):
    from bot.router import Router

    router = Router.__new__(Router)
    router.config = {"DOT_EDIT_ENABLED": True, "DOT_EDIT_PREFIX": ".", "DOT_EDIT_LOOKBACK": 25, **(cfg or {})}
    router._vision_orchestrator = object() if orchestrator else None
    router.logger = MagicMock()
    router._metric_inc = MagicMock()
    router._conversational_edit_enabled = MagicMock(return_value=True)
    router._image_edit_unavailable_action = MagicMock(return_value=None)
    router._run_conversational_edit_job = AsyncMock(return_value=BotAction(content="", files=[]))
    return router


def _message():
    return SimpleNamespace(id=1, guild=SimpleNamespace(id=2), author=SimpleNamespace(id=3), channel=SimpleNamespace(id=4))


async def test_handle_dot_edit_runs_job_with_lookback(monkeypatch):
    resolved = ResolvedEditImage(data=b"img", content_type="image/png", source="channel")
    resolve = AsyncMock(return_value=resolved)
    monkeypatch.setattr("bot.router.resolve_edit_source_image", resolve)
    router = _router()
    await router._handle_dot_edit(_message(), ".edit make her hair green -seed 7")
    assert resolve.await_args.kwargs["channel_lookback"] == 25
    _, prompt, got_resolved, flags = router._run_conversational_edit_job.await_args.args
    assert prompt == "make her hair green" and got_resolved is resolved and flags.seed == 7


async def test_handle_dot_edit_usage_help_and_errors(monkeypatch):
    router = _router()
    assert "usage" in (await router._handle_dot_edit(_message(), ".edit")).content
    assert "-seed" in (await router._handle_dot_edit(_message(), ".edit -h")).content
    bad = await router._handle_dot_edit(_message(), ".edit -steps 999 x")
    assert bad.error
    router._run_conversational_edit_job.assert_not_awaited()


async def test_handle_dot_edit_no_image(monkeypatch):
    monkeypatch.setattr("bot.router.resolve_edit_source_image", AsyncMock(return_value=None))
    action = await _router()._handle_dot_edit(_message(), ".edit make it blue")
    assert action.error and "no image found" in action.content


async def test_handle_dot_edit_disabled_and_no_provider():
    router = _router()
    router._conversational_edit_enabled.return_value = False
    assert "disabled" in (await router._handle_dot_edit(_message(), ".edit x")).content
    router = _router()
    router._image_edit_unavailable_action.return_value = BotAction(content="no provider", error=True)
    assert (await router._handle_dot_edit(_message(), ".edit make it blue")).content == "no provider"


def test_is_dot_edit_respects_config():
    assert _router()._is_dot_edit(".edit x")
    assert not _router({"DOT_EDIT_ENABLED": False})._is_dot_edit(".edit x")
    assert _router({"DOT_EDIT_PREFIX": ";"})._is_dot_edit(";edit x")


# -- gate: unaddressed ".edit" is a command, not chat ----------------------------


def _gate_router(**cfg):
    import logging

    from bot.router import Router

    bot = MagicMock()
    bot.config = {
        "OWNER_IDS": [],
        "REPLY_TRIGGERS": ["dm", "mention", "reply", "bot_threads", "owner", "command_prefix"],
        "REQUIRE_MENTION_IN_GUILDS": True,
        "DM_REQUIRE_MENTION": False,
        "COMMAND_PREFIX": "!",
        "AMBIENT_REPLY_ENABLED": False,
        **cfg,
    }
    bot.user = SimpleNamespace(id=9999)
    router = Router(bot=bot, flow_overrides={}, logger=MagicMock(spec=logging.Logger))
    router._detect_direct_vision_triggers = MagicMock(return_value=False)
    router._try_ambient_reply = MagicMock(return_value=False)
    return router


def _guild_message(content: str):
    import discord

    message = MagicMock()
    message.id = 55
    message.channel = MagicMock(spec=discord.TextChannel)
    message.guild = SimpleNamespace(id=42)
    message.author = SimpleNamespace(id=1111, bot=False)
    message.content = content
    message.mentions = []
    message.reference = None
    return message


def test_gate_allows_unaddressed_dot_edit():
    assert _gate_router()._should_process_message(_guild_message(".edit make her hair green")) is True


@pytest.mark.parametrize("content", [".editor sucks", "lol nice", "..."])
def test_gate_still_drops_ordinary_unaddressed_chat(content):
    assert _gate_router()._should_process_message(_guild_message(content)) is False


def test_gate_respects_kill_switch():
    router = _gate_router(DOT_EDIT_ENABLED=False)
    assert router._should_process_message(_guild_message(".edit make her hair green")) is False


async def test_dispatch_message_hands_dot_edit_to_handler():
    router = _gate_router()
    expected = BotAction(content="edited")
    router._handle_dot_edit = AsyncMock(return_value=expected)
    message = _guild_message(".edit make her hair green")
    message.attachments = []
    assert await router.dispatch_message(message) is expected
    assert router._handle_dot_edit.await_args.args[1] == ".edit make her hair green"


# -- bounded Discord reads (a silent discord.py wait once stalled .edit 3m25s) ----------


async def test_reply_uses_cached_reference_without_api_call(monkeypatch):
    import discord

    replied = MagicMock(spec=discord.Message)
    monkeypatch.setattr(ce, "collect_image_urls_from_message", lambda m: ["ref"] if m is replied else [])
    monkeypatch.setattr(ce, "_download_ref", AsyncMock(return_value=ResolvedEditImage(data=b"x", content_type="image/png", source="reply")))
    channel = SimpleNamespace(fetch_message=AsyncMock())
    message = SimpleNamespace(content=".edit x", channel=channel, reference=SimpleNamespace(message_id=5, resolved=replied, cached_message=None))
    assert (await resolve_edit_source_image(message, 25)).source == "reply"
    channel.fetch_message.assert_not_awaited()


async def test_hung_reference_fetch_is_bounded(monkeypatch):
    import asyncio

    monkeypatch.setattr(ce, "DISCORD_READ_TIMEOUT_S", 0.05)
    monkeypatch.setattr(ce, "collect_image_urls_from_message", lambda m: [])

    async def hang(_id):
        await asyncio.sleep(30)

    message = SimpleNamespace(content=".edit x", channel=SimpleNamespace(fetch_message=hang), reference=SimpleNamespace(message_id=5, resolved=None, cached_message=None))
    assert await asyncio.wait_for(resolve_edit_source_image(message, 25), timeout=2) is None


async def test_hung_channel_history_is_bounded(monkeypatch):
    import asyncio

    monkeypatch.setattr(ce, "DISCORD_READ_TIMEOUT_S", 0.05)
    monkeypatch.setattr(ce, "collect_image_urls_from_message", lambda m: [])

    def history(limit, before):
        async def gen():
            await asyncio.sleep(30)
            yield None

        return gen()

    message = SimpleNamespace(content=".edit x", reference=None, channel=SimpleNamespace(history=history))
    assert await asyncio.wait_for(resolve_edit_source_image(message, 25, channel_lookback=5), timeout=2) is None


# -- upload permission is checked before the edit runs ----------------------------


def _perm_message(attach):
    perms = SimpleNamespace(attach_files=attach)
    return SimpleNamespace(
        id=1,
        guild=SimpleNamespace(id=2, me=object()),
        author=SimpleNamespace(id=3),
        channel=SimpleNamespace(id=4, permissions_for=lambda me: perms),
    )


async def test_edit_refused_without_attach_files_before_any_job():
    from bot.router import Router

    router = Router.__new__(Router)
    router.logger = MagicMock()
    router._metric_inc = MagicMock()
    router._vision_orchestrator = SimpleNamespace(submit_job=AsyncMock())
    resolved = ResolvedEditImage(data=b"x", content_type="image/png", source="reply")
    action = await Router._run_conversational_edit_job(router, _perm_message(False), "make it green", resolved)
    assert action.error and "Attach Files" in action.content
    router._vision_orchestrator.submit_job.assert_not_awaited()


def test_attach_check_passes_when_allowed_or_unknown():
    from bot.router import Router

    router = Router.__new__(Router)
    router.logger = MagicMock()
    router._metric_inc = MagicMock()
    assert Router._edit_upload_denied_action(router, _perm_message(True)) is None
    assert Router._edit_upload_denied_action(router, SimpleNamespace(guild=None, channel=None)) is None  # DM
    assert Router._edit_upload_denied_action(router, MagicMock()) is None  # mocks never refuse
