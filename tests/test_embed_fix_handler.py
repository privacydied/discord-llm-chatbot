"""Discord-side behaviour of the SaucyBot port (bot/embed_fix/handler.py) and
its wiring into LLMBot.on_message."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.embed_fix import handler as h

_ALL_PERMS = dict(send_messages=True, embed_links=True, read_message_history=True, send_messages_in_threads=True, manage_messages=True)


def _perms(**overrides):
    return discord.Permissions(**{**_ALL_PERMS, **overrides})


def _channel(*, perms=None, nsfw=False, spec=None):
    channel = MagicMock(spec=spec) if spec else MagicMock()
    channel.permissions_for = MagicMock(return_value=perms or _perms())
    channel.is_nsfw = MagicMock(return_value=nsfw)
    return channel


def _message(content, *, channel=None, guild=True, author_bot=False):
    msg = MagicMock()
    msg.id = 1
    msg.content = content
    msg.author = SimpleNamespace(bot=author_bot, id=2)
    msg.guild = SimpleNamespace(id=3, me=SimpleNamespace(id=4)) if guild else None
    msg.channel = channel or _channel()
    msg.edit = AsyncMock()
    msg.reply = AsyncMock()
    return msg


@pytest.fixture
def reply(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(h, "safe_reply", mock)
    monkeypatch.setattr(h, "is_server_feature_enabled", lambda guild_id, feature: True)
    return mock


async def test_replies_with_fixed_link_and_suppresses_original(reply):
    msg = _message("lol https://x.com/jack/status/20")
    fixes = await h.maybe_fix_embeds(msg, {})
    assert [f.rewritten for f in fixes] == ["https://vxtwitter.com/jack/status/20"]
    reply.assert_awaited_once()
    args, kwargs = reply.await_args
    assert args[1] == "https://vxtwitter.com/jack/status/20"
    assert kwargs["mention_author"] is False
    assert kwargs["allowed_mentions"].to_dict()["parse"] == []
    msg.edit.assert_awaited_once_with(suppress=True)


async def test_multiple_links_go_in_one_reply(reply):
    msg = _message("https://x.com/a/status/1 https://instagram.com/reel/AB")
    await h.maybe_fix_embeds(msg, {})
    assert reply.await_args.args[1] == "https://vxtwitter.com/a/status/1\nhttps://vxinstagram.com/reel/AB/"


async def test_no_suppress_without_manage_messages(reply):
    msg = _message("https://x.com/a/status/1", channel=_channel(perms=_perms(manage_messages=False)))
    assert await h.maybe_fix_embeds(msg, {})
    msg.edit.assert_not_awaited()


async def test_no_suppress_when_disabled_in_config(reply):
    msg = _message("https://x.com/a/status/1")
    assert await h.maybe_fix_embeds(msg, {"EMBED_FIX_SUPPRESS_ORIGINAL": False})
    msg.edit.assert_not_awaited()


@pytest.mark.parametrize(
    "perms",
    [_perms(send_messages=False), _perms(embed_links=False), _perms(read_message_history=False)],
)
async def test_missing_channel_permissions_skip(reply, perms):
    msg = _message("https://x.com/a/status/1", channel=_channel(perms=perms))
    assert await h.maybe_fix_embeds(msg, {}) == []
    reply.assert_not_awaited()


async def test_thread_requires_send_in_threads(reply):
    thread = _channel(perms=_perms(send_messages_in_threads=False), spec=discord.Thread)
    assert await h.maybe_fix_embeds(_message("https://x.com/a/status/1", channel=thread), {}) == []
    thread_ok = _channel(perms=_perms(), spec=discord.Thread)
    assert await h.maybe_fix_embeds(_message("https://x.com/a/status/1", channel=thread_ok), {})


@pytest.mark.parametrize(
    ("msg_kwargs", "config"),
    [
        ({"guild": False}, {}),  # DMs: SaucyBot CanCreateEmbed(allowDirectMessages=false)
        ({"author_bot": True}, {}),
        ({}, {"EMBED_FIX_ENABLED": False}),
        ({}, {"EMBED_FIX_DISABLED_SITES": "twitter"}),
    ],
)
async def test_skip_conditions(reply, msg_kwargs, config):
    assert await h.maybe_fix_embeds(_message("https://x.com/a/status/1", **msg_kwargs), config) == []
    reply.assert_not_awaited()


@pytest.mark.parametrize("content", ["<https://x.com/a/status/1>", "||https://x.com/a/status/1||", "no links here", "https://example.com"])
async def test_opt_out_and_non_matching_content(reply, content):
    assert await h.maybe_fix_embeds(_message(content), {}) == []
    reply.assert_not_awaited()


async def test_guild_toggle_off_skips(reply, monkeypatch):
    seen = []
    monkeypatch.setattr(h, "is_server_feature_enabled", lambda guild_id, feature: seen.append((guild_id, feature)) or False)
    assert await h.maybe_fix_embeds(_message("https://x.com/a/status/1"), {}) == []
    assert seen == [(3, "embed_fix")]
    reply.assert_not_awaited()


async def test_toggle_lookup_failure_fails_open(reply, monkeypatch):
    def boom(guild_id, feature):
        raise OSError("disk")

    monkeypatch.setattr(h, "is_server_feature_enabled", boom)
    assert await h.maybe_fix_embeds(_message("https://x.com/a/status/1"), {})


async def test_nsfw_site_gated_to_nsfw_channels(reply):
    content = "https://furaffinity.net/view/1 https://x.com/a/status/1"
    sfw = await h.maybe_fix_embeds(_message(content, channel=_channel(nsfw=False)), {})
    assert [f.site for f in sfw] == ["twitter"]
    nsfw = await h.maybe_fix_embeds(_message(content, channel=_channel(nsfw=True)), {})
    assert [f.site for f in nsfw] == ["furaffinity", "twitter"]
    unrestricted = await h.maybe_fix_embeds(_message(content), {"EMBED_FIX_RESTRICT_NSFW": False})
    assert [f.site for f in unrestricted] == ["furaffinity", "twitter"]


async def test_only_nsfw_links_in_sfw_channel_sends_nothing(reply):
    assert await h.maybe_fix_embeds(_message("https://furaffinity.net/view/1"), {}) == []
    reply.assert_not_awaited()


async def test_reply_failure_skips_suppress(monkeypatch):
    monkeypatch.setattr(h, "is_server_feature_enabled", lambda guild_id, feature: True)
    resp = SimpleNamespace(status=403, reason="Forbidden")
    monkeypatch.setattr(h, "safe_reply", AsyncMock(side_effect=discord.Forbidden(resp, "nope")))
    msg = _message("https://x.com/a/status/1")
    assert await h.maybe_fix_embeds(msg, {}) == []
    msg.edit.assert_not_awaited()


async def test_suppress_failure_is_swallowed(reply):
    msg = _message("https://x.com/a/status/1")
    msg.edit = AsyncMock(side_effect=discord.HTTPException(SimpleNamespace(status=500, reason="x"), "boom"))
    assert await h.maybe_fix_embeds(msg, {})


async def test_real_safe_reply_keeps_url_intact(monkeypatch):
    """The public-output sanitizer must not mangle the rewritten URL."""
    monkeypatch.setattr(h, "is_server_feature_enabled", lambda guild_id, feature: True)
    msg = _message("https://x.com/some_user/status/1234567890123456789?s=20")
    await h.maybe_fix_embeds(msg, {})
    msg.reply.assert_awaited_once()
    assert msg.reply.await_args.args[0] == "https://vxtwitter.com/some_user/status/1234567890123456789"
    assert msg.reply.await_args.kwargs["mention_author"] is False


def test_settings_from_config_parses_strings():
    s = h.EmbedFixSettings.from_config({"EMBED_FIX_DISABLED_SITES": "tiktok,Reddit", "EMBED_FIX_DOMAIN_OVERRIDES": "twitter=fixupx.com", "EMBED_FIX_MAX_LINKS": 3})
    assert s.disabled_sites == frozenset({"tiktok", "reddit"})
    assert s.domain_overrides == {"twitter": "fixupx.com"}
    assert s.max_links == 3


# --- wiring into LLMBot -----------------------------------------------------


def _fake_bot(config):
    return SimpleNamespace(config=config, logger=MagicMock(), _track_background_task=MagicMock())


async def test_llmbot_schedules_tracked_task(monkeypatch):
    from bot.core import bot as core_bot

    called = []

    async def fake_fix(message, config):
        called.append(message.id)
        return []

    monkeypatch.setattr(core_bot, "maybe_fix_embeds", fake_fix)
    fake = _fake_bot({})
    core_bot.LLMBot._schedule_embed_fix(fake, _message("https://x.com/a/status/1"))
    fake._track_background_task.assert_called_once()
    await fake._track_background_task.call_args.args[0]
    assert called == [1]


@pytest.mark.parametrize(
    ("config", "content", "guild"),
    [({"EMBED_FIX_ENABLED": False}, "https://x.com/a/status/1", True), ({}, "plain chat", True), ({}, "https://x.com/a/status/1", False)],
)
def test_llmbot_skips_scheduling(config, content, guild):
    from bot.core import bot as core_bot

    fake = _fake_bot(config)
    core_bot.LLMBot._schedule_embed_fix(fake, _message(content, guild=guild))
    fake._track_background_task.assert_not_called()


def test_on_message_schedules_before_router_gate():
    """Link posts are usually not addressed to the bot, so the hook must run
    before the router gate can drop the message."""
    import inspect

    from bot.core.bot import LLMBot

    src = inspect.getsource(LLMBot.on_message)
    assert src.index("_schedule_embed_fix(message)") < src.index("_should_process_message(message)")
    assert src.index("message_processor.on_message(message)") < src.index("_schedule_embed_fix(message)")


def test_feature_toggle_registered():
    from bot.server_features import FEATURE_DEFAULTS, normalize_feature_name

    assert FEATURE_DEFAULTS["embed_fix"] is True
    for alias in ("saucy", "linkfix", "embeds", "embed-fix"):
        assert normalize_feature_name(alias) == "embed_fix"
