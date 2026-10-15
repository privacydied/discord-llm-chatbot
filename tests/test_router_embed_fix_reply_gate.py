"""Replies to embed-fix posts (vxtwitter, oginstagram, ...) trigger the bot only
when the user types @bot; Discord's default reply ping alone is ignored."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.embed_fix import handler as embed_handler
from bot.embed_fix.sites import is_embed_fix_post, is_proxy_link_reply
from bot.router import GATE_REASON_EMBED_FIX_REPLY, GATE_REASON_REPLY_PING_OFF, Router

BOT_ID = 9999
BASE_CONFIG = {
    "OWNER_IDS": [],
    "REPLY_TRIGGERS": ["dm", "mention", "reply", "bot_threads", "owner", "command_prefix"],
    "REQUIRE_MENTION_IN_GUILDS": True,
    "DM_REQUIRE_MENTION": False,
    "COMMAND_PREFIX": "!",
}
CARD = [SimpleNamespace(url="https://ppv.to/live/x")]


def _router(**overrides) -> Router:
    bot = MagicMock()
    bot.config = {**BASE_CONFIG, **overrides}
    bot.user = SimpleNamespace(id=BOT_ID)
    router = Router(bot=bot, flow_overrides={}, logger=MagicMock(spec=logging.Logger))
    router._detect_direct_vision_triggers = MagicMock(return_value=True)  # must not bypass
    router._try_ambient_reply = MagicMock(return_value=True)  # must not bypass
    return router


def _reply(target_content: str | None, *, embeds=(), ping: bool = True, typed_mention: bool = False) -> MagicMock:
    """A reply to a bot message, shaped like discord.py (ping ON -> bot in mentions)."""
    bot_user = SimpleNamespace(id=BOT_ID)
    target = SimpleNamespace(author=bot_user, content=target_content, embeds=list(embeds))
    message = MagicMock()
    message.id = 4321
    message.channel = MagicMock(spec=discord.TextChannel)
    message.guild = SimpleNamespace(id=42)
    message.author = SimpleNamespace(id=1111, bot=False)
    message.content = f"<@{BOT_ID}> what is this" if typed_mention else "lmao"
    message.mentions = [bot_user] if (ping or typed_mention) else []
    message.reference = SimpleNamespace(message_id=777, resolved=target, cached_message=None)
    return message


@pytest.mark.parametrize(
    "content, embeds",
    [
        ("https://vxtwitter.com/a/status/1", ()),
        ("https://www.oginstagram.com/reel/AB", ()),
        ("https://vxreddit.com/r/x/comments/1\nhttps://vxtiktok.com/@a/video/2", ()),
        (None, CARD),  # ppv card post, no text
        ("", [SimpleNamespace(url="https://www.instagram.com/p/AB/")]),  # upload card
    ],
)
def test_ping_only_reply_to_embed_fix_post_is_blocked(content, embeds):
    router = _router()
    message = _reply(content, embeds=embeds)
    assert router._should_process_message(message) is False
    assert router.pop_gate_denied_reason(message.id) == GATE_REASON_EMBED_FIX_REPLY
    router._try_ambient_reply.assert_not_called()
    router._detect_direct_vision_triggers.assert_not_called()


def test_typed_mention_in_reply_to_embed_fix_post_is_processed():
    assert _router()._should_process_message(_reply("https://vxtwitter.com/a/status/1", typed_mention=True)) is True


def test_ping_off_reply_to_embed_fix_post_keeps_ping_off_reason():
    router = _router()
    message = _reply("https://vxtwitter.com/a/status/1", ping=False)
    assert router._should_process_message(message) is False
    assert router.pop_gate_denied_reason(message.id) == GATE_REASON_REPLY_PING_OFF


def test_ping_reply_to_normal_bot_answer_still_processed():
    assert _router()._should_process_message(_reply("sure, here's how that works")) is True


def test_flag_false_restores_ping_reply():
    router = _router(EMBED_FIX_REPLY_REQUIRES_MENTION=False)
    assert router._should_process_message(_reply("https://vxtwitter.com/a/status/1")) is True


def test_domain_override_hosts_are_recognised():
    router = _router(EMBED_FIX_DOMAIN_OVERRIDES="twitter=fxtwitter.com")
    assert router._should_process_message(_reply("https://fxtwitter.com/a/status/1")) is False


@pytest.mark.parametrize(
    "content, expected",
    [
        ("https://vxtwitter.com/a/status/1/en", True),
        ("<https://vxtwitter.com/a/status/1>", True),
        ("https://vxtwitter.com/a/status/1 check this", False),
        ("https://example.com/a", False),
        ("https://x.com/a/status/1", False),
        ("", False),
    ],
)
def test_is_proxy_link_reply(content, expected):
    assert is_proxy_link_reply(content) is expected


def test_is_embed_fix_post():
    assert is_embed_fix_post(None) is False
    assert is_embed_fix_post(SimpleNamespace(content="", embeds=CARD)) is True
    assert is_embed_fix_post(SimpleNamespace(content="hello", embeds=CARD)) is False
    # bot answer with a normal link card is not an embed-fix post
    assert is_embed_fix_post(SimpleNamespace(content="", embeds=[SimpleNamespace(url="https://example.com")])) is False


def test_ping_reply_to_bot_answer_mentioning_a_proxy_link_is_processed():
    assert _router()._should_process_message(_reply("here: https://vxtwitter.com/a/status/1 is the tweet")) is True


async def test_embed_fix_reply_is_not_silent(monkeypatch):
    reply = AsyncMock()
    monkeypatch.setattr(embed_handler, "safe_reply", reply)
    fix = embed_handler.find_fixes("https://x.com/a/status/1")[0]
    await embed_handler._send_fixes(MagicMock(id=1), [fix])
    assert "silent" not in reply.await_args.kwargs


@pytest.mark.parametrize("raw, expected", [(None, True), ("false", False), ("true", True)])
def test_config_default_is_true(monkeypatch, raw, expected):
    from bot.config import invalidate_config_cache, load_config

    if raw is None:
        monkeypatch.delenv("EMBED_FIX_REPLY_REQUIRES_MENTION", raising=False)
    else:
        monkeypatch.setenv("EMBED_FIX_REPLY_REQUIRES_MENTION", raw)
    invalidate_config_cache()
    try:
        assert load_config()["EMBED_FIX_REPLY_REQUIRES_MENTION"] is expected
    finally:
        invalidate_config_cache()
