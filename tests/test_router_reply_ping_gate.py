"""Reply-ping gate: a reply to the bot with Discord's reply ping turned off
must not trigger a response (ALLOW_REPLY_TO_BOT_WITHOUT_MENTION=false)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import discord
import pytest

from bot.router import GATE_REASON_REPLY_PING_OFF, Router

BOT_ID = 9999
BASE_CONFIG = {
    "OWNER_IDS": [],
    "REPLY_TRIGGERS": ["dm", "mention", "reply", "bot_threads", "owner", "command_prefix"],
    "REQUIRE_MENTION_IN_GUILDS": True,
    "DM_REQUIRE_MENTION": False,
    "BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO": True,
    "COMMAND_PREFIX": "!",
}


def _router(**overrides) -> Router:
    bot = MagicMock()
    bot.config = {**BASE_CONFIG, **overrides}
    bot.user = SimpleNamespace(id=BOT_ID)
    router = Router(bot=bot, flow_overrides={}, logger=MagicMock(spec=logging.Logger))
    router._detect_direct_vision_triggers = MagicMock(return_value=True)  # must not bypass
    router._try_ambient_reply = MagicMock(return_value=True)  # must not bypass
    return router


def _reply_to_bot(*, ping: bool, explicit_mention: bool = False, dm: bool = False) -> MagicMock:
    """Build a reply to a bot message the way discord.py presents it.

    Reply ping ON puts the replied-to author in ``message.mentions``; OFF leaves
    them out. An explicit ``<@id>`` in the text also lands in ``mentions``.
    """
    bot_user = SimpleNamespace(id=BOT_ID)
    message = MagicMock()
    message.id = 1234
    message.channel = MagicMock(spec=discord.DMChannel if dm else discord.TextChannel)
    message.guild = None if dm else SimpleNamespace(id=42)
    message.author = SimpleNamespace(id=1111, bot=False)
    message.content = f"<@{BOT_ID}> what about this" if explicit_mention else "what about this"
    message.mentions = [bot_user] if (ping or explicit_mention) else []
    message.reference = SimpleNamespace(message_id=777, resolved=SimpleNamespace(author=bot_user), cached_message=None)
    return message


def test_reply_with_ping_off_is_blocked_by_default():
    router = _router()  # flag unset -> default False
    message = _reply_to_bot(ping=False)
    assert router._should_process_message(message) is False
    assert router.pop_gate_denied_reason(message.id) == GATE_REASON_REPLY_PING_OFF
    router._try_ambient_reply.assert_not_called()
    router._detect_direct_vision_triggers.assert_not_called()


def test_reply_with_ping_on_is_processed():
    assert _router()._should_process_message(_reply_to_bot(ping=True)) is True


def test_reply_with_ping_off_but_explicit_mention_is_processed():
    assert _router()._should_process_message(_reply_to_bot(ping=False, explicit_mention=True)) is True


def test_flag_true_restores_old_behaviour():
    router = _router(ALLOW_REPLY_TO_BOT_WITHOUT_MENTION=True)
    router._detect_direct_vision_triggers = MagicMock(return_value=False)
    assert router._should_process_message(_reply_to_bot(ping=False)) is True


def test_ping_off_blocked_when_mentions_not_required():
    router = _router(REQUIRE_MENTION_IN_GUILDS=False)
    assert router._should_process_message(_reply_to_bot(ping=False)) is False


def test_dm_reply_with_ping_off_still_processed():
    assert _router()._should_process_message(_reply_to_bot(ping=False, dm=True)) is True


@pytest.mark.parametrize("raw, expected", [(None, False), ("true", True), ("false", False)])
def test_config_default_is_false(monkeypatch, raw, expected):
    from bot.config import invalidate_config_cache, load_config

    if raw is None:
        monkeypatch.delenv("ALLOW_REPLY_TO_BOT_WITHOUT_MENTION", raising=False)
    else:
        monkeypatch.setenv("ALLOW_REPLY_TO_BOT_WITHOUT_MENTION", raw)
    invalidate_config_cache()
    try:
        assert load_config()["ALLOW_REPLY_TO_BOT_WITHOUT_MENTION"] is expected
    finally:
        invalidate_config_cache()
