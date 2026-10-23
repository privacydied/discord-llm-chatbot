"""Replies to allowlisted bots, loop guard, ping rules [SFT][REH]."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.core import peer_bots
from bot.core.peer_bots import BotReplyBudget, accept_peer_bot_message, is_addressed_peer_bot, parse_allowlist

OWN = 1392310898311303259
PEER = 555000000000000001
OTHER_BOT = 555000000000000002
CFG = {"BOT_REPLY_ALLOWLIST": f"{PEER}", "BOT_REPLY_MAX_PER_WINDOW": 2, "BOT_REPLY_WINDOW_S": 300}


@pytest.fixture(autouse=True)
def _fresh_budget(monkeypatch):
    monkeypatch.setattr(peer_bots, "_budget", BotReplyBudget())


def _msg(author_id: int, *, mention: bool = False, reply_to: int | None = None, channel: int = 7, bot: bool = True):
    reference = None
    if reply_to is not None:
        reference = SimpleNamespace(message_id=1, resolved=SimpleNamespace(author=SimpleNamespace(id=reply_to)))
    return SimpleNamespace(
        id=99,
        author=SimpleNamespace(id=author_id, bot=bot),
        mentions=[SimpleNamespace(id=OWN)] if mention else [],
        reference=reference,
        channel=SimpleNamespace(id=channel),
    )


def test_parse_allowlist_accepts_mentions_and_separators():
    assert parse_allowlist(f"<@{PEER}>; {OTHER_BOT}, junk,") == {PEER, OTHER_BOT}
    assert parse_allowlist(None) == frozenset()


def test_only_allowlisted_bots_that_address_us():
    assert is_addressed_peer_bot(_msg(PEER, mention=True), CFG, OWN)
    assert is_addressed_peer_bot(_msg(PEER, reply_to=OWN), CFG, OWN)
    assert not is_addressed_peer_bot(_msg(PEER), CFG, OWN)  # no ambient replies to bots
    assert not is_addressed_peer_bot(_msg(PEER, reply_to=OTHER_BOT), CFG, OWN)
    assert not is_addressed_peer_bot(_msg(OTHER_BOT, mention=True), CFG, OWN)
    assert not is_addressed_peer_bot(_msg(OWN, mention=True), CFG, OWN)
    assert not is_addressed_peer_bot(_msg(PEER, mention=True), {}, OWN)


def test_loop_guard_caps_replies_per_channel():
    assert accept_peer_bot_message(_msg(PEER, mention=True), CFG, OWN)
    assert accept_peer_bot_message(_msg(PEER, mention=True), CFG, OWN)
    assert not accept_peer_bot_message(_msg(PEER, mention=True), CFG, OWN)
    assert accept_peer_bot_message(_msg(PEER, mention=True, channel=8), CFG, OWN)  # per channel


def test_budget_window_expires():
    budget = BotReplyBudget()
    assert budget.try_consume(1, 1, 300, now=0)
    assert not budget.try_consume(1, 1, 300, now=299)
    assert budget.try_consume(1, 1, 300, now=300)


async def test_message_processor_drops_unlisted_bots_and_accepts_peers():
    from bot.core.message_processor import MessageProcessor

    bot = SimpleNamespace(user=SimpleNamespace(id=OWN), config=CFG)
    processor = MessageProcessor.__new__(MessageProcessor)
    processor.bot = bot
    processor.logger = SimpleNamespace(debug=lambda *a, **k: None)
    assert processor._accept_peer_bot(_msg(PEER, mention=True)) is True
    assert processor._accept_peer_bot(_msg(OTHER_BOT, mention=True)) is False
    assert processor._accept_peer_bot(_msg(PEER)) is False


def test_send_path_treats_peer_bots_as_people():
    from bot.core.bot import LLMBot

    bot = LLMBot.__new__(LLMBot)
    bot.config = CFG
    assert bot._is_ignored_bot(SimpleNamespace(id=OTHER_BOT, bot=True)) is True
    assert bot._is_ignored_bot(SimpleNamespace(id=PEER, bot=True)) is False
    assert bot._is_ignored_bot(SimpleNamespace(id=1, bot=False)) is False


def test_allowlist_off_answers_any_addressing_bot():
    cfg = {**CFG, "BOT_REPLY_ALLOWLIST_ENABLED": False}
    assert is_addressed_peer_bot(_msg(OTHER_BOT, mention=True), cfg, OWN)
    assert not is_addressed_peer_bot(_msg(OTHER_BOT), cfg, OWN)  # still must address us
    assert not is_addressed_peer_bot(_msg(OWN, mention=True), cfg, OWN)
    assert accept_peer_bot_message(_msg(OTHER_BOT, mention=True), cfg, OWN)
    assert accept_peer_bot_message(_msg(OTHER_BOT, mention=True), cfg, OWN)
    assert not accept_peer_bot_message(_msg(OTHER_BOT, mention=True), cfg, OWN)  # loop guard


def test_send_path_with_allowlist_off_pings_any_bot():
    from bot.core.bot import LLMBot

    bot = LLMBot.__new__(LLMBot)
    bot.config = {"BOT_REPLY_ALLOWLIST_ENABLED": "false"}
    assert bot._is_ignored_bot(SimpleNamespace(id=OTHER_BOT, bot=True)) is False
