"""Tests for ambient emoji reactions: gate, choice parsing, allowlist, failure paths."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bot.router_components.ambient import (
    REASON_DISABLED,
    REASON_PROBABILITY,
    AmbientCooldowns,
    should_ambient_react,
    should_ambient_reply,
)
from bot.router_components.ambient_react import (
    OUTCOME_ERROR,
    OUTCOME_FIRED,
    OUTCOME_NONE,
    OUTCOME_REJECTED,
    build_candidates,
    can_react,
    parse_choice,
    react_to_message,
)

_REACT_CONFIG = {
    "AMBIENT_REACT_ENABLED": True,
    "AMBIENT_REACT_PROBABILITY": 1.0,
    "AMBIENT_REACT_CHANNEL_COOLDOWN_S": 0,
    "AMBIENT_REACT_GLOBAL_COOLDOWN_S": 0,
    "COMMAND_PREFIX": "!",
}


class _Rng:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _emoji(name: str, usable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(name=name, available=True, is_usable=lambda: usable)


def _msg(content: str = "kendrick just dropped a surprise album", emojis=None, add_reactions=True) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.author.bot = False
    msg.type.value = 0
    msg.mentions = []
    msg.channel.id = 42
    msg.guild.emojis = emojis or []
    msg.channel.permissions_for.return_value = SimpleNamespace(add_reactions=add_reactions, read_message_history=True)
    msg.add_reaction = AsyncMock()
    return msg


# ── gate ─────────────────────────────────────────────────────────────────────


def test_react_gate_fires_with_react_keys():
    fired, reason = should_ambient_react(_msg(), _REACT_CONFIG, AmbientCooldowns(), True, now=1000.0, rng=_Rng(0.0))
    assert (fired, reason) == (True, "fired")


def test_react_gate_disabled_by_default():
    fired, reason = should_ambient_react(_msg(), {}, AmbientCooldowns(), True, now=1000.0, rng=_Rng(0.0))
    assert (fired, reason) == (False, REASON_DISABLED)


def test_react_gate_uses_three_percent_default():
    config = {k: v for k, v in _REACT_CONFIG.items() if k != "AMBIENT_REACT_PROBABILITY"}
    assert should_ambient_react(_msg(), config, AmbientCooldowns(), True, now=1000.0, rng=_Rng(0.029))[0]
    fired, reason = should_ambient_react(_msg(), config, AmbientCooldowns(), True, now=1000.0, rng=_Rng(0.031))
    assert (fired, reason) == (False, REASON_PROBABILITY)


def test_react_keys_do_not_enable_replies():
    fired, reason = should_ambient_reply(_msg(), _REACT_CONFIG, AmbientCooldowns(), True, now=1000.0, rng=_Rng(0.0))
    assert (fired, reason) == (False, REASON_DISABLED)


# ── choice parsing / allowlist ───────────────────────────────────────────────


def test_parse_unicode_exact():
    options = build_candidates(SimpleNamespace(emojis=[]))
    assert parse_choice("💀", options) == (OUTCOME_FIRED, "💀")


def test_parse_variation_selector_tolerated():
    options = build_candidates(SimpleNamespace(emojis=[]))
    assert parse_choice("❤", options) == (OUTCOME_FIRED, "❤️")


def test_parse_none():
    options = build_candidates(SimpleNamespace(emojis=[]))
    assert parse_choice("NONE", options) == (OUTCOME_NONE, None)
    assert parse_choice("none.", options) == (OUTCOME_NONE, None)


def test_parse_custom_emoji_by_name_any_form():
    kekw = _emoji("KEKW")
    options = build_candidates(SimpleNamespace(emojis=[kekw]))
    for answer in (":KEKW:", "kekw", "<:KEKW:123456>"):
        assert parse_choice(answer, options) == (OUTCOME_FIRED, kekw), answer


def test_parse_rejects_off_list_emoji():
    options = build_candidates(SimpleNamespace(emojis=[]))
    assert parse_choice("🖕", options) == (OUTCOME_REJECTED, None)


def test_parse_single_option_inside_chatter():
    options = build_candidates(SimpleNamespace(emojis=[]))
    assert parse_choice("i'd go with 🔥 here", options) == (OUTCOME_FIRED, "🔥")


def test_parse_ambiguous_chatter_rejected():
    options = build_candidates(SimpleNamespace(emojis=[]))
    assert parse_choice("maybe 🔥 or 💀", options) == (OUTCOME_REJECTED, None)


def test_unusable_custom_emoji_excluded():
    options = build_candidates(SimpleNamespace(emojis=[_emoji("locked", usable=False)]))
    assert ":locked:" not in options


def test_can_react_requires_permission():
    assert can_react(_msg())
    assert not can_react(_msg(add_reactions=False))


# ── end to end ───────────────────────────────────────────────────────────────


async def test_react_adds_chosen_emoji():
    msg = _msg()
    generate = AsyncMock(return_value={"text": "🔥"})
    result = await react_to_message(msg, generate, timeout_s=5)
    assert result.outcome == OUTCOME_FIRED
    msg.add_reaction.assert_awaited_once_with("🔥")
    prompt = generate.await_args.kwargs["prompt"]
    assert "kendrick just dropped" in prompt


async def test_react_none_adds_nothing():
    msg = _msg()
    result = await react_to_message(msg, AsyncMock(return_value={"text": "NONE"}), timeout_s=5)
    assert result.outcome == OUTCOME_NONE
    msg.add_reaction.assert_not_awaited()


async def test_react_model_timeout_is_swallowed():
    async def slow(**_kwargs):
        await asyncio.sleep(1)
        return {"text": "🔥"}

    msg = _msg()
    result = await react_to_message(msg, slow, timeout_s=0.01)
    assert result.outcome == OUTCOME_ERROR
    msg.add_reaction.assert_not_awaited()


async def test_react_discord_error_is_swallowed():
    msg = _msg()
    msg.add_reaction.side_effect = RuntimeError("Forbidden")
    result = await react_to_message(msg, AsyncMock(return_value={"text": "💀"}), timeout_s=5)
    assert result.outcome == OUTCOME_ERROR
