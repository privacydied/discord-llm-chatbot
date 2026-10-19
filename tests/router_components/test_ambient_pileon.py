"""Tests for ambient pile-on reactions: gate, mode, eligibility, message lookup."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bot.core.startup import create_bot_intents
from bot.router_components.ambient import (
    REASON_DISABLED,
    REASON_PROBABILITY,
    AmbientCooldowns,
    should_ambient_pileon,
)
from bot.router_components.ambient_pileon import (
    MODE_PICK,
    MODE_PILEON,
    REASON_ALREADY_REACTED,
    REASON_BELOW_MIN_COUNT,
    REASON_FOREIGN_EMOJI,
    REASON_NO_PERMISSION,
    REASON_OWN_MESSAGE,
    REASON_REACTION_GONE,
    REASON_UNAVAILABLE,
    pileon_block_reason,
    react_mode,
    resolve_message,
)

BOT_ID = 999
_CONFIG = {
    "AMBIENT_REACT_ENABLED": True,
    "AMBIENT_REACT_PROBABILITY": 1.0,
    "AMBIENT_REACT_CHANNEL_COOLDOWN_S": 0,
    "AMBIENT_REACT_GLOBAL_COOLDOWN_S": 0,
}


class _Rng:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class _Emoji:
    def __init__(self, text: str, emoji_id: int | None = None) -> None:
        self.text = text
        self.id = emoji_id

    def __str__(self) -> str:
        return self.text


def _bot(known_emoji_ids: tuple[int, ...] = ()) -> MagicMock:
    bot = MagicMock()
    bot.user.id = BOT_ID
    bot.get_emoji.side_effect = lambda i: object() if i in known_emoji_ids else None
    bot.cached_messages = []
    return bot


def _message(emoji_text: str = "💀", count: int = 2, me: bool = False, author_id: int = 1, perms: bool = True):
    msg = MagicMock()
    msg.author.id = author_id
    msg.reactions = [SimpleNamespace(emoji=emoji_text, count=count, me=me)]
    msg.channel.permissions_for.return_value = SimpleNamespace(add_reactions=perms, read_message_history=True)
    return msg


# ── gate ─────────────────────────────────────────────────────────────────────


def test_pileon_gate_fires_and_does_not_record_cooldown():
    cooldowns = AmbientCooldowns()
    fired, reason = should_ambient_pileon(_CONFIG, cooldowns, True, 42, now=1000.0, rng=_Rng(0.0))
    assert (fired, reason) == (True, "fired")
    assert cooldowns.global_elapsed(1000.0) == 1000.0  # untouched


def test_pileon_gate_respects_switch_and_probability():
    assert should_ambient_pileon({}, AmbientCooldowns(), True, 42, now=1.0, rng=_Rng(0.0)) == (False, REASON_DISABLED)
    config = dict(_CONFIG, AMBIENT_REACT_PROBABILITY=0.1)
    assert should_ambient_pileon(config, AmbientCooldowns(), True, 42, now=1.0, rng=_Rng(0.5)) == (
        False,
        REASON_PROBABILITY,
    )


def test_react_mode_defaults_to_pileon():
    assert react_mode({}) == MODE_PILEON
    assert react_mode({"AMBIENT_REACT_MODE": "PICK"}) == MODE_PICK
    assert react_mode({"AMBIENT_REACT_MODE": "nonsense"}) == MODE_PILEON


def test_intents_include_guild_reactions():
    assert create_bot_intents().guild_reactions is True


# ── eligibility ──────────────────────────────────────────────────────────────


def test_joins_landing_reaction():
    assert pileon_block_reason(_message(count=2), _Emoji("💀"), _bot(), 2) is None


def test_block_reasons():
    bot = _bot()
    cases = [
        (None, _Emoji("💀"), REASON_UNAVAILABLE),
        (_message(author_id=BOT_ID), _Emoji("💀"), REASON_OWN_MESSAGE),
        (_message(emoji_text="🔥"), _Emoji("💀"), REASON_REACTION_GONE),
        (_message(me=True), _Emoji("💀"), REASON_ALREADY_REACTED),
        (_message(count=1), _Emoji("💀"), REASON_BELOW_MIN_COUNT),
        (_message(perms=False), _Emoji("💀"), REASON_NO_PERMISSION),
    ]
    for msg, emoji, expected in cases:
        assert pileon_block_reason(msg, emoji, bot, 2) == expected, expected


def test_custom_emoji_must_be_usable():
    msg = _message(emoji_text="<:kekw:5>")
    assert pileon_block_reason(msg, _Emoji("<:kekw:5>", 5), _bot(), 2) == REASON_FOREIGN_EMOJI
    assert pileon_block_reason(msg, _Emoji("<:kekw:5>", 5), _bot(known_emoji_ids=(5,)), 2) is None


# ── message lookup ───────────────────────────────────────────────────────────


async def test_resolve_prefers_cache():
    bot = _bot()
    cached = SimpleNamespace(id=7)
    bot.cached_messages = [cached]
    assert await resolve_message(bot, SimpleNamespace(message_id=7, channel_id=1)) is cached
    bot.get_channel.assert_not_called()


async def test_resolve_fetches_when_not_cached_and_swallows_errors():
    bot = _bot()
    channel = MagicMock()
    channel.fetch_message = AsyncMock(return_value="fetched")
    bot.get_channel.return_value = channel
    assert await resolve_message(bot, SimpleNamespace(message_id=7, channel_id=1)) == "fetched"
    channel.fetch_message = AsyncMock(side_effect=RuntimeError("404"))
    assert await resolve_message(bot, SimpleNamespace(message_id=7, channel_id=1)) is None


# ── router wiring ────────────────────────────────────────────────────────────


def _fake_router(config=None, bot=None):
    return SimpleNamespace(
        config=dict(_CONFIG) if config is None else config,
        bot=bot or _bot(),
        _ambient_react_cooldowns=AmbientCooldowns(),
        _ambient_react_tasks=set(),
        _metric_inc=MagicMock(),
        logger=MagicMock(),
        _run_pileon=AsyncMock(),
    )


def _payload(user_id: int = 1, member_bot: bool = False, guild_id: int | None = 10):
    return SimpleNamespace(
        guild_id=guild_id,
        channel_id=42,
        message_id=7,
        user_id=user_id,
        member=SimpleNamespace(bot=member_bot),
        emoji=_Emoji("💀"),
    )


async def test_router_spawns_pileon_for_human_reaction(monkeypatch):
    from bot.router import Router

    monkeypatch.setattr("bot.server_features.is_server_feature_enabled", lambda *_a: True)
    fake = _fake_router()
    Router.handle_reaction_add(fake, _payload())
    assert len(fake._ambient_react_tasks) == 1
    await next(iter(fake._ambient_react_tasks))
    fake._run_pileon.assert_awaited_once()


async def test_router_ignores_bot_reactions_and_pick_mode(monkeypatch):
    from bot.router import Router

    monkeypatch.setattr("bot.server_features.is_server_feature_enabled", lambda *_a: True)
    for fake, payload in (
        (_fake_router(), _payload(user_id=BOT_ID)),
        (_fake_router(), _payload(member_bot=True)),
        (_fake_router(), _payload(guild_id=None)),
        (_fake_router(config=dict(_CONFIG, AMBIENT_REACT_MODE="pick")), _payload()),
    ):
        Router.handle_reaction_add(fake, payload)
        assert not fake._ambient_react_tasks


async def test_run_pileon_copies_emoji_and_records_cooldown():
    from bot.router import Router

    msg = _message(count=2)
    msg.id = 7
    msg.add_reaction = AsyncMock()
    bot = _bot()
    bot.cached_messages = [msg]
    fake = _fake_router(bot=bot)
    payload = _payload()
    await Router._run_pileon(fake, payload)
    msg.add_reaction.assert_awaited_once_with(payload.emoji)
    assert fake._ambient_react_cooldowns.channel_elapsed(42, 0.0) < 0  # recorded at a monotonic time > 0


async def test_run_pileon_skips_single_reaction_without_burning_cooldown():
    from bot.router import Router

    msg = _message(count=1)
    msg.id = 7
    msg.add_reaction = AsyncMock()
    bot = _bot()
    bot.cached_messages = [msg]
    fake = _fake_router(bot=bot)
    await Router._run_pileon(fake, _payload())
    msg.add_reaction.assert_not_awaited()
    assert fake._ambient_react_cooldowns.global_elapsed(5.0) == 5.0
