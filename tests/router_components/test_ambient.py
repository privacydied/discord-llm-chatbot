"""Tests for bot.router_components.ambient — deterministic, no I/O, no real RNG."""

from __future__ import annotations

from unittest.mock import MagicMock

from bot.router_components.ambient import (
    REASON_BOT_AUTHOR,
    REASON_CHANNEL_COOLDOWN,
    REASON_CHANNEL_NOT_ALLOWED,
    REASON_COMMAND,
    REASON_DISABLED,
    REASON_GLOBAL_COOLDOWN,
    REASON_GUILD_FEATURE_OFF,
    REASON_MENTION,
    REASON_PROBABILITY,
    REASON_QUIET_HOURS,
    REASON_SYSTEM_MESSAGE,
    REASON_TOO_SHORT,
    AmbientCooldowns,
    _in_quiet_hours,
    _parse_channel_allowlist,
    _parse_quiet_hours,
    should_ambient_reply,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_BASE_CONFIG = {
    "AMBIENT_REPLY_ENABLED": True,
    "AMBIENT_REPLY_PROBABILITY": 1.0,  # always fire by default in tests
    "AMBIENT_REPLY_MIN_CHARS": 12,
    "AMBIENT_REPLY_CHANNEL_COOLDOWN_S": 0,
    "AMBIENT_REPLY_GLOBAL_COOLDOWN_S": 0,
    "AMBIENT_REPLY_CHANNELS": "",
    "AMBIENT_REPLY_QUIET_HOURS": "",
    "COMMAND_PREFIX": "!",
}


def _msg(
    content: str = "hello from the other side",
    bot_author: bool = False,
    msg_type_value: int = 0,
    channel_id: int = 111,
    has_bot_mention: bool = False,
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    author = MagicMock()
    author.bot = bot_author
    msg.author = author

    msg_type = MagicMock()
    msg_type.value = msg_type_value
    msg.type = msg_type

    channel = MagicMock()
    channel.id = channel_id
    msg.channel = channel

    if has_bot_mention:
        bot_user = MagicMock()
        bot_user.bot = True
        msg.mentions = [bot_user]
    else:
        msg.mentions = []

    return msg


class _FixedRng:
    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        return self._value


_FIRE_RNG = _FixedRng(0.0)  # always fires
_MISS_RNG = _FixedRng(1.0)  # never fires


def _decide(msg=None, config=None, cooldowns=None, guild_feat=True, now=1000.0, rng=None):
    if msg is None:
        msg = _msg()
    if config is None:
        config = dict(_BASE_CONFIG)
    if cooldowns is None:
        cooldowns = AmbientCooldowns()
    if rng is None:
        rng = _FIRE_RNG
    return should_ambient_reply(msg, config, cooldowns, guild_feat, now=now, rng=rng)


# ── feature flags ─────────────────────────────────────────────────────────────


def test_disabled_globally():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_ENABLED": False}
    fire, reason = _decide(config=cfg)
    assert not fire
    assert reason == REASON_DISABLED


def test_guild_feature_off():
    fire, reason = _decide(guild_feat=False)
    assert not fire
    assert reason == REASON_GUILD_FEATURE_OFF


# ── author eligibility ────────────────────────────────────────────────────────


def test_bot_author_suppressed():
    fire, reason = _decide(msg=_msg(bot_author=True))
    assert not fire
    assert reason == REASON_BOT_AUTHOR


def test_system_message_suppressed():
    fire, reason = _decide(msg=_msg(msg_type_value=7))
    assert not fire
    assert reason == REASON_SYSTEM_MESSAGE


# ── content filters ───────────────────────────────────────────────────────────


def test_command_suppressed():
    fire, reason = _decide(msg=_msg(content="!status"))
    assert not fire
    assert reason == REASON_COMMAND


def test_mention_suppressed():
    fire, reason = _decide(msg=_msg(has_bot_mention=True))
    assert not fire
    assert reason == REASON_MENTION


def test_too_short_suppressed():
    fire, reason = _decide(msg=_msg(content="lol"))
    assert not fire
    assert reason == REASON_TOO_SHORT


def test_exact_min_chars_passes():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_MIN_CHARS": 5}
    fire, reason = _decide(msg=_msg(content="hello"), config=cfg)
    assert fire
    assert reason == "fired"


# ── channel allowlist ─────────────────────────────────────────────────────────


def test_channel_not_in_allowlist():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_CHANNELS": "999,888"}
    fire, reason = _decide(msg=_msg(channel_id=111), config=cfg)
    assert not fire
    assert reason == REASON_CHANNEL_NOT_ALLOWED


def test_channel_in_allowlist_passes():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_CHANNELS": "111,222"}
    fire, reason = _decide(msg=_msg(channel_id=111), config=cfg)
    assert fire


def test_empty_allowlist_allows_all():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_CHANNELS": ""}
    fire, _ = _decide(msg=_msg(channel_id=42), config=cfg)
    assert fire


# ── quiet hours ───────────────────────────────────────────────────────────────


def test_quiet_hours_simple():
    # 23:00–07:00 UTC; pick epoch at 00:00 UTC
    # epoch 0 = 1970-01-01T00:00:00Z → hour 0 → inside quiet window
    fire, reason = _decide(
        config={**_BASE_CONFIG, "AMBIENT_REPLY_QUIET_HOURS": "23-7"},
        now=0.0,
    )
    assert not fire
    assert reason == REASON_QUIET_HOURS


def test_quiet_hours_outside_range():
    # Hour 12 UTC is outside 23-7
    import datetime

    noon_utc = datetime.datetime(2024, 1, 1, 12, 0, 0).timestamp()
    fire, reason = _decide(
        config={**_BASE_CONFIG, "AMBIENT_REPLY_QUIET_HOURS": "23-7"},
        now=noon_utc,
    )
    assert fire


def test_quiet_hours_disabled_when_empty():
    assert not _in_quiet_hours(1000.0, "")
    assert not _in_quiet_hours(1000.0, "  ")


def test_quiet_hours_invalid_format_ignored():
    # bad format → treated as "no quiet hours" → should not suppress
    fire, _ = _decide(
        config={**_BASE_CONFIG, "AMBIENT_REPLY_QUIET_HOURS": "bad"},
    )
    assert fire


# ── cooldowns ─────────────────────────────────────────────────────────────────


def test_global_cooldown_suppresses():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_GLOBAL_COOLDOWN_S": 300}
    cd = AmbientCooldowns()
    cd.record(111, 900.0)  # fired at t=900
    fire, reason = _decide(config=cfg, cooldowns=cd, now=1000.0)  # only 100s later
    assert not fire
    assert reason == REASON_GLOBAL_COOLDOWN


def test_global_cooldown_expires():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_GLOBAL_COOLDOWN_S": 300}
    cd = AmbientCooldowns()
    cd.record(111, 600.0)  # fired at t=600
    fire, _ = _decide(config=cfg, cooldowns=cd, now=1000.0)  # 400s later
    assert fire


def test_channel_cooldown_suppresses():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_CHANNEL_COOLDOWN_S": 1800, "AMBIENT_REPLY_GLOBAL_COOLDOWN_S": 0}
    cd = AmbientCooldowns()
    cd._channel_last[111] = 500.0  # direct injection, bypasses global
    fire, reason = _decide(config=cfg, cooldowns=cd, now=1000.0)
    assert not fire
    assert reason == REASON_CHANNEL_COOLDOWN


def test_channel_cooldown_different_channels():
    """Cooldown on channel A should not suppress channel B."""
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_CHANNEL_COOLDOWN_S": 1800, "AMBIENT_REPLY_GLOBAL_COOLDOWN_S": 0}
    cd = AmbientCooldowns()
    # Both channels last fired at t=0; now=2000 → 2000s elapsed > 1800s cooldown
    # Explicitly note channel 111 fired recently (t=1500) but channel 222 did not
    cd._channel_last[111] = 1500.0  # channel 111 is cooling down (only 500s ago)
    # channel 222 defaults to 0.0 → 2000s elapsed → clears cooldown
    fire, _ = _decide(config=cfg, cooldowns=cd, now=2000.0, msg=_msg(channel_id=222))
    assert fire


def test_cooldowns_updated_on_fire():
    cd = AmbientCooldowns()
    now = 5000.0
    fire, _ = _decide(cooldowns=cd, now=now)
    assert fire
    assert cd._global_last == now
    assert cd._channel_last.get(111) == now


# ── probability ───────────────────────────────────────────────────────────────


def test_probability_zero_always_fires():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_PROBABILITY": 1.0}
    fire, _ = _decide(config=cfg, rng=_FixedRng(0.0))
    assert fire


def test_probability_one_never_fires():
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_PROBABILITY": 0.5}
    fire, reason = _decide(config=cfg, rng=_FixedRng(1.0))
    assert not fire
    assert reason == REASON_PROBABILITY


def test_probability_boundary():
    # prob=0.5, rng=0.49 → fires; rng=0.50 → suppressed
    cfg = {**_BASE_CONFIG, "AMBIENT_REPLY_PROBABILITY": 0.5}
    fire_yes, _ = _decide(config=cfg, rng=_FixedRng(0.49))
    fire_no, reason = _decide(config=cfg, rng=_FixedRng(0.50))
    assert fire_yes
    assert not fire_no
    assert reason == REASON_PROBABILITY


# ── parse helpers ─────────────────────────────────────────────────────────────


def test_parse_quiet_hours_valid():
    assert _parse_quiet_hours("23-7") == (23, 7)
    assert _parse_quiet_hours("0-8") == (0, 8)


def test_parse_quiet_hours_invalid():
    assert _parse_quiet_hours("") is None
    assert _parse_quiet_hours("bad") is None
    assert _parse_quiet_hours("25-7") is None


def test_parse_channel_allowlist():
    assert _parse_channel_allowlist("") is None
    assert _parse_channel_allowlist("111,222") == {111, 222}
    assert _parse_channel_allowlist("  333  ") == {333}


# ── BoundedDict capacity ─────────────────────────────────────────────────────


def test_bounded_dict_evicts_oldest():
    from bot.utils.bounded_lru import BoundedDict

    d: BoundedDict[int, int] = BoundedDict(maxsize=3)
    d[1] = 10
    d[2] = 20
    d[3] = 30
    d[4] = 40  # should evict key 1
    assert 1 not in d
    assert 4 in d
    assert len(d) == 3
