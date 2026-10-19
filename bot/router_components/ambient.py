"""Ambient reply decision logic — cheap, synchronous, zero I/O.

Implements the gate that decides whether the bot should emit an unprompted
reply to a message it wasn't mentioned in.  Call ``should_ambient_reply``
from the router's silence-gate path.  All state lives in ``AmbientCooldowns``
which is instantiated once per router.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any, Protocol

from bot.utils.bounded_lru import BoundedDict


logger = logging.getLogger("bot.ambient")

# ── public sentinel for suppression reason ──────────────────────────────────
REASON_DISABLED = "disabled"
REASON_GUILD_FEATURE_OFF = "guild_feature_off"
REASON_BOT_AUTHOR = "bot_author"
REASON_SYSTEM_MESSAGE = "system_message"
REASON_COMMAND = "command"
REASON_MENTION = "mention"
REASON_TOO_SHORT = "too_short"
REASON_CHANNEL_NOT_ALLOWED = "channel_not_allowed"
REASON_QUIET_HOURS = "quiet_hours"
REASON_GLOBAL_COOLDOWN = "global_cooldown"
REASON_CHANNEL_COOLDOWN = "channel_cooldown"
REASON_PROBABILITY = "probability"


class _Rng(Protocol):
    def random(self) -> float: ...


class AmbientCooldowns:
    """In-memory cooldown state for the ambient-reply feature.

    Bounded dicts cap memory: at most 512 channels tracked, global state
    is a single float.  Thread-unsafe but the router runs on a single
    asyncio event loop thread, so this is fine.
    """

    _CHANNEL_MAXSIZE = 512

    def __init__(self) -> None:
        self._channel_last: BoundedDict[int, float] = BoundedDict(self._CHANNEL_MAXSIZE)
        self._global_last: float = 0.0

    def channel_elapsed(self, channel_id: int, now: float) -> float:
        last = self._channel_last.get(channel_id, 0.0)
        return now - last

    def global_elapsed(self, now: float) -> float:
        return now - self._global_last

    def record(self, channel_id: int, now: float) -> None:
        self._channel_last[channel_id] = now
        self._global_last = now


def _parse_quiet_hours(spec: str) -> tuple[int, int] | None:
    """Parse "HH-HH" UTC quiet-hour range.  Returns (start, end) or None."""
    spec = (spec or "").strip()
    if not spec:
        return None
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", spec)
    if not m:
        logger.warning("ambient QUIET_HOURS invalid format %r (expected HH-HH); ignoring", spec)
        return None
    start, end = int(m.group(1)), int(m.group(2))
    if not (0 <= start <= 23 and 0 <= end <= 23):
        logger.warning("ambient QUIET_HOURS out of range %r; ignoring", spec)
        return None
    return start, end


def _in_quiet_hours(now: float, spec: str) -> bool:
    parsed = _parse_quiet_hours(spec)
    if parsed is None:
        return False
    import datetime

    hour = datetime.datetime.utcfromtimestamp(now).hour
    start, end = parsed
    if start <= end:
        return start <= hour < end
    # wraps midnight: e.g. 23-7 means 23,0,1,2,3,4,5,6
    return hour >= start or hour < end


def _parse_channel_allowlist(spec: str) -> set[int] | None:
    """Parse "id1,id2,..." channel allowlist.  Returns set or None (= all allowed)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    ids: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.isdigit():
            ids.add(int(tok))
        elif tok:
            logger.warning("ambient CHANNELS bad token %r; skipping", tok)
    return ids or None


# Per-kind config defaults. Keys are suffixes of "<PREFIX>_<KEY>". [CMV]
REPLY_PREFIX = "AMBIENT_REPLY"
REACT_PREFIX = "AMBIENT_REACT"
_GATE_DEFAULTS: dict[str, dict[str, Any]] = {
    REPLY_PREFIX: {
        "PROBABILITY": 0.002,
        "MIN_CHARS": 12,
        "CHANNEL_COOLDOWN_S": 1800,
        "GLOBAL_COOLDOWN_S": 600,
    },
    REACT_PREFIX: {
        "PROBABILITY": 0.03,
        "MIN_CHARS": 12,
        "CHANNEL_COOLDOWN_S": 300,
        "GLOBAL_COOLDOWN_S": 60,
    },
}


def _author_gate(message: Any, config: dict[str, Any]) -> str | None:
    """Return a suppression reason for ineligible authors/content, else None."""
    author = getattr(message, "author", None)
    if author is None or getattr(author, "bot", False):
        return REASON_BOT_AUTHOR
    msg_type = getattr(message, "type", None)
    # discord.MessageType.default == 0; anything else = system message
    if msg_type is not None and getattr(msg_type, "value", msg_type) != 0:
        return REASON_SYSTEM_MESSAGE
    content: str = (getattr(message, "content", "") or "").strip()
    if content.startswith(config.get("COMMAND_PREFIX", "!")):
        return REASON_COMMAND
    # Bot mentions route normally via the main gate.
    if any(getattr(u, "bot", False) for u in (getattr(message, "mentions", []) or [])):
        return REASON_MENTION
    return None


def _ambient_gate(
    message: Any,
    config: dict[str, Any],
    cooldowns: AmbientCooldowns,
    guild_feature_enabled: bool,
    prefix: str,
    now: float,
    rng: _Rng,
) -> tuple[bool, str]:
    """Shared ambient gate; ``prefix`` selects the AMBIENT_REPLY_* or AMBIENT_REACT_* keys."""
    defaults = _GATE_DEFAULTS[prefix]

    def cfg(key: str) -> Any:
        return config.get(f"{prefix}_{key}", defaults.get(key))

    if not config.get(f"{prefix}_ENABLED", False):
        return False, REASON_DISABLED
    if not guild_feature_enabled:
        return False, REASON_GUILD_FEATURE_OFF
    reason = _author_gate(message, config)
    if reason is not None:
        return False, reason
    content: str = (getattr(message, "content", "") or "").strip()
    if len(content) < cfg("MIN_CHARS"):
        return False, REASON_TOO_SHORT
    channel_id: int = getattr(getattr(message, "channel", None), "id", 0) or 0
    allowlist = _parse_channel_allowlist(cfg("CHANNELS") or "")
    if allowlist is not None and channel_id not in allowlist:
        return False, REASON_CHANNEL_NOT_ALLOWED
    if _in_quiet_hours(now, cfg("QUIET_HOURS") or ""):
        return False, REASON_QUIET_HOURS
    if cooldowns.global_elapsed(now) < cfg("GLOBAL_COOLDOWN_S"):
        return False, REASON_GLOBAL_COOLDOWN
    if cooldowns.channel_elapsed(channel_id, now) < cfg("CHANNEL_COOLDOWN_S"):
        return False, REASON_CHANNEL_COOLDOWN
    if rng.random() >= cfg("PROBABILITY"):
        return False, REASON_PROBABILITY
    cooldowns.record(channel_id, now)
    return True, "fired"


def should_ambient_reply(
    message: Any,
    config: dict[str, Any],
    cooldowns: AmbientCooldowns,
    guild_feature_enabled: bool,
    now: float | None = None,
    rng: _Rng | None = None,
) -> tuple[bool, str]:
    """Return (fire, reason) for an unprompted ambient reply.

    ``reason`` is a short string used for logging/metrics regardless of outcome.
    All checks are O(1) and I/O-free.
    """
    return _ambient_gate(
        message,
        config,
        cooldowns,
        guild_feature_enabled,
        REPLY_PREFIX,
        time.monotonic() if now is None else now,
        rng or random,
    )


def should_ambient_react(
    message: Any,
    config: dict[str, Any],
    cooldowns: AmbientCooldowns,
    guild_feature_enabled: bool,
    now: float | None = None,
    rng: _Rng | None = None,
) -> tuple[bool, str]:
    """Return (fire, reason) for an ambient emoji reaction. Same gate, AMBIENT_REACT_* keys."""
    return _ambient_gate(
        message,
        config,
        cooldowns,
        guild_feature_enabled,
        REACT_PREFIX,
        time.monotonic() if now is None else now,
        rng or random,
    )
