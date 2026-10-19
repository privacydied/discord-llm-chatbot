"""Ambient pile-on: join a reaction other people already put on a message.

Triggered from ``on_raw_reaction_add``. When a message's emoji reaches the
configured count (default 2), the bot may add the same emoji. No model call:
it only ever copies a reaction humans chose, so it can't pick something off.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import discord

from bot.router_components.ambient_react import can_react

logger = logging.getLogger("bot.ambient")

MODE_PILEON = "pileon"
MODE_PICK = "pick"
DEFAULT_MIN_COUNT = 2
FETCH_TIMEOUT_S = 10.0

REASON_UNAVAILABLE = "message_unavailable"
REASON_OWN_MESSAGE = "own_message"
REASON_REACTION_GONE = "reaction_gone"
REASON_ALREADY_REACTED = "already_reacted"
REASON_BELOW_MIN_COUNT = "below_min_count"
REASON_FOREIGN_EMOJI = "foreign_emoji"
REASON_NO_PERMISSION = "no_permission"


def react_mode(config: dict[str, Any]) -> str:
    """AMBIENT_REACT_MODE: 'pileon' (default) copies existing reactions; 'pick' asks the model per message."""
    mode = str(config.get("AMBIENT_REACT_MODE") or MODE_PILEON).strip().lower()
    return mode if mode in (MODE_PILEON, MODE_PICK) else MODE_PILEON


async def resolve_message(bot: Any, payload: Any) -> Any | None:
    """Cached message first (no I/O); otherwise one bounded fetch."""
    cached = discord.utils.get(getattr(bot, "cached_messages", None) or [], id=payload.message_id)
    if cached is not None:
        return cached
    channel = bot.get_channel(payload.channel_id)
    if channel is None or not hasattr(channel, "fetch_message"):
        return None
    try:
        return await asyncio.wait_for(channel.fetch_message(payload.message_id), timeout=FETCH_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 — deleted, no access, or slow: skip the pile-on
        logger.debug("ambient.pileon fetch failed: %s", exc)
        return None


def matching_reaction(message: Any, emoji: Any) -> Any | None:
    key = str(emoji)
    return next((r for r in getattr(message, "reactions", None) or [] if str(r.emoji) == key), None)


def pileon_block_reason(message: Any, emoji: Any, bot: Any, min_count: int) -> str | None:
    """Why the bot must not join this reaction, or None if it may."""
    if message is None:
        return REASON_UNAVAILABLE
    if getattr(getattr(message, "author", None), "id", None) == getattr(bot.user, "id", None):
        return REASON_OWN_MESSAGE
    reaction = matching_reaction(message, emoji)
    if reaction is None:
        return REASON_REACTION_GONE
    if getattr(reaction, "me", False):
        return REASON_ALREADY_REACTED
    if getattr(reaction, "count", 0) < min_count:
        return REASON_BELOW_MIN_COUNT
    # Custom emoji from a server the bot isn't in can't be used. [IV]
    if getattr(emoji, "id", None) and bot.get_emoji(emoji.id) is None:
        return REASON_FOREIGN_EMOJI
    if not can_react(message):
        return REASON_NO_PERMISSION
    return None
