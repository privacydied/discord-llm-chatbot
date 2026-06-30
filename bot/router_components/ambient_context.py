"""Locally-scoped context builder for ambient (unprompted) replies.

When an ambient reply fires, the message was never addressed to the bot —
there is no mention, no reply-to-bot anchor. Left alone, `_flow_process_text`
falls through to the rolling per-channel conversation buffer
(`EnhancedContextManager`), which can hold an earlier, unrelated exchange
that hijacks the reply (the bot answers as if continuing that old topic).

This module builds a minimal, message-local alternative instead: the
triggering message (always), plus either its referenced/reply-to message
(preferred, when the trigger is itself a Discord reply) or up to a couple of
immediately-preceding channel messages — explicitly labeled as background,
never presented as the bot's own conversation history. [CA][REH]
"""

from __future__ import annotations

import asyncio

import discord

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Bounded, cheap lookback — this is background flavor, not history retrieval.
AMBIENT_BACKGROUND_LIMIT = 2
AMBIENT_FETCH_TIMEOUT_S = 5.0

_DISCORD_FETCH_ERRORS = (
    asyncio.TimeoutError,
    discord.NotFound,
    discord.Forbidden,
    discord.HTTPException,
    AttributeError,
    TypeError,
)


def _format_line(message: discord.Message) -> str:
    """Render a single message as `Author: text` for background context."""
    author = getattr(message.author, "display_name", None) or getattr(message.author, "name", None) or "Unknown"
    text = (message.content or "").strip() or "[no text content]"
    return f"{author}: {text}"


async def _resolve_referenced_message(message: discord.Message, timeout_s: float) -> discord.Message | None:
    """Resolve the message this one replies to, if any. Cache-first, fetch fallback."""
    ref = getattr(message, "reference", None)
    if ref is None:
        return None
    resolved = getattr(ref, "resolved", None)
    if isinstance(resolved, discord.Message):
        return resolved
    ref_id = getattr(ref, "message_id", None)
    if not ref_id:
        return None
    try:
        return await asyncio.wait_for(message.channel.fetch_message(ref_id), timeout=timeout_s)
    except _DISCORD_FETCH_ERRORS:
        return None


async def _collect_recent_background(message: discord.Message, limit: int, timeout_s: float) -> list[discord.Message]:
    """Fetch up to `limit` channel messages immediately preceding `message`."""
    if limit <= 0:
        return []

    async def _drain() -> list[discord.Message]:
        items: list[discord.Message] = []
        async for m in message.channel.history(limit=limit, before=message):
            items.append(m)
        return items

    try:
        return await asyncio.wait_for(_drain(), timeout=timeout_s)
    except _DISCORD_FETCH_ERRORS:
        return []


async def build_ambient_local_context(
    message: discord.Message,
    *,
    background_limit: int = AMBIENT_BACKGROUND_LIMIT,
    timeout_s: float = AMBIENT_FETCH_TIMEOUT_S,
) -> str:
    """Build a minimal, locally-scoped context block for an ambient reply.

    Never touches the rolling per-channel buffer. If the triggering message
    is a Discord reply, its referenced message is used as immediate context
    in preference to channel-order neighbors; otherwise a small bounded
    window of recent channel messages is included as labeled background.
    The triggering message itself is always included, clearly marked as the
    turn the model must respond to.
    """
    blocks: list[str] = []

    ref_msg = await _resolve_referenced_message(message, timeout_s)
    if ref_msg is not None:
        blocks.append("[Replying to — immediate context]\n" + _format_line(ref_msg))
    else:
        recent = await _collect_recent_background(message, background_limit, timeout_s)
        if recent:
            recent = list(reversed(recent))  # chronological order: oldest -> newest
            lines = "\n".join(_format_line(m) for m in recent)
            blocks.append(
                "[For context, recent messages in this channel — background only, NOT a conversation you are part of]\n" + lines,
            )

    blocks.append("[Message to respond to]\n" + _format_line(message))

    return "\n\n".join(blocks)
