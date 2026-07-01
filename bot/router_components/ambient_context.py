"""Locally-scoped context builder for ambient (unprompted) replies.

When an ambient reply fires, the message was never addressed to the bot —
there is no mention, no reply-to-bot anchor. Left alone, `_flow_process_text`
falls through to the rolling per-channel conversation buffer
(`EnhancedContextManager`), which can hold an earlier, unrelated exchange
that hijacks the reply (the bot answers as if continuing that old topic).

This module builds a locally-scoped alternative instead: the triggering
message (always, clearly marked as the thing to respond to), its
referenced/reply-to message when the trigger is itself a Discord reply
(labeled as immediate context), and a bounded window of recent channel
messages fetched directly from Discord (`channel.history`) — never the
rolling buffer — labeled as observed background. The bot should SEE what the
channel is actually discussing so context-dependent triggers ("on second
thought", a bare reaction) are grounded, but the background block never
relabels prior turns (including the bot's own) as the model's active
conversation to continue; they're shown as one participant among others
inside a clearly-marked "you are joining an ongoing conversation" block.
[CA][REH]
"""

from __future__ import annotations

import asyncio

import discord

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Default background-history depth when the caller doesn't pass one
# explicitly. Callers (Router) pass the configured depth
# (AMBIENT_REPLY_CONTEXT_MESSAGES, itself defaulting to MAX_CONTEXT_MESSAGES)
# so this is only a defensive fallback. [CMV]
AMBIENT_BACKGROUND_LIMIT = 10
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


def _dedupe_against_reference(
    recent: list[discord.Message],
    ref_msg: discord.Message | None,
) -> list[discord.Message]:
    """Drop the resolved reference from the background window if it also
    landed in the recent-history fetch, so it isn't shown twice."""
    if ref_msg is None:
        return recent
    ref_id = getattr(ref_msg, "id", None)
    return [m for m in recent if getattr(m, "id", None) != ref_id]


def _format_background_block(recent: list[discord.Message]) -> str:
    """Render the background block: oldest -> newest, clearly labeled as
    observed context the model is joining, not its own conversation."""
    chronological = list(reversed(recent))  # Discord history() is newest-first
    lines = "\n".join(_format_line(m) for m in chronological)
    return "[Recent channel messages — you are joining an ongoing conversation, not continuing your own turns]\n" + lines


async def build_ambient_local_context(
    message: discord.Message,
    *,
    background_limit: int = AMBIENT_BACKGROUND_LIMIT,
    timeout_s: float = AMBIENT_FETCH_TIMEOUT_S,
) -> str:
    """Build a locally-scoped context block for an ambient reply.

    Never touches the rolling per-channel buffer (`EnhancedContextManager`).
    Always includes a bounded window of recent channel messages — sized by
    `background_limit` (callers pass the configured depth) — as clearly
    labeled background, so context-dependent triggers stay grounded in what
    the channel is actually discussing. When the triggering message is
    itself a Discord reply, its resolved reference is additionally included
    as a distinct, more specific immediate-context block. The triggering
    message itself is always included last, clearly marked as the turn the
    model must respond to.
    """
    blocks: list[str] = []

    ref_msg = await _resolve_referenced_message(message, timeout_s)
    if ref_msg is not None:
        blocks.append("[Replying to — immediate context]\n" + _format_line(ref_msg))

    recent = await _collect_recent_background(message, background_limit, timeout_s)
    recent = _dedupe_against_reference(recent, ref_msg)
    if recent:
        blocks.append(_format_background_block(recent))

    blocks.append("[Message to respond to]\n" + _format_line(message))

    return "\n\n".join(blocks)
