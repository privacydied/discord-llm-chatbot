"""Tests for bot.router_components.ambient_context — deterministic, no I/O, no live backend.

Covers the fix for: ambient replies inheriting unrelated channel context. The
builder here must never touch the rolling per-channel buffer; it only looks
at the triggering message, its reply target (if any), and a small bounded
window of immediately-preceding channel messages.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.router_components.ambient_context import (
    AMBIENT_BACKGROUND_LIMIT,
    build_ambient_local_context,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _author(name: str) -> MagicMock:
    author = MagicMock()
    author.display_name = name
    author.name = name
    return author


def _msg(content: str, author_name: str = "Trigger", reference=None, msg_id: int = 1) -> MagicMock:
    msg = MagicMock(spec=discord.Message)
    msg.id = msg_id
    msg.content = content
    msg.author = _author(author_name)
    msg.reference = reference
    msg.channel = MagicMock()
    return msg


def _channel_with_history(history_messages: list) -> MagicMock:
    """Channel mock whose `.history(limit=, before=)` yields the given messages."""
    channel = MagicMock()

    def _history(limit=None, before=None):
        async def _gen():
            for m in history_messages[: (limit or len(history_messages))]:
                yield m

        return _gen()

    channel.history = MagicMock(side_effect=_history)
    return channel


# ── no reference: background window ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_lone_message_centers_trigger_with_recent_background() -> None:
    """Topic-B trigger with topic-A-flavored recent channel messages: trigger
    must be present and clearly marked; background must be clearly labeled
    (never presented as the bot's own conversation history)."""
    trigger = _msg("topic B: what about pizza?", author_name="Bob", msg_id=100)
    background = [
        _msg("topic A: let's talk about cars", author_name="Alice", msg_id=98),
        _msg("topic A: I like sedans", author_name="Carl", msg_id=99),
    ]
    trigger.channel = _channel_with_history(list(reversed(background)))  # newest-first, as Discord returns

    result = await build_ambient_local_context(trigger)

    assert "[Message to respond to]" in result
    assert "topic B: what about pizza?" in result
    assert "[For context, recent messages in this channel" in result
    assert "NOT a conversation you are part of" in result
    assert "topic A: let's talk about cars" in result
    assert "topic A: I like sedans" in result
    # Background must read oldest -> newest
    assert result.index("let's talk about cars") < result.index("I like sedans")
    # Background block must come before the trigger block
    assert result.index("[For context") < result.index("[Message to respond to]")


@pytest.mark.asyncio
async def test_lone_message_no_history_available_still_centers_trigger() -> None:
    trigger = _msg("hello channel", msg_id=200)
    trigger.channel = _channel_with_history([])

    result = await build_ambient_local_context(trigger)

    assert result == "[Message to respond to]\nTrigger: hello channel"


@pytest.mark.asyncio
async def test_background_window_is_bounded() -> None:
    trigger = _msg("trigger text", msg_id=300)
    many = [_msg(f"msg {i}", author_name=f"User{i}", msg_id=i) for i in range(10)]
    trigger.channel = _channel_with_history(many)

    result = await build_ambient_local_context(trigger, background_limit=AMBIENT_BACKGROUND_LIMIT)

    # Only AMBIENT_BACKGROUND_LIMIT background lines should appear (plus the trigger line)
    background_lines = [ln for ln in result.splitlines() if ln.startswith("User")]
    assert len(background_lines) == AMBIENT_BACKGROUND_LIMIT


# ── reference present: referenced message preferred over background ────────


@pytest.mark.asyncio
async def test_reply_trigger_uses_resolved_reference_as_immediate_context() -> None:
    referenced = _msg("the original thing being replied to", author_name="OriginalAuthor", msg_id=50)
    ref = MagicMock()
    ref.resolved = referenced
    ref.message_id = referenced.id
    trigger = _msg("ambient reply to a reply chain", author_name="Bob", reference=ref, msg_id=51)
    # Background history should NOT be consulted when a reference resolves
    trigger.channel = _channel_with_history([_msg("unrelated background", msg_id=49)])

    result = await build_ambient_local_context(trigger)

    assert "[Replying to — immediate context]" in result
    assert "the original thing being replied to" in result
    assert "[Message to respond to]" in result
    assert "ambient reply to a reply chain" in result
    # Channel-order neighbors must not be pulled in when a reference is available
    assert "unrelated background" not in result
    trigger.channel.history.assert_not_called()


@pytest.mark.asyncio
async def test_reply_trigger_fetches_unresolved_reference() -> None:
    referenced = _msg("fetched reference text", author_name="OriginalAuthor", msg_id=60)
    ref = MagicMock()
    ref.resolved = None
    ref.message_id = referenced.id
    trigger = _msg("ambient reply", reference=ref, msg_id=61)
    trigger.channel = MagicMock()
    trigger.channel.fetch_message = AsyncMock(return_value=referenced)
    trigger.channel.history = MagicMock(side_effect=AssertionError("history should not be called"))

    result = await build_ambient_local_context(trigger)

    trigger.channel.fetch_message.assert_awaited_once_with(referenced.id)
    assert "fetched reference text" in result


@pytest.mark.asyncio
async def test_reply_trigger_falls_back_to_background_when_fetch_fails() -> None:
    ref = MagicMock()
    ref.resolved = None
    ref.message_id = 999
    trigger = _msg("ambient reply, broken reference", reference=ref, msg_id=62)
    background = [_msg("recent unrelated chatter", msg_id=61)]
    trigger.channel = _channel_with_history(background)
    trigger.channel.fetch_message = AsyncMock(side_effect=discord.NotFound(MagicMock(), "missing"))

    result = await build_ambient_local_context(trigger)

    assert "[Replying to" not in result
    assert "recent unrelated chatter" in result
    assert "[Message to respond to]" in result


@pytest.mark.asyncio
async def test_history_fetch_errors_are_swallowed() -> None:
    trigger = _msg("trigger with broken history", msg_id=70)
    channel = MagicMock()

    def _raise(limit=None, before=None):
        raise discord.HTTPException(MagicMock(), "boom")

    channel.history = MagicMock(side_effect=_raise)
    trigger.channel = channel

    result = await build_ambient_local_context(trigger)

    assert result == "[Message to respond to]\nTrigger: trigger with broken history"
