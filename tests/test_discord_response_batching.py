from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.action import BotAction
from bot.core.bot import _DISCORD_MAX_CONTENT_LEN, LLMBot


class _FakeSentMessage:
    def __init__(self, content: str, embeds=None, files=None) -> None:
        self.content = content
        self.embeds = embeds or []
        self.files = files or []
        self.reply = AsyncMock()
        self.id = id(self)


class _MessageProcessorMock:
    """Simulates MessageProcessor.on_message filtering: returns False for
    bot/self authors, True for legitimate user messages.
    """

    def __init__(self, bot_stub) -> None:
        self._bot_stub = bot_stub
        self.enqueue = AsyncMock()

    async def on_message(self, message) -> bool:
        author = getattr(message, "author", None)
        if author is None:
            return True
        author_is_bot = bool(getattr(author, "bot", False))
        try:
            author_is_self = getattr(author, "id", None) == getattr(self._bot_stub.user, "id", None)
        except Exception:
            author_is_self = False
        if author_is_bot or author_is_self:
            return False  # Drop bot/self messages
        return True


@pytest.fixture
def bot_stub():
    bot = MagicMock(spec=LLMBot)
    bot.user = MagicMock(id=99999, bot=True)
    bot.config = {}
    bot.logger = MagicMock()
    bot.enhanced_context_manager = None
    # MessageProcessor handles bot/self author filtering + dedup + queue
    bot.message_processor = _MessageProcessorMock(bot)

    async def _call_with_discord_retry(_name, factory, **_kwargs):
        return await factory()

    bot._call_with_discord_retry = _call_with_discord_retry
    return bot


# ===================== chunking: basic behavior =====================


@pytest.mark.asyncio
async def test_chunk_short_text_returns_single_chunk_unchanged(bot_stub) -> None:
    text = "short response"

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert chunks == [text]


@pytest.mark.asyncio
async def test_chunk_text_just_over_limit_returns_two_chunks(bot_stub) -> None:
    text = "x" * (_DISCORD_MAX_CONTENT_LEN + 1)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 2
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_prefers_paragraph_boundaries(bot_stub) -> None:
    text = ("a" * 1880) + "\n\n" + ("b" * 80)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 2
    assert chunks[0].endswith("\n\n")
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_prefers_newline_boundaries(bot_stub) -> None:
    # Use length > 1950 so it actually requires splitting.
    text = ("a" * 1940) + "\n" + ("b" * 50)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 2
    assert chunks[0].endswith("\n")
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_prefers_simple_sentence_boundaries(bot_stub) -> None:
    # Use length > 1950 so it actually requires splitting.
    text = ("a" * 1945) + ". " + ("b" * 50)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 2
    assert chunks[0].endswith(". ")
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_hard_splits_long_unbroken_text(bot_stub) -> None:
    text = "z" * ((_DISCORD_MAX_CONTENT_LEN * 2) + 17)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 3
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_does_not_mangle_code_block_worse_than_current(bot_stub) -> None:
    # We allow splits inside code fences if necessary, but we must:
    # - preserve content (no loss)
    # - not introduce random extra backticks or markers.
    code = "print('hello')\n"
    text = "intro\n\n```python\n" + code * 400 + "\n```\n"

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    joined = "".join(chunks)
    assert joined == text, "Chunking must preserve full content"
    assert all(len(c) <= _DISCORD_MAX_CONTENT_LEN for c in chunks)
    # Ensure no extra code fences are invented:
    assert joined.count("```") == text.count("```"), "No extra code fences introduced by chunking"


# ===================== send_chunked_reply: behavior =====================


@pytest.mark.asyncio
async def test_send_chunked_short_message_sends_once(bot_stub) -> None:
    # Single-chunk case via _send_chunked_reply.
    content = "short reply"
    action = BotAction(content=content, embeds=None)

    message = MagicMock(spec=discord.Message)
    message.id = 123
    message.content = "hi"
    message.reference = None
    message.guild = MagicMock(id=456)
    message.author = MagicMock(id=111, bot=False)
    message.channel = MagicMock(spec=discord.TextChannel)
    message.channel.id = 789

    sent_chunks = []

    async def _reply_side_effect(**kwargs):
        chunk = kwargs.get("content", "")
        sent_chunks.append(chunk)
        return _FakeSentMessage(content=chunk)

    async def _channel_send_side_effect(**kwargs):
        # For single-chunk case, first part uses reply, but if channel.send is called
        # (e.g., no reply_target), treat it similarly.
        chunk = kwargs.get("content", "")
        sent_chunks.append(chunk)
        return _FakeSentMessage(content=chunk)

    reply_target = MagicMock(spec=discord.Message)
    reply_target.reply = AsyncMock(side_effect=_reply_side_effect)
    message.channel.send = AsyncMock(side_effect=_channel_send_side_effect)
    reply_target.channel = message.channel

    await LLMBot._send_chunked_reply(
        bot_stub,
        message=message,
        action=action,
        base_extra={"msg_id": message.id},
        force_reply_target=reply_target,
        target_message=None,
        dispatch_meta={"trigger_message_id": message.id},
        content=content,
        files=None,
        chunks=[content],
    )

    # Short message: only one send, no continuations.
    assert len(sent_chunks) == 1

    # _send_chunked_reply may prepend a user mention when using reply_target;
    # ensure our logical content is included.
    assert content in sent_chunks[0]


@pytest.mark.asyncio
async def test_send_chunked_long_message_is_sequential_and_ordered(bot_stub) -> None:
    content = "x" * 3500
    action = BotAction(content=content, embeds=[discord.Embed(title="alpha")])
    files = [MagicMock(name="file")]

    message = MagicMock(spec=discord.Message)
    message.id = 123
    message.content = "hi"
    message.reference = None
    message.guild = MagicMock(id=456)
    message.author = MagicMock(id=111, bot=False, mention="<@111>")
    message.channel = MagicMock(spec=discord.TextChannel)
    message.channel.id = 789

    reply_calls = []
    channel_send_calls = []
    sent_messages = []

    async def _reply_side_effect(**kwargs):
        reply_calls.append(kwargs)
        sent = _FakeSentMessage(
            content=kwargs.get("content", ""),
            embeds=kwargs.get("embeds") or [],
            files=kwargs.get("files") or [],
        )
        sent_messages.append(sent)
        return sent

    async def _channel_send_side_effect(**kwargs):
        channel_send_calls.append(kwargs)
        sent = _FakeSentMessage(
            content=kwargs.get("content", ""),
            embeds=kwargs.get("embeds") or [],
            files=kwargs.get("files") or [],
        )
        sent_messages.append(sent)
        return sent

    reply_target = MagicMock(spec=discord.Message)
    reply_target.reply = AsyncMock(side_effect=_reply_side_effect)
    message.channel.send = AsyncMock(side_effect=_channel_send_side_effect)
    reply_target.channel = message.channel

    chunks = LLMBot._chunk_message_content(bot_stub, content)
    result = await LLMBot._send_chunked_reply(
        bot_stub,
        message=message,
        action=action,
        base_extra={"msg_id": message.id},
        force_reply_target=reply_target,
        target_message=None,
        dispatch_meta={"trigger_message_id": message.id},
        content=content,
        files=files,
        chunks=chunks,
    )

    # Basic invariants:
    assert result is sent_messages[-1]

    # Extract all sent contents (may include mention prefixes for first chunk).
    all_contents = [c["content"] for c in reply_calls + channel_send_calls]

    # Ensure our logical content is fully preserved inside the concatenation.
    joined = "".join(all_contents)
    assert content in joined, "Full logical content must be preserved (ignoring mentions)"

    # First chunk may include a user mention prefix, so only strictly enforce
    # length limit on continuation chunks.
    (channel_send_calls[0]["content"] if channel_send_calls else "")
    for c in all_contents[1:]:
        assert len(c) <= _DISCORD_MAX_CONTENT_LEN, "Continuation chunk exceeds Discord limit"

    # First chunk: uses reply() with embeds/files.
    assert len(reply_calls) == 1, "First chunk should use reply()"
    assert reply_calls[0]["embeds"] == action.embeds
    assert reply_calls[0]["files"] == files

    # Continuation chunks: use channel.send(), not reply().
    assert len(channel_send_calls) >= 1, "Continuations must use channel.send()"
    for call in channel_send_calls:
        # No embeds/files on continuations (by current policy)
        assert call.get("embeds") == []
        assert call.get("files") is None

    # Ensure ordering: all reply() calls before channel.send() calls.
    # Since we only use reply() once for first chunk, just confirm that
    # continuation chunks are not also using reply().
    for c in channel_send_calls:
        assert "reply" not in str(c), "Continuation should not be a reply call"

    # Continuation messages should not reply to bot's previous chunks.
    sent_messages[0].reply.assert_not_awaited()


# ===================== chunking: delivery order =====================


@pytest.mark.asyncio
async def test_send_chunked_parts_are_sent_in_order_one_at_a_time(bot_stub) -> None:
    """Parts must leave in chunk order, and each send must complete before the next starts.

    The older ordering test inspects `reply_calls + channel_send_calls`, which is
    ordered by *call type* rather than by time -- it cannot observe a reordering.
    This test records one shared timeline across both APIs, and makes each send
    await a real event-loop turn so any concurrency (gather/create_task) would
    interleave and be caught. Discord assigns snowflakes in receipt order, so a
    later-but-shorter part overtaking an earlier one is only possible if the sends
    are in flight together.
    """
    # Deliberately lopsided: a long first part and a tiny trailing part, the exact
    # shape where a short chunk could overtake a long one if sends raced.
    content = ("A" * 3000) + "\n\ntail"
    action = BotAction(content=content, embeds=[])

    message = MagicMock(spec=discord.Message)
    message.id = 123
    message.content = "hi"
    message.reference = None
    message.guild = MagicMock(id=456)
    message.author = MagicMock(id=111, bot=False, mention="<@111>")
    message.channel = MagicMock(spec=discord.TextChannel)
    message.channel.id = 789

    timeline: list[tuple[str, str]] = []
    in_flight = 0
    overlaps: list[str] = []

    async def _record(api: str, **kwargs):
        nonlocal in_flight
        body = kwargs.get("content", "") or ""
        in_flight += 1
        if in_flight > 1:
            overlaps.append(body[:20])
        timeline.append(("start", body))
        # Yield control: a concurrent implementation would start the next send here.
        await asyncio.sleep(0)
        timeline.append(("end", body))
        in_flight -= 1
        return _FakeSentMessage(content=body, embeds=kwargs.get("embeds") or [], files=kwargs.get("files") or [])

    async def _reply(**kwargs):
        return await _record("reply", **kwargs)

    async def _send(**kwargs):
        return await _record("send", **kwargs)

    reply_target = MagicMock(spec=discord.Message)
    reply_target.reply = AsyncMock(side_effect=_reply)
    message.channel.send = AsyncMock(side_effect=_send)
    reply_target.channel = message.channel

    chunks = LLMBot._chunk_message_content(bot_stub, content)
    assert len(chunks) == 2, "fixture must produce exactly two parts"

    await LLMBot._send_chunked_reply(
        bot_stub,
        message=message,
        action=action,
        base_extra={"msg_id": message.id},
        force_reply_target=reply_target,
        target_message=None,
        dispatch_meta={"trigger_message_id": message.id},
        content=content,
        files=None,
        chunks=chunks,
    )

    assert overlaps == [], f"sends overlapped, delivery order is not guaranteed: {overlaps}"

    # Strict alternation start/end/start/end proves one-at-a-time sequencing.
    assert [phase for phase, _ in timeline] == ["start", "end", "start", "end"]

    # The wire order must match the chunk order, not length order.
    wire_order = [body for phase, body in timeline if phase == "start"]
    assert wire_order[0].endswith(chunks[0]), "first part on the wire must be chunk 1"
    assert wire_order[1] == chunks[1], "second part on the wire must be chunk 2"
    assert len(wire_order[0]) > len(wire_order[1]), "fixture should have a long part then a short one"


@pytest.mark.asyncio
async def test_send_chunked_stops_after_a_failed_part(bot_stub) -> None:
    """A mid-sequence failure must not let later parts through out of context.

    Emitting part 3 after part 2 was lost would present the user with text that
    reads as continuous but silently skips a section.
    """
    content = ("A" * 1900) + "\n\n" + ("B" * 1900) + "\n\n" + ("C" * 1900)
    action = BotAction(content=content, embeds=[])

    message = MagicMock(spec=discord.Message)
    message.id = 123
    message.content = "hi"
    message.reference = None
    message.guild = MagicMock(id=456)
    message.author = MagicMock(id=111, bot=False, mention="<@111>")
    message.channel = MagicMock(spec=discord.TextChannel)
    message.channel.id = 789

    sent_bodies: list[str] = []

    async def _reply(**kwargs):
        sent_bodies.append(kwargs.get("content", ""))
        return _FakeSentMessage(content=kwargs.get("content", ""))

    async def _send(**kwargs):
        sent_bodies.append(kwargs.get("content", ""))
        raise discord.errors.HTTPException(MagicMock(status=500), {"code": 0, "message": "boom"})

    reply_target = MagicMock(spec=discord.Message)
    reply_target.reply = AsyncMock(side_effect=_reply)
    message.channel.send = AsyncMock(side_effect=_send)
    reply_target.channel = message.channel

    chunks = LLMBot._chunk_message_content(bot_stub, content)
    assert len(chunks) >= 3, "fixture must produce at least three parts"

    await LLMBot._send_chunked_reply(
        bot_stub,
        message=message,
        action=action,
        base_extra={"msg_id": message.id},
        force_reply_target=reply_target,
        target_message=None,
        dispatch_meta={"trigger_message_id": message.id},
        content=content,
        files=None,
        chunks=chunks,
    )

    assert len(sent_bodies) == 2, f"must stop at the failed part, attempted {len(sent_bodies)}"


# ===================== on_message: no self-reply loop =====================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("author_id", "self_id"),
    [
        (99999, 99999),  # self-authored bot message
        (22222, 99999),  # other bot-authored message
    ],
)
async def test_on_message_ignores_bot_authors(bot_stub, author_id, self_id) -> None:
    bot_stub.user.id = self_id
    bot_stub.router = MagicMock()

    message = MagicMock(spec=discord.Message)
    message.id = 42
    message.content = "hello"
    message.attachments = []
    message.author = MagicMock(id=author_id, bot=True)

    await LLMBot.on_message(bot_stub, message)

    bot_stub.router.dispatch_message.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_ignores_own_continuation_message_no_loop(bot_stub) -> None:
    bot_stub.user.id = 99999
    bot_stub.router = MagicMock()

    message = MagicMock(spec=discord.Message)
    message.id = 999
    message.content = "continuation of long response"
    message.attachments = []
    # Message is from the bot itself (self)
    message.author = MagicMock(id=99999, bot=True)

    await LLMBot.on_message(bot_stub, message)

    # Must not be dispatched to router.
    bot_stub.router.dispatch_message.assert_not_called()


@pytest.mark.asyncio
async def test_no_infinite_self_reply_loop_via_continuation(bot_stub) -> None:
    # Concept: if on_message ignored bot messages, we cannot get into a loop
    # where bot replies to its own continuations. This test encodes that guarantee.

    bot_stub.user.id = 99999
    bot_stub.router = MagicMock()

    # Simulate multiple continuation-like messages from the bot.
    for i in range(5):
        message = MagicMock(spec=discord.Message)
        message.id = 1000 + i
        message.content = f"continuation {i}"
        message.attachments = []
        message.author = MagicMock(id=99999, bot=True)

        await LLMBot.on_message(bot_stub, message)

    # None of these should trigger router dispatch.
    bot_stub.router.dispatch_message.assert_not_called()
