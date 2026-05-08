from __future__ import annotations

import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.action import BotAction
from bot.core.bot import LLMBot, _DISCORD_MAX_CONTENT_LEN


class _FakeSentMessage:
    def __init__(self, content: str, embeds=None, files=None):
        self.content = content
        self.embeds = embeds or []
        self.files = files or []
        self.reply = AsyncMock()
        self.id = id(self)


@pytest.fixture
def bot_stub():
    bot = MagicMock(spec=LLMBot)
    bot.user = MagicMock(id=99999, bot=True)
    bot.config = {}
    bot.logger = MagicMock()
    bot.enhanced_context_manager = None
    bot._processed_messages = OrderedDict()
    bot._dispatch_lock = asyncio.Lock()

    async def _call_with_discord_retry(_name, factory, **_kwargs):
        return await factory()

    bot._call_with_discord_retry = _call_with_discord_retry
    return bot


@pytest.mark.asyncio
async def test_chunk_short_text_returns_single_chunk_unchanged(bot_stub):
    text = "short response"

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert chunks == [text]


@pytest.mark.asyncio
async def test_chunk_text_just_over_limit_returns_two_chunks(bot_stub):
    text = "x" * (_DISCORD_MAX_CONTENT_LEN + 1)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 2
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_prefers_paragraph_boundaries(bot_stub):
    text = ("a" * 1880) + "\n\n" + ("b" * 80)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 2
    assert chunks[0].endswith("\n\n")
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_prefers_newline_boundaries(bot_stub):
    text = ("a" * 1895) + "\n" + ("b" * 50)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 2
    assert chunks[0].endswith("\n")
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_prefers_simple_sentence_boundaries(bot_stub):
    text = ("a" * 1890) + ". " + ("b" * 50)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 2
    assert chunks[0].endswith(". ")
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_hard_splits_long_unbroken_text(bot_stub):
    text = "z" * ((_DISCORD_MAX_CONTENT_LEN * 2) + 17)

    chunks = LLMBot._chunk_message_content(bot_stub, text)

    assert len(chunks) == 3
    assert "".join(chunks) == text
    assert all(len(chunk) <= _DISCORD_MAX_CONTENT_LEN for chunk in chunks)


@pytest.mark.asyncio
async def test_send_chunked_reply_is_sequential_and_first_part_only(bot_stub):
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

    message.reply = AsyncMock(side_effect=_reply_side_effect)

    result = await LLMBot._send_chunked_reply(
        bot_stub,
        message=message,
        action=action,
        base_extra={"msg_id": message.id},
        force_reply_target=None,
        target_message=None,
        dispatch_meta={"trigger_message_id": message.id},
        content=content,
        files=files,
        chunks=LLMBot._chunk_message_content(bot_stub, content),
    )

    assert result is sent_messages[-1]
    assert [call["content"] for call in reply_calls] == [content[:1900], content[1900:]]
    assert len(reply_calls) == 2
    assert reply_calls[0]["embeds"] == action.embeds
    assert reply_calls[0]["files"] == files
    assert reply_calls[1]["embeds"] == []
    assert reply_calls[1]["files"] is None
    sent_messages[0].reply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "author_id,self_id",
    [
        (99999, 99999),  # self-authored bot message
        (22222, 99999),  # other bot-authored message
    ],
)
async def test_on_message_ignores_bot_authors(bot_stub, author_id, self_id):
    bot_stub.user.id = self_id
    bot_stub.router = MagicMock()

    message = MagicMock(spec=discord.Message)
    message.id = 42
    message.content = "hello"
    message.attachments = []
    message.author = MagicMock(id=author_id, bot=True)

    await LLMBot.on_message(bot_stub, message)

    bot_stub.router.dispatch_message.assert_not_called()
    assert list(bot_stub._processed_messages.keys()) == []
