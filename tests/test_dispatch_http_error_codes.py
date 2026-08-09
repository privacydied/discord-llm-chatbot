"""Dispatch fallbacks keyed on Discord JSON error codes.

These two codes were previously conflated: the branch tested for 50035 but was
labelled (and intended) for "Unknown Message", which is 10008. So the
deleted-trigger-message fallback never ran, and a genuine 50035 was retried with
the identical payload that had just been rejected.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import discord
import pytest

from bot.action import BotAction
from bot.core.bot import _DISCORD_ERR_INVALID_FORM_BODY, _DISCORD_ERR_UNKNOWN_MESSAGE, LLMBot


class _FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class _FakeChannel:
    def __init__(self) -> None:
        self.sends: list[str] = []

    def typing(self):
        return _FakeTyping()

    async def send(self, content=None, embeds=None, files=None, **_kwargs):
        self.sends.append(content or "")
        return _FakeSent(content or "")


class _FakeSent:
    _counter = 5000

    def __init__(self, content: str) -> None:
        _FakeSent._counter += 1
        self.id = _FakeSent._counter
        self.content = content


def _http_exception(code: int, status: int = 400):
    """Build an HTTPException carrying a Discord JSON error code."""
    response = MagicMock()
    response.status = status
    response.reason = "Bad Request"
    response.headers = {}
    return discord.HTTPException(response, {"code": code, "message": "test"})


class _FailingMessage:
    """A trigger message whose reply() always fails with a given code."""

    def __init__(self, channel: _FakeChannel, code: int) -> None:
        self.id = 4242
        self.channel = channel
        self.content = "hi"
        self.reference = None
        self.author = MagicMock(id=111, bot=False, mention="<@111>")
        self.guild = MagicMock(id=222)
        self._code = code
        self.reply_attempts = 0

    async def reply(self, **_kwargs):
        self.reply_attempts += 1
        raise _http_exception(self._code)

    async def delete(self):
        return True


def _make_bot():
    bot = LLMBot(command_prefix="!", intents=discord.Intents.none(), config={})
    bot.enhanced_context_manager = None
    return bot


@pytest.mark.asyncio
async def test_unknown_message_falls_back_to_channel_send() -> None:
    """10008 means the trigger message is gone — degrade to a plain channel message.

    Before the code fix this branch was unreachable, so a deleted trigger message
    propagated the exception instead of the reply landing in the channel.
    """
    bot = _make_bot()
    channel = _FakeChannel()
    incoming = _FailingMessage(channel, _DISCORD_ERR_UNKNOWN_MESSAGE)

    await bot._execute_action(incoming, BotAction(content="short reply", embeds=[]))

    assert incoming.reply_attempts == 1
    assert channel.sends == ["short reply"], "must land in the channel, not raise"


@pytest.mark.asyncio
async def test_invalid_form_body_is_not_retried_identically() -> None:
    """50035 must propagate, not re-send the payload Discord just rejected."""
    bot = _make_bot()
    channel = _FakeChannel()
    incoming = _FailingMessage(channel, _DISCORD_ERR_INVALID_FORM_BODY)

    with pytest.raises(discord.HTTPException):
        await bot._execute_action(incoming, BotAction(content="short reply", embeds=[]))

    assert incoming.reply_attempts == 1
    assert channel.sends == [], "the rejected payload must not be re-sent verbatim"


@pytest.mark.asyncio
async def test_other_http_errors_still_propagate() -> None:
    bot = _make_bot()
    channel = _FakeChannel()
    incoming = _FailingMessage(channel, 50013)  # Missing Permissions

    with pytest.raises(discord.HTTPException):
        await bot._execute_action(incoming, BotAction(content="short reply", embeds=[]))

    assert channel.sends == []
