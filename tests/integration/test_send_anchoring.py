import pytest
from unittest.mock import AsyncMock, MagicMock
import discord

from bot.core.bot import LLMBot
from bot.action import BotAction


class _AsyncTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_reply_anchors_to_triggering_user(monkeypatch):
    """In REPLY_CASE, the bot should anchor to the triggering user message (never the bot/parent).
    We assert that message.reply() is called, not parent_msg.reply().
    """
    # Minimal LLMBot stub
    bot = MagicMock(spec=LLMBot)
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.user.bot = True
    bot.config = {"MEM_LOG_SUBSYS": "mem.test"}
    bot.logger = MagicMock()
    # Satisfy readiness check in _execute_action
    bot._is_ready = MagicMock()
    bot._is_ready.is_set.return_value = True
    # Disable post-send context manager hook
    bot.enhanced_context_manager = None

    # Mock channel typing context
    channel = MagicMock(spec=discord.TextChannel)
    channel.typing = MagicMock(return_value=_AsyncTyping())

    # Parent message (could be bot or human; irrelevant for anchoring)
    parent_msg = MagicMock(spec=discord.Message)
    parent_msg.id = 999
    parent_msg.author = MagicMock()
    parent_msg.reply = AsyncMock()

    # Triggering user message (the reply)
    user = MagicMock()
    user.id = 777
    user.bot = False

    message = MagicMock(spec=discord.Message)
    message.id = 1001
    message.author = user
    message.channel = channel
    message.content = "thoughts on X?"
    message.mentions = []
    message.reply = AsyncMock()

    # Emulate a reply to parent
    ref = MagicMock()
    ref.message_id = parent_msg.id
    ref.resolved = parent_msg
    message.reference = ref

    # Execute action
    action = BotAction(content="ok")

    # _execute_action is async on the real class; bind it to our MagicMock instance
    # by calling the function on the class with our instance
    exec_coro = LLMBot._execute_action(bot, message, action)
    assert hasattr(exec_coro, "__await__")
    await exec_coro

    # Assert that we anchored to the triggering user message, not the parent
    message.reply.assert_awaited()
    parent_msg.reply.assert_not_called()


@pytest.mark.asyncio
async def test_reply_missing_parent_still_targets_triggering_user(monkeypatch):
    """If the reply's parent cannot be fetched or is missing, we still anchor to the triggering user message.
    This verifies the fallback is resilient and does not crash or self-anchor.
    """
    # Minimal LLMBot stub
    bot = MagicMock(spec=LLMBot)
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.user.bot = True
    bot.config = {"MEM_LOG_SUBSYS": "mem.test"}
    bot.logger = MagicMock()

    # Satisfy readiness check in _execute_action
    bot._is_ready = MagicMock()
    bot._is_ready.is_set.return_value = True
    # Disable post-send context manager hook
    bot.enhanced_context_manager = None

    # Mock channel typing context and fetch_message raising
    async def _fetch_message(_):
        raise Exception("not found")

    channel = MagicMock(spec=discord.TextChannel)
    channel.typing = MagicMock(return_value=_AsyncTyping())
    channel.fetch_message = AsyncMock(side_effect=_fetch_message)

    # Parent message is unresolved and fetch fails
    ref = MagicMock()
    ref.message_id = 424242
    ref.resolved = None

    # Triggering user message (the reply)
    user = MagicMock()
    user.id = 888
    user.bot = False

    message = MagicMock(spec=discord.Message)
    message.id = 2002
    message.author = user
    message.channel = channel
    message.content = "@bot ?"
    message.mentions = []
    message.reference = ref
    message.reply = AsyncMock()

    action = BotAction(content="ok")
    await LLMBot._execute_action(bot, message, action)

    # Even with missing parent, we should reply to the triggering message
    message.reply.assert_awaited()
