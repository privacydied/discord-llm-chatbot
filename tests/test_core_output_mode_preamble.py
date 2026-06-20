import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.action import BotAction
from bot.public_output import sanitize_public_text
from bot.core.bot import LLMBot


class MockTextChannel:
    def __init__(self) -> None:
        self.id = 456
        self.name = "test-channel"
        self.type = "text"
        self.position = 0
        self.send = AsyncMock()
        self.guild = MagicMock()


class MockUser:
    def __init__(self, id, name, bot=False) -> None:
        self.id = id
        self.name = name
        self.discriminator = "1234"
        self.bot = bot
        self.mention = f"<@{id}>"


class MockMessage:
    def __init__(self, bot, is_dm=False) -> None:
        self.id = 112233
        self.content = f"<@{bot.user.id}> Hello bot!"
        self.author = MockUser(12345, "TestUser")
        self.channel = MockTextChannel() if not is_dm else MagicMock(
            id=789, type="dm", send=AsyncMock(), recipient=MockUser(12345, "TestUser")
        )
        self.guild = MagicMock(id=789, name="Test Guild")
        self.mentions = [bot.user]
        self.reference = None
        self.reply = AsyncMock()


@pytest.fixture
def bot():
    mock_bot = MagicMock(spec=LLMBot)
    mock_user = MockUser(99999, "TestBot", bot=True)
    type(mock_bot).user = MagicMock(return_value=mock_user)
    mock_bot.command_prefix = "!"
    mock_bot.user = mock_user
    mock_bot.config = {}
    mock_bot.tts_manager = MagicMock()
    mock_bot.tts_manager.is_available = MagicMock(return_value=False)
    mock_bot.voice_message_publisher = MagicMock()
    mock_bot.enhanced_context_manager = None
    mock_bot.logger = logging.getLogger("test_logger")
    mock_bot.logger.setLevel(logging.DEBUG)
    mock_bot.logger.handlers = []
    mock_bot._is_ready = MagicMock()
    mock_bot._is_ready.is_set = MagicMock(return_value=True)
    mock_bot._last_sent_message_for_finalize = None

    async def _call_with_discord_retry(operation, func, *, base_extra=None, attempts=3, **kwargs):
        import inspect

        result = func()
        if inspect.iscoroutine(result):
            result = await result
        sent = getattr(result, "result", result)
        mock_bot._last_sent_message_for_finalize = sent
        return sent

    mock_bot._call_with_discord_retry = _call_with_discord_retry

    return mock_bot


@pytest.mark.asyncio
async def test_mode_preamble_regression_prod_send_path(bot) -> None:
    action = BotAction(content="MODE: NORMAL\n\nchilling. what's the move?")
    assert action.content == "chilling. what's the move?"
    assert not action.content.lstrip().lower().startswith(("mode: normal", "mode: political"))

    message = MockMessage(bot)

    await LLMBot._execute_action(
        bot,
        message,
        action,
        target_message=None,
        dispatch_meta={},
    )

    assert message.reply.call_count == 1
    sent = message.reply.call_args
    sent_text = sent.kwargs.get("content") or (sent.args[0] if sent.args else "")
    assert sent_text == "chilling. what's the move?"
    assert not str(sent_text).lstrip().lower().startswith(("mode: normal", "mode: political"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MODE: NORMAL\n\nchilling, bro", "chilling, bro"),
        ("MODE: NORMAL\r\n\r\nchilling, bro", "chilling, bro"),
        ("MODE: NORMAL\nchilling, bro", "chilling, bro"),
        (" MODE: NORMAL\n\nchilling, bro", "chilling, bro"),
        ("**MODE: NORMAL**\n\nchilling, bro", "chilling, bro"),
        ("`MODE: NORMAL`\n\nchilling, bro", "chilling, bro"),
        ("> MODE: NORMAL\n\nchilling, bro", "chilling, bro"),
        ("MODE: POLITICAL\n\nactual answer", "actual answer"),
        ("MODE: NORMAL\nMODE: NORMAL\n\nactual answer", "actual answer"),
        ("normal mode in vim is useful", "normal mode in vim is useful"),
        ("mode: normal is the setting you asked about", "mode: normal is the setting you asked about"),
        ("hello\nMODE: NORMAL", "hello\nMODE: NORMAL"),
        ("the model said MODE: NORMAL yesterday", "the model said MODE: NORMAL yesterday"),
    ],
)
def test_regression_prod_send_path_after_existing_helpers(raw: str, expected: str) -> None:
    assert sanitize_public_text(raw) == expected
