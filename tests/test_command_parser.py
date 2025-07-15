import pytest
from unittest.mock import MagicMock

from bot.command_parser import parse_command
from bot.core.client import Bot
from bot.types import Command, ParsedCommand


@pytest.fixture
def mock_bot() -> MagicMock:
    """Creates a mock Bot object with a mock user."""
    bot = MagicMock(spec=Bot)
    bot.user = MagicMock()
    bot.user.id = 1234567890
    bot.user.mentioned_in.return_value = True
    return bot


@pytest.fixture
def mock_message() -> MagicMock:
    """Creates a mock discord.Message object."""
    message = MagicMock()
    message.channel = MagicMock()
    message.author = MagicMock()
    message.author.id = 54321
    return message


# Test Cases

def test_dm_chat_message(mock_bot, mock_message):
    """Test that a regular DM message is parsed as a CHAT command."""
    mock_message.content = "Hello there"
    mock_message.channel.type = 'private'

    result = parse_command(mock_message, mock_bot)

    assert result is not None
    assert result.command == Command.CHAT
    assert result.cleaned_content == "Hello there"

def test_guild_chat_message_with_mention(mock_bot, mock_message):
    """Test that a regular guild message with a mention is parsed as CHAT."""
    mock_message.content = f"<@!{mock_bot.user.id}> How are you?"
    mock_message.channel.type = 'text'

    result = parse_command(mock_message, mock_bot)

    assert result is not None
    assert result.command == Command.CHAT
    assert result.cleaned_content == "How are you?"

def test_guild_message_without_mention_is_ignored(mock_bot, mock_message):
    """Test that a guild message without a mention returns None."""
    mock_message.content = "!ping"
    mock_message.channel.type = 'text'
    mock_bot.user.mentioned_in.return_value = False

    result = parse_command(mock_message, mock_bot)

    assert result is None

def test_dm_ping_command(mock_bot, mock_message):
    """Test that a !ping command in a DM is parsed correctly."""
    mock_message.content = "!ping"
    mock_message.channel.type = 'private'

    result = parse_command(mock_message, mock_bot)

    assert result.command == Command.PING
    assert result.cleaned_content == ""

def test_guild_tts_command_with_args(mock_bot, mock_message):
    """Test a !tts command with arguments in a guild."""
    mock_message.content = f"<@!{mock_bot.user.id}> !tts some text to speak"
    mock_message.channel.type = 'text'

    result = parse_command(mock_message, mock_bot)

    assert result.command == Command.TTS
    assert result.cleaned_content == "some text to speak"

def test_guild_mention_only_is_ignored(mock_bot, mock_message):
    """Test that a message with only a mention is ignored."""
    mock_message.content = f"<@!{mock_bot.user.id}>"
    mock_message.channel.type = 'text'

    result = parse_command(mock_message, mock_bot)

    assert result is None, "A message with only a mention should be ignored"

def test_dm_unknown_command_is_ignored(mock_bot, mock_message):
    """Test that an unknown command like !foo is ignored."""
    mock_message.content = "!foo bar baz"
    mock_message.channel.type = 'private'

    result = parse_command(mock_message, mock_bot)

    assert result is None
