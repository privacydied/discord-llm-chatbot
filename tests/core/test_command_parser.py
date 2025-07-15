import pytest
from unittest.mock import MagicMock

import discord

from bot.command_parser import parse_command
from bot.types import Command, ParsedCommand


@pytest.fixture
def mock_bot():
    """Pytest fixture for a mock bot instance."""
    bot = MagicMock(spec=discord.ext.commands.Bot)
    bot.user = MagicMock(spec=discord.ClientUser)
    bot.user.id = 1234567890
    # Simulate the bot being mentioned
    bot.user.mentioned_in.return_value = True
    return bot


# --- DM Scenarios ---

def test_parse_dm_known_command(mock_bot):
    """Verify parsing a known command in a DM."""
    msg = MagicMock(spec=discord.Message)
    msg.content = "!say Hello World"
    msg.channel = MagicMock(spec=discord.DMChannel)
    result = parse_command(msg, mock_bot)
    assert result == ParsedCommand(command=Command.SAY, cleaned_content="Hello World")

def test_parse_dm_chat_message(mock_bot):
    """Verify parsing a non-command message in a DM."""
    msg = MagicMock(spec=discord.Message)
    msg.content = "just a regular message"
    msg.channel = MagicMock(spec=discord.DMChannel)
    result = parse_command(msg, mock_bot)
    assert result == ParsedCommand(command=Command.CHAT, cleaned_content="just a regular message")

# --- Guild Scenarios ---

def test_parse_guild_command_with_mention(mock_bot):
    """Verify parsing a command with a mention in a guild."""
    msg = MagicMock(spec=discord.Message)
    msg.content = f"<@!{mock_bot.user.id}> !ping"
    msg.channel = MagicMock(spec=discord.TextChannel)
    result = parse_command(msg, mock_bot)
    assert result == ParsedCommand(command=Command.PING, cleaned_content="")

def test_parse_guild_chat_with_mention(mock_bot):
    """Verify parsing a chat message with a mention in a guild."""
    msg = MagicMock(spec=discord.Message)
    msg.content = f"<@{mock_bot.user.id}> how are you?"
    msg.channel = MagicMock(spec=discord.TextChannel)
    result = parse_command(msg, mock_bot)
    assert result == ParsedCommand(command=Command.CHAT, cleaned_content="how are you?")

def test_parse_guild_message_no_mention(mock_bot):
    """Verify that messages in guilds without a mention are ignored."""
    msg = MagicMock(spec=discord.Message)
    msg.content = "!ping"
    msg.channel = MagicMock(spec=discord.TextChannel)
    # Simulate the bot NOT being mentioned
    mock_bot.user.mentioned_in.return_value = False
    result = parse_command(msg, mock_bot)
    assert result is None

# --- Edge Cases ---

def test_parse_empty_message(mock_bot):
    """Verify that an empty message is ignored."""
    msg = MagicMock(spec=discord.Message)
    msg.content = ""
    msg.channel = MagicMock(spec=discord.DMChannel)
    result = parse_command(msg, mock_bot)
    assert result is None

def test_parse_mention_only(mock_bot):
    """Verify that a message with only a mention is ignored."""
    msg = MagicMock(spec=discord.Message)
    msg.content = f"<@{mock_bot.user.id}>"
    msg.channel = MagicMock(spec=discord.TextChannel)
    result = parse_command(msg, mock_bot)
    assert result is None
