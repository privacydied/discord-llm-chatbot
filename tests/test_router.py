"""
Unit tests for the Router class.

Verifies that the router correctly dispatches messages to the appropriate processing
flow based on context (DM vs. Guild), command, and attachments. Ensures that the
'1 IN > 1 OUT' principle is maintained and that channel rules are enforced.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import discord
from dataclasses import dataclass
import logging

from bot.router import Router, ResponseMessage
from bot.types import Command

@dataclass
class MockParsedCommand:
    command: Command
    cleaned_content: str

@pytest.fixture
def mock_bot():
    """Fixture for a mocked bot instance with all necessary attributes."""
    bot = MagicMock(spec=discord.Client)
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.user.mentioned_in.return_value = True
    mock_config = MagicMock()
    mock_config.get.return_value = "Test prompt"
    bot.config = mock_config
    mock_tts_manager = AsyncMock()
    mock_tts_manager.voice = "test_voice"
    bot.tts_manager = mock_tts_manager
    bot.loop = AsyncMock()
    return bot

@pytest.fixture
@patch('bot.router.parse_command')
def router_setup(mock_parse_command, mock_bot, mock_logger):
    """Fixture to provide a router factory and the parse_command mock."""
    def _create_router(flow_overrides=None):
        return Router(mock_bot, flow_overrides=flow_overrides, logger=mock_logger)
    return _create_router, mock_parse_command, mock_logger

@pytest.fixture
def mock_logger():
    """Fixture for a mocked logger instance."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def mock_message():
    """Fixture for a mocked discord.Message instance."""
    message = MagicMock(spec=discord.Message)
    message.id = 11223344
    message.guild = None
    message.channel = MagicMock(spec=discord.DMChannel)
    message.author = MagicMock(spec=discord.User)
    message.author.id = 98765
    message.attachments = []
    return message

# --- Test Cases ---

@pytest.mark.asyncio
async def test_ping_command(router_setup, mock_message):
    """Test that the PING command returns 'Pong!'"""
    router_factory, mock_parse_command, _ = router_setup
    router = router_factory()
    mock_parse_command.return_value = MockParsedCommand(command=Command.PING, cleaned_content="")
    response = await router.dispatch_message(mock_message)
    assert response is not None
    assert response.text == "Pong!"

@pytest.mark.asyncio
async def test_unhandled_attachment_type(router_setup, mock_message):
    """Test that an unhandled attachment type returns an error message."""
    router_factory, mock_parse_command, _ = router_setup
    router = router_factory()
    attachment = MagicMock()
    attachment.filename = 'test.zip'
    mock_message.attachments = [attachment]
    mock_parse_command.return_value = MockParsedCommand(command=Command.CHAT, cleaned_content="Check out this file")
    response = await router.dispatch_message(mock_message)
    assert response is not None
    assert "I'm sorry, I can't process that type of attachment" in response.text

@pytest.mark.asyncio
async def test_no_processed_text_returns_error_message(router_setup, mock_message):
    """Test that if no text is produced, an error message is returned."""
    router_factory, mock_parse_command, _ = router_setup
    mock_flow_text = AsyncMock(return_value=None)
    router = router_factory(flow_overrides={'process_text': mock_flow_text})
    mock_parse_command.return_value = MockParsedCommand(command=Command.CHAT, cleaned_content="a valid message")
    response = await router.dispatch_message(mock_message)
    assert response is not None
    assert "I'm sorry, I wasn't able to process that" in response.text

@pytest.mark.asyncio
async def test_say_command_returns_only_audio(router_setup, mock_message):
    """Test that !say command returns only audio path and no text."""
    router_factory, mock_parse_command, _ = router_setup
    mock_flow_tts = AsyncMock(return_value="/path/to/tts.wav")
    router = router_factory(flow_overrides={'generate_tts': mock_flow_tts})
    mock_parse_command.return_value = MockParsedCommand(command=Command.SAY, cleaned_content="speak this")
    response = await router.dispatch_message(mock_message)
    assert response is not None
    assert response.audio_path == "/path/to/tts.wav"
    assert response.text is None

@pytest.mark.asyncio
async def test_error_in_dispatch_returns_error_message(router_setup, mock_message):
    """Test that a generic exception in dispatch returns a user-friendly error."""
    router_factory, mock_parse_command, _ = router_setup
    mock_flow_text = AsyncMock(side_effect=Exception("KABOOM"))
    router = router_factory(flow_overrides={'process_text': mock_flow_text})
    mock_parse_command.return_value = MockParsedCommand(command=Command.CHAT, cleaned_content="this will break")
    response = await router.dispatch_message(mock_message)
    assert response is not None
    assert "I'm sorry, an unexpected error occurred" in response.text

@pytest.mark.asyncio
async def test_guild_message_without_mention_is_ignored(router_setup, mock_message):
    """Test that messages in guilds without a mention are ignored."""
    router_factory, mock_parse_command, _ = router_setup
    router = router_factory()
    mock_message.guild = MagicMock()
    mock_parse_command.return_value = None  # This is the key for this test
    response = await router.dispatch_message(mock_message)
    assert response is None
