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

    # Mock for the config object
    mock_config = MagicMock()
    mock_config.get.return_value = "Test prompt"
    bot.config = mock_config

    # Mock for the tts_manager object
    mock_tts_manager = AsyncMock()
    mock_tts_manager.voice = "test_voice"
    bot.tts_manager = mock_tts_manager

    # Mock for the event loop
    bot.loop = AsyncMock()

    return bot

@pytest.fixture
@patch('bot.command_parser.parse_command')
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
async def test_dm_text_to_text_flow(router_setup, mock_message):
    """Test standard text-to-text flow in a DM."""
    router_factory, mock_parse_command, mock_logger = router_setup
    mock_flow_text = AsyncMock(return_value="Test response")
    router = router_factory(flow_overrides={'process_text': mock_flow_text})
    mock_message.content = "Hello, bot!"
    mock_parse_command.return_value = MockParsedCommand(command=Command.CHAT, cleaned_content="Hello, bot!")

    response = await router.dispatch_message(mock_message)

    mock_flow_text.assert_called_once_with("Hello, bot!", str(mock_message.author.id))
    assert response.text == "Test response"
    assert response.audio_path is None

@pytest.mark.asyncio
async def test_guild_text_to_tts_flow(router_setup, mock_message):
    """Test text-to-TTS flow in a guild with a !say command."""
    router_factory, mock_parse_command, mock_logger = router_setup
    mock_flow_text = AsyncMock(return_value="TTS response")
    mock_flow_tts = AsyncMock(return_value="/path/to/audio.wav")
    router = router_factory(flow_overrides={'process_text': mock_flow_text, 'generate_tts': mock_flow_tts})
    mock_message.guild = MagicMock()
    mock_parse_command.return_value = MockParsedCommand(command=Command.SAY, cleaned_content="How are you?")

    response = await router.dispatch_message(mock_message)

    mock_flow_text.assert_called_once_with("How are you?", str(mock_message.author.id))
    mock_flow_tts.assert_called_once_with("TTS response")
    assert response.text is None
    assert response.audio_path == "/path/to/audio.wav"

@pytest.mark.asyncio
async def test_dm_image_to_text_flow(router_setup, mock_message):
    """Test image-to-text flow in a DM."""
    router_factory, mock_parse_command, mock_logger = router_setup
    mock_flow_attachments = AsyncMock(return_value="Image processed")
    router = router_factory(flow_overrides={'process_attachments': mock_flow_attachments})
    mock_attachment = MagicMock(spec=discord.Attachment)
    mock_attachment.content_type = 'image/png'
    mock_message.attachments = [mock_attachment]
    mock_message.content = "What is this?"
    mock_parse_command.return_value = MockParsedCommand(command=Command.CHAT, cleaned_content="What is this?")

    response = await router.dispatch_message(mock_message)

    mock_flow_attachments.assert_called_once_with(mock_message, "What is this?")
    assert response.text == "Image processed"
    assert response.audio_path is None

@pytest.mark.asyncio
async def test_guild_message_without_mention_is_ignored(router_setup, mock_message):
    """Test that messages in guilds without a mention are ignored."""
    router_factory, mock_parse_command, mock_logger = router_setup
    mock_flow_text = AsyncMock()
    mock_flow_attachments = AsyncMock()
    router = router_factory(flow_overrides={'process_text': mock_flow_text, 'process_attachments': mock_flow_attachments})
    mock_message.guild = MagicMock()
    mock_parse_command.return_value = None  # This is the key for this test

    response = await router.dispatch_message(mock_message)

    assert response is None
    mock_flow_text.assert_not_called()
    mock_flow_attachments.assert_not_called()

@pytest.mark.asyncio
async def test_ping_command(router_setup, mock_message):
    """Test that the PING command returns 'Pong!'"""
    router_factory, mock_parse_command, mock_logger = router_setup
    router = router_factory()
    mock_parse_command.return_value = MockParsedCommand(command=Command.PING, cleaned_content="")

    response = await router.dispatch_message(mock_message)

    assert response.text == "Pong!"
    assert response.audio_path is None

@pytest.mark.asyncio
async def test_unhandled_attachment_type(router_setup, mock_message):
    """Test that an unhandled attachment type returns an error message."""
    router_factory, mock_parse_command, mock_logger = router_setup
    router = router_factory()
    mock_attachment = MagicMock(spec=discord.Attachment)
    mock_attachment.content_type = 'application/zip'
    mock_message.attachments = [mock_attachment]
    mock_parse_command.return_value = MockParsedCommand(command=Command.CHAT, cleaned_content="Check this out")

    response = await router.dispatch_message(mock_message)

    assert "I'm sorry, I can't process that type of attachment." in response.text

@pytest.mark.asyncio
async def test_no_processed_text_returns_error_message(router_setup, mock_message):
    """Test that if no text is produced, an error message is returned."""
    router_factory, mock_parse_command, mock_logger = router_setup
    mock_flow_text = AsyncMock(return_value=None) # Simulate flow returning nothing
    router = router_factory(flow_overrides={'process_text': mock_flow_text})
    mock_parse_command.return_value = MockParsedCommand(command=Command.CHAT, cleaned_content="Hello")

    response = await router.dispatch_message(mock_message)

    assert response.text == "I'm sorry, I couldn't process that. Please try again."

@pytest.mark.asyncio
async def test_say_command_returns_only_audio(router_setup, mock_message):
    """Test that !say command returns only audio path and no text."""
    router_factory, mock_parse_command, mock_logger = router_setup
    mock_flow_text = AsyncMock(return_value="TTS response")
    mock_flow_tts = AsyncMock(return_value="/path/to/audio.wav")
    router = router_factory(flow_overrides={'process_text': mock_flow_text, 'generate_tts': mock_flow_tts})
    mock_parse_command.return_value = MockParsedCommand(command=Command.SAY, cleaned_content="Say this")

    response = await router.dispatch_message(mock_message)

    assert response.text is None
    assert response.audio_path == "/path/to/audio.wav"

@pytest.mark.asyncio
async def test_error_in_dispatch_returns_error_message(router_setup, mock_message):
    """Test that a generic exception in dispatch returns a user-friendly error."""
    router_factory, mock_parse_command, mock_logger = router_setup
    mock_flow_text = AsyncMock(side_effect=Exception("A wild error appears!"))
    router = router_factory(flow_overrides={'process_text': mock_flow_text})
    mock_parse_command.return_value = MockParsedCommand(command=Command.CHAT, cleaned_content="Hello")

    response = await router.dispatch_message(mock_message)

    assert response.text == "I'm sorry, an unexpected error occurred."
