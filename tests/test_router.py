"""
Unit tests for the Router class.

Verifies that the router correctly dispatches messages to the appropriate processing
flow based on context (DM vs. Guild), command, and attachments. Ensures that the
'1 IN > 1 OUT' principle is maintained and that channel rules are enforced.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import discord

import os
import sys
# Add the project root to the path to allow importing the bot module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot.router import Router, ResponseMessage, InputModality, OutputModality
from bot.command_parser import Command, ParsedCommand

# Mock discord objects for testing
@pytest.fixture
def router(mocker):
    """Fixture to create a Router instance with mocked dependencies."""
    mock_bot = MagicMock(spec=discord.Client)
    mock_bot.user.id = 12345
    mock_bot.loop = asyncio.get_event_loop()

    mock_config = {
        "DISCORD_TOKEN": "test_token",
        "SYSTEM_PROMPT": "You are a helpful assistant.",
        "KOKORO_MODEL_PATH": "tts/onnx",
        "KOKORO_VOICE_PACK_PATH": "tts/voices",
    }

    # Mock the TTSManager and PDFProcessor
    mock_tts_manager = mocker.AsyncMock()
    mock_pdf_processor = mocker.AsyncMock()

    # Create the Router instance with all dependencies
    router_instance = Router(bot=mock_bot, config=mock_config, tts_manager=mock_tts_manager)
    
    # Attach mocks directly to the instance
    router_instance.brain_infer = mocker.AsyncMock(return_value="Mocked brain response.")
    router_instance.see_infer = mocker.AsyncMock(return_value="Mocked vision response.")

    return router_instance

@pytest.fixture
def mock_message():
    """Fixture for a mocked discord.Message instance."""
    message = MagicMock()
    message.guild = None # Default to DM
    message.channel = MagicMock()
    message.channel.__class__.__name__ = 'DMChannel'
    message.author.id = 987654321
    message.attachments = []
    return message

# --- Test Cases ---

@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_dm_text_to_text_flow(mock_parse_command, router, mock_message):
    """Test standard text-to-text flow in a DM."""
    # Arrange
    mock_message.content = "Hello, bot!"
    mock_parse_command.return_value = ParsedCommand(Command.GENERAL, "Hello, bot!")
    # The router fixture already patches brain_infer
    brain_infer_mock = router.brain_infer
    brain_infer_mock.return_value = "Hello, user!"

    # Act
    response = await router.dispatch_message(mock_message)

    # Assert
    brain_infer_mock.assert_called_once_with("Hello, bot!", str(mock_message.author.id))
    assert isinstance(response, ResponseMessage)
    assert response.text == "Hello, user!"
    assert response.audio_path is None

@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_guild_text_to_tts_flow(mock_parse_command, router, mock_message):
    """Test text-to-TTS flow in a guild with a mention."""
    # Arrange
    mock_message.guild = MagicMock() # It's a guild message
    mock_message.channel.__class__.__name__ = 'TextChannel'
    router.bot.user.mentioned_in.return_value = True
    mock_message.content = f"<@{router.bot.user.id}> !speak How are you?"
    
    mock_parse_command.return_value = ParsedCommand(Command.SPEAK, "How are you?")
    
    brain_infer_mock = router.brain_infer
    brain_infer_mock.return_value = "I am well, thank you!"
    
    router.tts_manager.generate_tts.return_value = "/path/to/audio.wav"
    
    # Act
    response = await router.dispatch_message(mock_message)

    # Assert
    brain_infer_mock.assert_called_once_with("How are you?", str(mock_message.author.id))
    router.tts_manager.generate_tts.assert_called_once_with("I am well, thank you!")
    assert isinstance(response, ResponseMessage)
    assert response.text == "I am well, thank you!"
    assert response.audio_path == "/path/to/audio.wav"

@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_dm_image_to_text_flow(mock_parse_command, router, mock_message):
    """Test image-to-text flow in a DM."""
    # Arrange
    mock_attachment = MagicMock()
    mock_attachment.filename = 'test.png'
    mock_attachment.url = 'http://example.com/test.png'
    mock_attachment.content_type = 'image/png'
    mock_message.attachments = [mock_attachment]
    mock_message.content = "What is this?"

    mock_parse_command.return_value = ParsedCommand(Command.GENERAL, "What is this?")
    see_infer_mock = router.see_infer
    see_infer_mock.return_value = "It is a cat."
    brain_infer_mock = router.brain_infer
    brain_infer_mock.return_value = "The image shows a cat sitting on a mat."

    # Act
    with patch('tempfile.NamedTemporaryFile') as mock_tmp_file:
        mock_tmp_file.return_value.__enter__.return_value.name = "/tmp/test.png"
        response = await router.dispatch_message(mock_message)

    # Assert
    see_infer_mock.assert_called_once()
    brain_infer_mock.assert_called_once()
    assert isinstance(response, ResponseMessage)
    assert "The image shows a cat" in response.text
    assert response.audio_path is None

@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_guild_message_without_mention_is_ignored(mock_parse_command, router, mock_message):
    """Test that a message in a guild without a mention is ignored."""
    # Arrange
    mock_message.guild = MagicMock()
    mock_message.channel.__class__.__name__ = 'TextChannel'
    router.bot.user.mentioned_in.return_value = False # The key for this test
    mock_message.content = "Hello? Anyone there?"

    # Act
    response = await router.dispatch_message(mock_message)

    # Assert
    assert response is None
