"""
Tests for empty-body messages with attachments and !speak/!say commands with empty input.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import discord
from pathlib import Path

from bot.router import Router, ResponseMessage
from bot.command_parser import Command, ParsedCommand


@pytest.fixture
def mock_bot():
    """Pytest fixture for a mock bot instance."""
    bot = MagicMock(name="LLMBot")
    bot.user = MagicMock(spec=discord.ClientUser)
    bot.user.id = 1234567890
    bot.config = MagicMock()
    bot.tts_manager = MagicMock()
    bot.loop = AsyncMock()
    return bot


@pytest.fixture
def router(mock_bot):
    """Pytest fixture for a Router instance with mocked dependencies."""
    router = Router(mock_bot)
    router._flows = {
        "process_text": AsyncMock(return_value="AI response for text"),
        "process_url": AsyncMock(return_value="AI response for URL"),
        "process_audio": AsyncMock(return_value="AI response for audio"),
        "process_attachments": AsyncMock(return_value="AI response for attachment"),
        "generate_tts": AsyncMock(return_value="/tmp/audio.wav"),
    }
    return router


@pytest.mark.asyncio
async def test_empty_dm_with_image_attachment(router, mock_bot):
    """Test that an empty DM with only an image attachment triggers VL processing."""
    # Create a mock message with empty content but with an image attachment
    message = MagicMock(spec=discord.Message)
    message.id = 555
    message.content = ""  # Empty content
    message.author.id = 987
    message.guild = None  # DM channel

    # Create a mock channel that's a DMChannel
    channel = MagicMock(spec=discord.DMChannel)
    message.channel = channel

    # Create a mock attachment
    attachment = MagicMock(spec=discord.Attachment)
    attachment.content_type = "image/jpeg"
    attachment.filename = "empty_test_image.jpg"
    message.attachments = [attachment]

    # Mock the command parser to return a CHAT command with empty content
    with patch(
        "bot.router.parse_command",
        return_value=ParsedCommand(command=Command.CHAT, cleaned_content=""),
    ):
        # Call the router's dispatch_message method
        response = await router.dispatch_message(message)

    # Verify that process_attachments was called with the message and empty content
    router._flows["process_attachments"].assert_called_once_with(message, "")

    # Verify the response
    assert isinstance(response, ResponseMessage)
    assert response.text == "AI response for attachment"
    assert len(response.text.split()) >= 5  # Ensure caption has at least 5 words
    assert response.audio_path is None


@pytest.mark.asyncio
async def test_speak_command_with_no_text_and_history(router, mock_bot):
    """Test that !speak with no text falls back to previous message and returns audio."""
    from bot.commands.tts_cmds import TTSCommands

    # Create a mock context
    ctx = MagicMock()
    ctx.author.id = 123
    ctx.message.id = 789

    # Create mock message history with a previous message
    previous_message = MagicMock()
    previous_message.id = 456
    previous_message.author.id = 123  # Same author as current message
    previous_message.content = "This is a previous message"

    # Mock the channel history to return our previous message
    ctx.channel.history.return_value.__aiter__.return_value = [previous_message]

    # Create TTS commands instance with mocked dependencies
    tts_cog = TTSCommands(mock_bot)
    tts_cog.router = router

    # Mock the TTS manager to return a valid audio path
    mock_bot.tts_manager.is_available.return_value = True
    mock_bot.tts_manager.generate_tts = AsyncMock(return_value="/tmp/test_audio.wav")
    mock_bot.tts_manager.voice = "default"

    # Call the speak command with no text
    await tts_cog.say(ctx, text=None)

    # Verify TTS was generated with the previous message content
    mock_bot.tts_manager.generate_tts.assert_called_once_with(
        "This is a previous message", "default"
    )

    # Verify the response was sent with the audio file
    ctx.send.assert_called_once()
    call_args = ctx.send.call_args[1]
    assert "file" in call_args
    assert isinstance(call_args["file"], discord.File)
    assert call_args["file"].filename == "test_audio.wav"


@pytest.mark.asyncio
async def test_speak_command_with_no_text_and_no_history(router, mock_bot):
    """Test that !speak with no text and no message history returns an error embed."""
    from bot.commands.tts_cmds import TTSCommands

    # Create a mock context
    ctx = MagicMock()
    ctx.author.id = 123
    ctx.message.id = 789

    # Mock empty channel history (no previous messages)
    ctx.channel.history.return_value.__aiter__.return_value = []

    # Create TTS commands instance with mocked dependencies
    tts_cog = TTSCommands(mock_bot)
    tts_cog.router = router

    # Mock the TTS manager
    mock_bot.tts_manager.is_available.return_value = True

    # Call the speak command with no text
    await tts_cog.say(ctx, text=None)

    # Verify an error message was sent
    ctx.send.assert_called_once_with(
        "❌ Please provide text to speak or send a message before using !say"
    )

    # Verify TTS was not generated
    mock_bot.tts_manager.generate_tts.assert_not_called()


@pytest.mark.asyncio
async def test_missing_import_smoke_test():
    """Test that the os module is properly imported in openai_backend.py."""
    import importlib.util
    import sys

    # Get the path to the openai_backend.py file
    file_path = Path(__file__).parent.parent.parent / "bot" / "openai_backend.py"
    assert file_path.exists(), "openai_backend.py file not found"

    # Load the module
    spec = importlib.util.spec_from_file_location("openai_backend", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["openai_backend"] = module
    spec.loader.exec_module(module)

    # Check that the os module is imported
    assert hasattr(module, "os"), "os module is not imported in openai_backend.py"

    # Check that the get_base64_image function exists
    assert hasattr(module, "get_base64_image"), "get_base64_image function not found"
