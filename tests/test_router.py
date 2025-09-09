"""
Unit tests for the refactored Router class.

Ensures all modality flows (text, audio, URL, image, doc) are correctly routed,
and that the '1 IN > 1 OUT' principle is strictly enforced. Verifies command
handling for static, cog-based, and standard chat commands.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import discord
import logging

from bot.router import InputModality, Router, BotAction
from bot.command_parser import ParsedCommand
from bot.types import Command


@pytest.fixture
def mock_bot():
    """Fixture for a mocked bot instance with necessary attributes."""
    bot = MagicMock(spec=discord.Client)
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.user.mentioned_in.return_value = True
    bot.config = {'TTS_ENABLED_USERS': set(), 'TTS_ENABLED_SERVERS': set()}
    bot.tts_manager = AsyncMock()
    bot.brain = AsyncMock()
    bot.loop = AsyncMock()
    return bot

@pytest.fixture
def router(mock_bot):
    """Provides a Router instance with a mocked bot."""
    return Router(bot=mock_bot, logger=MagicMock(spec=logging.Logger))

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
    message.content = "Hello"
    return message

# --- Test Cases ---

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command_type, expected_text, should_be_none",
    [
        (Command.PING, "Pong!", False),
        (Command.HELP, "See `/help` for a list of commands.", False),
        (Command.CHAT, "Processed text", False),
        (Command.TTS, None, True),      # Cog-handled
        (Command.SAY, None, True),      # Cog-handled
        (Command.TTS_ALL, None, True), # Cog-handled
        (Command.SPEAK, None, True),   # Cog-handled
        (Command.IGNORE, None, True),   # Ignored
    ],
)
@patch("bot.router.parse_command")
async def test_command_handling(
    mock_parse_command, router, mock_message, command_type, expected_text, should_be_none
):
    """Test router's handling of static, cog, and ignored commands."""
    mock_parse_command.return_value = ParsedCommand(
        command=command_type, cleaned_content="Hello"
    )

    if command_type == Command.CHAT:
        # For a standard chat, we need to mock the full flow
        router._flows['process_text'] = AsyncMock(return_value=expected_text)
        with patch.object(router, "_get_input_modality", return_value=InputModality.TEXT_ONLY):
            response = await router.dispatch_message(mock_message)
    else:
        response = await router.dispatch_message(mock_message)

    if should_be_none:
        assert response is None, f"Expected None for command {command_type.name}, but got a response."
    else:
        assert response is not None, f"Expected a response for command {command_type.name}, but got None."
        assert response.text == expected_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "modality, flow_key, expected_output",
    [
        (InputModality.TEXT_ONLY, "process_text", "Processed text"),
        (InputModality.GENERAL_URL, "process_url", "Processed URL"),
        (InputModality.AUDIO_VIDEO_FILE, "process_audio", "Processed audio"),
        (InputModality.SINGLE_IMAGE, "process_attachments", "Processed attachments"),
        (InputModality.PDF_DOCUMENT, "process_attachments", "Processed attachments"),
    ],
)
@patch("bot.router.parse_command")
async def test_modality_flows(
    mock_parse_command,
    router,
    mock_message,
    modality,
    flow_key,
    expected_output,
):
    """Verify that each input modality is routed to the correct processing flow."""
    mock_parse_command.return_value = ParsedCommand(
        command=Command.CHAT, cleaned_content="Test content"
    )

    if modality == InputModality.GENERAL_URL:
        mock_message.content = "https://example.com"

    # Mock the specific flow method in the _flows dictionary
    mock_flow_method = AsyncMock(return_value=expected_output)
    router._flows[flow_key] = mock_flow_method

    with patch.object(router, "_get_input_modality", return_value=modality) as mock_get_modality:
        response = await router.dispatch_message(mock_message)

    mock_get_modality.assert_called_once_with(mock_message)
    mock_flow_method.assert_called_once()
    assert response is not None
    assert response.text == expected_output


@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_no_processed_text_returns_error(mock_parse_command, router, mock_message):
    """Test that if a flow returns no text, a user-friendly error is returned."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows['process_text'] = AsyncMock(return_value=None) # Simulate a flow failure

    with patch.object(router, '_get_input_modality', return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    assert response.text.startswith("Error:")


@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_exception_in_flow_returns_error(mock_parse_command, router, mock_message):
    """Test that an exception during processing returns a generic error message."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows['process_text'] = AsyncMock(side_effect=Exception("Critical failure!"))

    with patch.object(router, '_get_input_modality', return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    assert response.text.startswith("Error:")

@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_empty_string_prevention(mock_parse_command, router, mock_message):
    """Verify empty string responses are converted to error messages."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows['process_text'] = AsyncMock(return_value="")  # Empty string

    with patch.object(router, '_get_input_modality', return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    assert response.text.startswith("Error:")

@pytest.mark.asyncio

@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_error_embed_generation(mock_parse_command, router, mock_message):
    """Verify error conditions generate proper embed responses."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows['process_text'] = AsyncMock(return_value=None)  # Simulate failure

    with patch.object(router, '_get_input_modality', return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    assert hasattr(response, 'embed'), "Error responses should include an embed"
    assert response.embed.title.startswith("Error:")


@pytest.mark.asyncio
async def test_dm_plain_text_reply():
    """Test that a plain text DM returns a BotAction with content."""
    # Setup
    mock_bot = MagicMock()
    mock_logger = MagicMock()
    mock_metrics = MagicMock()
    flows = {
        'process_text': AsyncMock(return_value="Hello, world!")
    }
    router = Router(mock_bot, flows, mock_logger, mock_metrics)
    
    # Create a mock DM message
    mock_message = MagicMock()
    mock_message.channel = MagicMock()
    mock_message.channel.__class__.__name__ = "DMChannel"
    mock_message.content = "Hello"
    mock_message.attachments = []
    
    # Execute
    action = await router.dispatch_message(mock_message)
    
    # Verify
    assert isinstance(action, BotAction)
    assert action.content == "Hello, world!"
    assert not action.error

@pytest.mark.asyncio
async def test_guild_unmentioned_ignored():
    """Test that an unmentioned guild message returns None."""
    # Setup
    mock_bot = MagicMock()
    mock_logger = MagicMock()
    mock_metrics = MagicMock()
    flows = {
        'process_text': AsyncMock(return_value="Hello, world!")
    }
    router = Router(mock_bot, flows, mock_logger, mock_metrics)
    
    # Create a mock guild message without mention
    mock_message = MagicMock()
    mock_message.channel = MagicMock()
    mock_message.channel.__class__.__name__ = "TextChannel"
    mock_message.content = "Hello"
    mock_message.attachments = []
    # Simulate parse_command returning None (no mention)
    router.parse_command = MagicMock(return_value=None)
    
    # Execute
    action = await router.dispatch_message(mock_message)
    
    # Verify
    assert action is None
