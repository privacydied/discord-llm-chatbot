"""
Unit tests for the refactored Router class.

Ensures all modality flows (text, audio, URL, image, doc) are correctly routed,
and that the '1 IN > 1 OUT' principle is strictly enforced. Verifies command
handling for static, cog-based, and standard chat commands.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import discord
from dataclasses import dataclass
import logging

from bot.router import InputModality, Router, ResponseMessage
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
        (InputModality.URL, "process_url", "Processed URL"),
        (InputModality.AUDIO, "process_audio", "Processed audio"),
        (InputModality.IMAGE, "process_attachments", "Processed attachments"),
        (InputModality.DOCUMENT, "process_attachments", "Processed attachments"),
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

    if modality == InputModality.URL:
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
    assert response.text == "I'm sorry, I wasn't able to process that."


@pytest.mark.asyncio
@patch('bot.router.parse_command')
async def test_exception_in_flow_returns_error(mock_parse_command, router, mock_message):
    """Test that an exception during processing returns a generic error message."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows['process_text'] = AsyncMock(side_effect=Exception("Critical failure!"))

    with patch.object(router, '_get_input_modality', return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    assert response.text == "An unexpected error occurred. Please try again later."
