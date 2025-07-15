"""
Integration tests for the four canonical user-to-bot interaction flows.

Ensures that:
1. TEXT -> TEXT
2. TEXT -> TTS
3. IMAGE -> TEXT
4. DOCUMENT -> TEXT

... all work as expected and respect the '1 IN > 1 OUT' principle.
"""
import pytest
import discord
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from bot.core.bot import LLMBot
from bot.router import Router
from bot.command_parser import Command


@pytest.fixture
def mock_bot():
    """Fixture for a mocked bot instance with a real Router."""
    bot = MagicMock(spec=LLMBot)
    bot.user = MagicMock()
    bot.user.id = 123456789
    bot.config = MagicMock()
    bot.tts_manager = AsyncMock()
    bot.loop = AsyncMock()
    bot.tts_manager.voice = 'default_voice'

    # Use a real router instance attached to the mock bot
    router = Router(bot=bot)
    bot.router = router
    return bot


@pytest.fixture
def mock_message():
    """Fixture for a generic mocked Discord message."""
    message = AsyncMock(spec=discord.Message)
    message.id = 987654321
    message.author = MagicMock(spec=discord.Member)
    message.author.id = 12345
    message.guild = None  # Default to DM
    message.attachments = []
    message.content = ""
    return message


@pytest.mark.asyncio
@patch('bot.router.brain_infer', new_callable=AsyncMock)
async def test_flow_text_to_text(mock_brain_infer, mock_bot, mock_message):
    """1. TEXT -> TEXT: Test a standard text command returns a text response."""
    # Arrange
    mock_message.content = "!chat Hello there"
    mock_brain_infer.return_value = "General Kenobi!"

    # Act
    response = await mock_bot.router.dispatch_message(mock_message)

    # Assert
    assert response is not None, "Router should have returned a response"
    assert response.text == "General Kenobi!", "Response text is incorrect"
    assert response.audio_path is None, "Response should not have an audio path"
    mock_brain_infer.assert_called_once_with("Hello there", '12345')


@pytest.mark.asyncio
@patch('bot.router.brain_infer', new_callable=AsyncMock)
async def test_flow_text_to_tts(mock_brain_infer, mock_bot, mock_message):
    """2. TEXT -> TTS: Test a !say command returns a TTS audio response."""
    # Arrange
    mock_message.content = "!say Hello there"
    mock_brain_infer.return_value = "General Kenobi!"
    mock_bot.tts_manager.generate_tts.return_value = "/path/to/test.wav"

    # Act
    response = await mock_bot.router.dispatch_message(mock_message)

    # Assert
    assert response is not None, "Router should have returned a response"
    assert response.text is None, "!say command should not have a text response"
    assert response.audio_path == "/path/to/test.wav", "Response audio path is incorrect"
    mock_brain_infer.assert_called_once_with("Hello there", '12345')
    mock_bot.tts_manager.generate_tts.assert_called_once_with("General Kenobi!", 'default_voice')


@pytest.mark.asyncio
@patch('bot.router.see_infer', new_callable=AsyncMock)
async def test_flow_image_to_text(mock_see_infer, mock_bot, mock_message):
    """3. IMAGE -> TEXT: Test an image upload returns a text description."""
    # Arrange
    mock_attachment = MagicMock(spec=discord.Attachment)
    mock_attachment.filename = 'test_image.png'
    mock_attachment.url = 'http://example.com/test_image.png'
    mock_message.attachments = [mock_attachment]
    mock_message.content = "What is this?"
    mock_see_infer.return_value = "This is a test image."

    # Act
    response = await mock_bot.router.dispatch_message(mock_message)

    # Assert
    assert response is not None, "Router should have returned a response"
    assert response.text == "This is a test image.", "Response text is incorrect"
    assert response.audio_path is None, "Response should not have an audio path"
    mock_see_infer.assert_called_once()


@pytest.mark.asyncio
@patch('bot.router.Router._process_document', new_callable=AsyncMock)
async def test_flow_document_to_text(mock_process_document, mock_bot, mock_message):
    """4. DOCUMENT -> TEXT: Test a document upload returns a text summary."""
    # Arrange
    mock_attachment = MagicMock(spec=discord.Attachment)
    mock_attachment.filename = 'test_doc.txt'
    mock_attachment.url = 'http://example.com/test_doc.txt'
    mock_attachment.save = AsyncMock()
    mock_message.attachments = [mock_attachment]
    mock_message.content = "Summarize this document."
    mock_process_document.return_value = "This is the document content."

    with patch('bot.router.tempfile.NamedTemporaryFile') as mock_temp_file:
        # Mock the context manager for NamedTemporaryFile
        mock_file_handle = MagicMock()
        mock_file_handle.name = '/tmp/fake_doc.txt'
        mock_temp_file.return_value.__enter__.return_value = mock_file_handle

        # Act
        response = await mock_bot.router.dispatch_message(mock_message)

    # Assert
    assert response is not None, "Router should have returned a response"
    # The final response is generated by brain_infer after the document is read
    # For this test, we check that the document processing was called correctly.
    # A full integration test would also patch brain_infer.
    mock_attachment.save.assert_called_once_with(Path('/tmp/fake_doc.txt'))
    mock_process_document.assert_called_once_with('/tmp/fake_doc.txt', '.txt')
