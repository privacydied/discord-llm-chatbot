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
from bot.router import Router, OutputModality


@pytest.fixture
def mock_bot():
    """Fixture for a mocked bot instance with a real Router."""
    bot = MagicMock(spec=LLMBot)
    bot.user = MagicMock()
    bot.user.id = 123456789
    bot.config = MagicMock()
    bot.tts_manager = AsyncMock()
    bot.loop = AsyncMock()
    bot.tts_manager.voice = "default_voice"

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
@patch("bot.router.brain_infer", new_callable=AsyncMock)
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
    mock_brain_infer.assert_called_once_with("Hello there")


@pytest.mark.asyncio
@patch("bot.router.brain_infer", new_callable=AsyncMock)
async def test_flow_text_to_tts(mock_brain_infer, mock_bot, mock_message):
    """2. TEXT -> TTS: Test a text input returns a voice message."""
    # Arrange
    mock_message.content = "!chat Hello there"
    mock_brain_infer.return_value = "General Kenobi!"

    router = Router(bot=mock_bot)
    router._flows["generate_tts"] = AsyncMock(return_value="/tmp/fake_tts.mp3")

    with patch.object(router, "_get_output_modality", return_value=OutputModality.TTS):
        # Act
        response = await router.dispatch_message(mock_message)

    # Assert
    assert response is not None, "Router should have returned a response"
    assert response.audio_path == "/tmp/fake_tts.mp3", (
        "Response should have an audio path"
    )
    assert response.text == "General Kenobi!", "Response text is incorrect"
    router._flows["generate_tts"].assert_called_once_with("General Kenobi!")


@pytest.mark.asyncio
@patch("bot.router.brain_infer", new_callable=AsyncMock)
@patch("bot.router.see_infer", new_callable=AsyncMock)
async def test_flow_image_to_text(
    mock_see_infer, mock_brain_infer, mock_bot, mock_message
):
    """3. IMAGE -> TEXT: Test an image upload returns a text description."""
    # Arrange
    mock_attachment = MagicMock(spec=discord.Attachment)
    mock_attachment.filename = "test_image.png"
    mock_attachment.content_type = "image/png"
    mock_attachment.read = AsyncMock(return_value=b"fake_image_bytes")
    mock_message.attachments = [mock_attachment]
    mock_message.content = "What is this?"

    mock_see_infer.return_value = "a cat sitting on a table"
    mock_brain_infer.return_value = "The image shows a cat sitting on a table."

    router = Router(bot=mock_bot)

    # Act
    response = await router.dispatch_message(mock_message)

    # Assert
    assert response is not None, "Router should have returned a response"
    mock_see_infer.assert_called_once_with(
        image_data=b"fake_image_bytes",
        prompt="User uploaded an image with the prompt: 'What is this?'",
        mime_type="image/png",
    )
    mock_brain_infer.assert_called_once_with(
        "User uploaded an image with the prompt: 'What is this?'. The image contains: a cat sitting on a table"
    )
    assert response.text == "The image shows a cat sitting on a table.", (
        "Response text is incorrect"
    )
    # Ensure the attachment save method was not called for images
    mock_attachment.save.assert_not_called()
    assert response.audio_path is None, "Response should not have an audio path"


@pytest.mark.asyncio
@patch("bot.router.os.remove")
@patch("bot.router.brain_infer", new_callable=AsyncMock)
@patch("bot.router.Router._process_document", new_callable=AsyncMock)
async def test_flow_document_to_text(
    mock_process_document, mock_brain_infer, mock_os_remove, mock_bot, mock_message
):
    """4. DOCUMENT -> TEXT: Test a document upload returns a text summary."""
    # Arrange
    mock_attachment = MagicMock(spec=discord.Attachment)
    mock_attachment.filename = "test_doc.pdf"
    mock_attachment.content_type = "application/pdf"
    mock_attachment.save = AsyncMock()
    mock_message.attachments = [mock_attachment]
    mock_message.content = "Summarize this document."
    mock_process_document.return_value = "This is the document content."
    mock_brain_infer.return_value = "This is a summary of the document."

    router = Router(bot=mock_bot)

    with patch("bot.router.tempfile.NamedTemporaryFile") as mock_temp_file:
        mock_file_handle = MagicMock()
        mock_file_handle.name = "/tmp/fake_doc.pdf"
        mock_temp_file.return_value.__enter__.return_value = mock_file_handle

        # Act
        response = await router.dispatch_message(mock_message)

    # Assert
    assert response is not None, "Router should have returned a response"
    mock_attachment.save.assert_called_once_with(Path("/tmp/fake_doc.pdf"))
    mock_process_document.assert_called_once_with("/tmp/fake_doc.pdf", ".pdf")
    mock_brain_infer.assert_called_once_with(
        "DOCUMENT CONTENT:\n---\nThis is the document content.\n---\n\nUSER'S PROMPT: Summarize this document."
    )
    assert response.text == "This is a summary of the document."
    mock_os_remove.assert_called_once_with("/tmp/fake_doc.pdf")
