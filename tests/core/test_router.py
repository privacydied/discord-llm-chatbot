import pytest
from unittest.mock import MagicMock, AsyncMock

import discord

from bot.router import Router, ResponseMessage
from bot.command_parser import ParsedCommand
from bot.types import Command


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
def mock_flows():
    """Pytest fixture for mock processing flows."""
    return {
        'process_text': AsyncMock(return_value="AI response text"),
        'process_attachments': AsyncMock(return_value="AI response for attachment"),
        'generate_tts': AsyncMock(return_value="/path/to/audio.wav"),
    }


@pytest.fixture
def router(mock_bot, mock_flows):
    """Pytest fixture for a Router instance with mocked flows."""
    return Router(bot=mock_bot, flow_overrides=mock_flows)


@pytest.mark.asyncio
async def test_dispatch_text_only_message(router, mock_bot, mock_flows):
    """Verify that a simple text message is dispatched to the text processor."""
    message = MagicMock(spec=discord.Message)
    message.id = 111
    message.content = "Hello bot"
    message.attachments = []
    message.author.id = 987

    # Mock parse_command to return a CHAT command
    with pytest.MonkeyPatch.context() as m:
        m.setattr("bot.router.parse_command", MagicMock(return_value=ParsedCommand(command=Command.CHAT, cleaned_content="Hello bot")))
        response = await router.dispatch_message(message)

    # Assert the correct flow was called and response is correct
    mock_flows['process_text'].assert_called_once_with("Hello bot", "987")
    assert isinstance(response, ResponseMessage)
    assert response.text == "AI response text"
    assert response.audio_path is None


@pytest.mark.asyncio
async def test_dispatch_ping_command(router, mock_bot):
    """Verify that a !ping command is handled directly."""
    message = MagicMock(spec=discord.Message)
    message.id = 222

    with pytest.MonkeyPatch.context() as m:
        m.setattr("bot.router.parse_command", MagicMock(return_value=ParsedCommand(command=Command.PING, cleaned_content="")))
        response = await router.dispatch_message(message)

    assert isinstance(response, ResponseMessage)
    assert response.text == "Pong!"
    assert response.audio_path is None


@pytest.mark.asyncio
async def test_dispatch_tts_command(router, mock_bot, mock_flows):
    """Verify a TTS command is dispatched to the TTS generator."""
    message = MagicMock(spec=discord.Message)
    message.id = 333
    message.author.id = 987

    with pytest.MonkeyPatch.context() as m:
        m.setattr("bot.router.parse_command", MagicMock(return_value=ParsedCommand(command=Command.SAY, cleaned_content="speak this")))
        response = await router.dispatch_message(message)

    mock_flows['generate_tts'].assert_called_once_with("speak this")
    assert isinstance(response, ResponseMessage)
    assert response.text is None
    assert response.audio_path == "/path/to/audio.wav"


@pytest.mark.asyncio
async def test_dispatch_image_attachment(router, mock_bot, mock_flows):
    """Verify an image attachment is dispatched to the attachment processor."""
    message = MagicMock(spec=discord.Message)
    message.id = 444
    message.content = "what is this?"
    message.author.id = 987
    attachment = MagicMock(spec=discord.Attachment)
    attachment.content_type = 'image/png'
    attachment.filename = 'test_image.png'
    message.attachments = [attachment]

    with pytest.MonkeyPatch.context() as m:
        m.setattr("bot.router.parse_command", MagicMock(return_value=ParsedCommand(command=Command.CHAT, cleaned_content="what is this?")))
        response = await router.dispatch_message(message)

    mock_flows['process_attachments'].assert_called_once_with(message, "what is this?")
    assert isinstance(response, ResponseMessage)
    assert response.text == "AI response for attachment"
    assert response.audio_path is None


@pytest.mark.asyncio
async def test_process_document_txt(router):
    """Verify that the router can correctly process a .txt file."""
    import tempfile
    file_content = "This is a test text file."
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

    result = await router._process_document(tmp_file_path, '.txt')
    assert result == file_content

    # Clean up the temporary file
    import os
    os.remove(tmp_file_path)


@pytest.mark.asyncio
async def test_process_document_docx(router):
    """Verify that the router can correctly process a .docx file by mocking the docx library."""
    # Mock the docx.Document object
    mock_doc = MagicMock()
    mock_para1 = MagicMock()
    mock_para1.text = "Hello docx."
    mock_para2 = MagicMock()
    mock_para2.text = "This is a test."
    mock_doc.paragraphs = [mock_para1, mock_para2]

    with pytest.MonkeyPatch.context() as m:
        # Mock the import to think docx is installed
        m.setattr("bot.router.DOCX_SUPPORT", True)
        # Mock the Document class to return our mock document
        m.setattr("bot.router.docx.Document", MagicMock(return_value=mock_doc))

        result = await router._process_document("/fake/path/to/document.docx", '.docx')

    assert result == "Hello docx.\nThis is a test."


@pytest.mark.asyncio
async def test_process_document_pdf(router):
    """Verify that the router can correctly process a .pdf file by mocking the PDFProcessor."""
    # Ensure the router's PDF processor is mocked
    router.pdf_processor = MagicMock()
    router.pdf_processor.process = AsyncMock(return_value={'text': 'This is a test PDF.'})

    with pytest.MonkeyPatch.context() as m:
        m.setattr("bot.router.PDF_SUPPORT", True)
        result = await router._process_document("/fake/path/to/document.pdf", '.pdf')

    router.pdf_processor.process.assert_called_once_with("/fake/path/to/document.pdf")
    assert result == "This is a test PDF."
