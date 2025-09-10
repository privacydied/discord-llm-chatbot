"""
Change Summary:
- Created comprehensive test suite for sequential multimodal processing
- Tests mixed-modality batches (images + PDFs + URLs + videos) with proper handler invocation
- Tests timeout scenarios with asyncio.TimeoutError simulation
- Tests error recovery with exception handling and user feedback
- Tests no-input scenarios (text-only messages)
- Uses pytest-asyncio for async test support
- Mocks all external dependencies (vision, STT, PDF processing, web scraping)
- Validates sequential processing order and _flow_process_text integration

This test suite ensures the refactored multimodal system processes all input types
sequentially without skipping any modalities, with proper error handling and user feedback.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from discord import Message, Attachment, Embed, DMChannel

from bot.router import Router
from bot.modality import (
    InputModality,
    InputItem,
    collect_input_items,
    map_item_to_modality,
)


@pytest.fixture
def mock_bot():
    """Create a mock Discord bot for testing."""
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.user.in_message = MagicMock(return_value=False)
    bot.config = MagicMock()
    bot.tts_manager = MagicMock()
    bot.loop = asyncio.get_event_loop()
    bot.context_manager = MagicMock()
    bot.context_manager.get_context_string = AsyncMock(return_value="Test context")
    return bot


@pytest.fixture
def router(mock_bot):
    """Create a Router instance for testing."""
    return Router(mock_bot)


@pytest.fixture
def mock_message():
    """Create a mock Discord message."""
    message = MagicMock(spec=Message)
    message.id = 67890
    message.content = "Test message"
    message.attachments = []
    message.embeds = []
    message.mentions = []
    message.channel = MagicMock(spec=DMChannel)
    message.channel.typing = AsyncMock()
    message.reply = AsyncMock()
    message.guild = None
    return message


@pytest.fixture
def mock_attachment():
    """Create a mock Discord attachment."""
    attachment = MagicMock(spec=Attachment)
    attachment.filename = "test.jpg"
    attachment.content_type = "image/jpeg"
    attachment.save = AsyncMock()
    return attachment


@pytest.fixture
def mock_pdf_attachment():
    """Create a mock PDF attachment."""
    attachment = MagicMock(spec=Attachment)
    attachment.filename = "document.pdf"
    attachment.content_type = "application/pdf"
    attachment.save = AsyncMock()
    return attachment


class TestMultimodalSequence:
    """Test suite for sequential multimodal processing."""

    @pytest.mark.asyncio
    async def test_mixed_attachments_and_urls(
        self, router, mock_message, mock_attachment, mock_pdf_attachment
    ):
        """Test processing message with mixed attachments and URLs."""
        # Setup message with multiple modalities
        mock_message.content = "Check this out: https://youtube.com/watch?v=test123 and https://example.com"
        mock_message.attachments = [mock_attachment, mock_pdf_attachment]

        # Mock all handlers to return predictable results
        with (
            patch.object(
                router, "_handle_video_url", new_callable=AsyncMock
            ) as mock_video,
            patch.object(
                router, "_handle_general_url", new_callable=AsyncMock
            ) as mock_url,
            patch.object(router, "_handle_image", new_callable=AsyncMock) as mock_image,
            patch.object(router, "_handle_pdf", new_callable=AsyncMock) as mock_pdf,
            patch.object(
                router, "_flow_process_text", new_callable=AsyncMock
            ) as mock_text_flow,
        ):
            mock_video.return_value = "Video transcription: Hello world"
            mock_url.return_value = "Web content: Example website"
            mock_image.return_value = "Image analysis: A beautiful photo"
            mock_pdf.return_value = "PDF content: Important document"

            # Execute multimodal processing
            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Verify all handlers were called in correct order
            mock_video.assert_called_once()
            mock_url.assert_called_once()
            mock_image.assert_called_once()
            mock_pdf.assert_called_once()

            # Verify _flow_process_text was called for each result plus remaining text
            assert mock_text_flow.call_count == 5  # 4 handlers + remaining text

            # Check that all handler results were processed
            call_args = [call[0][0] for call in mock_text_flow.call_args_list]
            assert "Video transcription: Hello world" in call_args
            assert "Web content: Example website" in call_args
            assert "Image analysis: A beautiful photo" in call_args
            assert "PDF content: Important document" in call_args

    @pytest.mark.asyncio
    async def test_attachment_plus_embed(self, router, mock_message, mock_attachment):
        """Test processing message with attachment and embed."""
        # Setup message with attachment and embed
        mock_message.attachments = [mock_attachment]
        mock_embed = MagicMock(spec=Embed)
        mock_embed.image = MagicMock()
        mock_embed.image.url = "https://example.com/embed_image.jpg"
        mock_message.embeds = [mock_embed]

        with (
            patch.object(router, "_handle_image", new_callable=AsyncMock) as mock_image,
            patch.object(
                router, "_flow_process_text", new_callable=AsyncMock
            ) as mock_text_flow,
        ):
            mock_image.return_value = "Image processed"

            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Should call image handler twice (attachment + embed)
            assert mock_image.call_count == 2
            # Should call text flow twice for handlers + once for remaining text
            assert mock_text_flow.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self, router, mock_message, mock_attachment):
        """Test timeout handling for slow handlers."""
        mock_message.attachments = [mock_attachment]

        with patch.object(
            router, "_handle_image", new_callable=AsyncMock
        ) as mock_image:
            # Simulate timeout by making handler sleep longer than timeout
            async def slow_handler(item):
                await asyncio.sleep(60)  # Longer than 30s timeout
                return "Should not reach here"

            mock_image.side_effect = slow_handler

            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Verify timeout error reply was sent
            mock_message.reply.assert_called_once()
            reply_call = mock_message.reply.call_args[0][0]
            assert "timed out" in reply_call.lower()
            assert "single_image" in reply_call.lower()

    @pytest.mark.asyncio
    async def test_error_recovery(
        self, router, mock_message, mock_attachment, mock_pdf_attachment
    ):
        """Test error recovery when one handler fails."""
        mock_message.attachments = [mock_attachment, mock_pdf_attachment]

        with (
            patch.object(router, "_handle_image", new_callable=AsyncMock) as mock_image,
            patch.object(router, "_handle_pdf", new_callable=AsyncMock) as mock_pdf,
            patch.object(
                router, "_flow_process_text", new_callable=AsyncMock
            ) as mock_text_flow,
        ):
            # First handler fails, second succeeds
            mock_image.side_effect = ValueError("Image processing failed")
            mock_pdf.return_value = "PDF content: Success"

            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Verify error reply was sent for failed handler
            mock_message.reply.assert_called_once()
            reply_call = mock_message.reply.call_args[0][0]
            assert "error occurred" in reply_call.lower()
            assert "single_image" in reply_call.lower()

            # Verify successful handler still processed
            mock_pdf.assert_called_once()
            # Should be called twice: once for PDF handler result, once for remaining text
            assert mock_text_flow.call_count == 2
            # Verify PDF content was processed
            call_args = [call[0][0] for call in mock_text_flow.call_args_list]
            assert "PDF content: Success" in call_args

    @pytest.mark.asyncio
    async def test_no_inputs_text_only(self, router, mock_message):
        """Test processing of text-only message with no attachments or URLs."""
        mock_message.content = "Just a simple text message"
        mock_message.attachments = []
        mock_message.embeds = []

        with patch.object(
            router, "_invoke_text_flow", new_callable=AsyncMock
        ) as mock_text_flow:
            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Should call text flow once with the message content
            mock_text_flow.assert_called_once_with(
                "Just a simple text message", mock_message, "test context"
            )

    @pytest.mark.asyncio
    async def test_mention_stripping(self, router, mock_message, mock_bot):
        """Test that bot mentions are properly stripped from text content."""
        mock_message.content = f"<@{mock_bot.user.id}> Hello bot!"
        mock_message.mentions = [mock_bot.user]

        with patch.object(
            router, "_invoke_text_flow", new_callable=AsyncMock
        ) as mock_text_flow:
            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Should strip mention and process remaining text
            mock_text_flow.assert_called_once_with(
                "Hello bot!", mock_message, "test context"
            )

    @pytest.mark.asyncio
    async def test_url_stripping_from_text(self, router, mock_message):
        """Test that URLs are stripped from text content after being processed separately."""
        mock_message.content = "Check this https://example.com and this text"

        with (
            patch.object(
                router, "_handle_general_url", new_callable=AsyncMock
            ) as mock_url,
            patch.object(
                router, "_flow_process_text", new_callable=AsyncMock
            ) as mock_text_flow,
        ):
            mock_url.return_value = "Web content processed"

            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Should process URL separately and remaining text
            mock_url.assert_called_once()

            # Should be called twice: once for URL handler result, once for remaining text
            assert mock_text_flow.call_count == 2

            # Check that URL handler result was processed
            call_args = [call[0][0] for call in mock_text_flow.call_args_list]
            assert "Web content processed" in call_args

            # Check that final text processing call strips URLs
            final_call = mock_text_flow.call_args_list[-1]
            final_text = final_call[0][0]
            assert "Check this" in final_text and "and this text" in final_text

    @pytest.mark.asyncio
    async def test_empty_handler_results(self, router, mock_message, mock_attachment):
        """Test handling of empty results from handlers."""
        mock_message.attachments = [mock_attachment]

        with (
            patch.object(router, "_handle_image", new_callable=AsyncMock) as mock_image,
            patch.object(
                router, "_flow_process_text", new_callable=AsyncMock
            ) as mock_text_flow,
        ):
            # Handler returns empty result
            mock_image.return_value = ""

            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Should call text flow once for remaining text (empty handler result is skipped)
            assert mock_text_flow.call_count == 1
            # Verify it's processing the remaining text content
            call_args = mock_text_flow.call_args_list[0][0]
            assert call_args[0] == "Test message"

    @pytest.mark.asyncio
    async def test_sequential_processing_order(self, router, mock_message):
        """Test that items are processed in correct sequential order."""
        mock_message.content = "URL: https://example.com"
        mock_attachment1 = MagicMock(spec=Attachment)
        mock_attachment1.filename = "image1.jpg"
        mock_attachment1.content_type = "image/jpeg"
        mock_attachment2 = MagicMock(spec=Attachment)
        mock_attachment2.filename = "doc.pdf"
        mock_attachment2.content_type = "application/pdf"
        mock_message.attachments = [mock_attachment1, mock_attachment2]

        call_order = []

        async def track_url_call(item):
            call_order.append("url")
            return "URL processed"

        async def track_image_call(item):
            call_order.append("image")
            return "Image processed"

        async def track_pdf_call(item):
            call_order.append("pdf")
            return "PDF processed"

        with (
            patch.object(router, "_handle_general_url", side_effect=track_url_call),
            patch.object(router, "_handle_image", side_effect=track_image_call),
            patch.object(router, "_handle_pdf", side_effect=track_pdf_call),
            patch.object(router, "_flow_process_text", new_callable=AsyncMock),
        ):
            await router._process_multimodal_message_internal(
                mock_message, "test context"
            )

            # Verify processing order: URLs first, then attachments in order
            assert call_order == ["url", "image", "pdf"]


class TestModalityDetection:
    """Test suite for modality detection and item collection."""

    def test_collect_input_items_mixed(self, mock_message, mock_attachment):
        """Test collection of mixed input items."""
        mock_message.content = (
            "Check https://example.com and https://youtube.com/watch?v=123"
        )
        mock_message.attachments = [mock_attachment]
        mock_embed = MagicMock(spec=Embed)
        mock_message.embeds = [mock_embed]

        items = collect_input_items(mock_message)

        assert len(items) == 4  # 2 URLs + 1 attachment + 1 embed
        assert items[0].source_type == "url"
        assert items[1].source_type == "url"
        assert items[2].source_type == "attachment"
        assert items[3].source_type == "embed"

    @pytest.mark.asyncio
    async def test_map_item_to_modality_attachment(self, mock_attachment):
        """Test modality mapping for attachments."""
        item = InputItem("attachment", mock_attachment, 0)
        modality = await map_item_to_modality(item)
        assert modality == InputModality.SINGLE_IMAGE

    @pytest.mark.asyncio
    async def test_map_item_to_modality_video_url(self):
        """Test modality mapping for video URLs."""
        item = InputItem("url", "https://youtube.com/watch?v=test123", 0)
        modality = await map_item_to_modality(item)
        assert modality == InputModality.VIDEO_URL

    @pytest.mark.asyncio
    async def test_map_item_to_modality_general_url(self):
        """Test modality mapping for general URLs."""
        item = InputItem("url", "https://example.com/page", 0)
        modality = await map_item_to_modality(item)
        assert modality == InputModality.GENERAL_URL

    @pytest.mark.asyncio
    async def test_map_item_to_modality_pdf_url(self):
        """Test modality mapping for PDF URLs."""
        item = InputItem("url", "https://example.com/document.pdf", 0)
        modality = await map_item_to_modality(item)
        assert modality == InputModality.PDF_DOCUMENT

    @pytest.mark.asyncio
    async def test_map_item_to_modality_unknown(self):
        """Test modality mapping for unknown items."""
        item = InputItem("unknown_type", "unknown_payload", 0)
        modality = await map_item_to_modality(item)
        assert modality == InputModality.UNKNOWN


class TestHandlerMethods:
    """Test suite for individual handler methods."""

    @pytest.mark.asyncio
    async def test_handle_image_attachment(self, router, mock_attachment):
        """Test image handler with attachment."""
        item = InputItem("attachment", mock_attachment, 0)

        with patch("bot.router.see_infer") as mock_see:
            mock_see.return_value = MagicMock(content="A beautiful image", error=False)

            result = await router._handle_image(item)

            assert "Image analysis:" in result
            assert "A beautiful image" in result

    @pytest.mark.asyncio
    async def test_handle_video_url(self, router):
        """Test video URL handler."""
        item = InputItem("url", "https://youtube.com/watch?v=test123", 0)

        with patch("bot.router.hear_infer_from_url") as mock_hear:
            mock_hear.return_value = "Hello from video"

            result = await router._handle_video_url(item)

            assert "Video transcription" in result
            assert "Hello from video" in result

    @pytest.mark.asyncio
    async def test_handle_pdf_attachment(self, router, mock_pdf_attachment):
        """Test PDF handler with attachment."""
        router.pdf_processor = MagicMock()
        router.pdf_processor.process = AsyncMock(return_value="PDF text content")

        item = InputItem("attachment", mock_pdf_attachment, 0)

        result = await router._handle_pdf(item)

        assert "PDF content" in result
        assert "PDF text content" in result

    @pytest.mark.asyncio
    async def test_handle_general_url(self, router):
        """Test general URL handler."""
        item = InputItem("url", "https://example.com", 0)

        with patch("bot.router.process_url") as mock_process:
            mock_process.return_value = "Web page content"

            result = await router._handle_general_url(item)

            assert "Web content" in result
            assert "Web page content" in result

    @pytest.mark.asyncio
    async def test_handle_unknown(self, router):
        """Test unknown item handler."""
        item = InputItem("unknown", "unknown_payload", 0)

        result = await router._handle_unknown(item)

        assert "Unsupported input type" in result
        assert "unknown" in result
