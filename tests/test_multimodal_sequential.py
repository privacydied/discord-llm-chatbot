"""
Comprehensive tests for sequential multimodal processing.
Verifies the 1 IN → 1 OUT rule, retry logic, and result aggregation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from discord import Message, Attachment, Embed
from bot.router import Router
from bot.modality import InputModality, InputItem, collect_input_items
from bot.result_aggregator import ResultAggregator
from bot.multimodal_retry import run_with_retries


@pytest.fixture
def mock_bot():
    """Create a mock bot instance."""
    bot = MagicMock()
    bot.user.id = 12345
    bot.user.mention = "<@12345>"
    bot.config = {}
    bot.context_manager.get_context_string = AsyncMock(return_value="mock context")
    return bot


@pytest.fixture
def router(mock_bot):
    """Create a router instance with mocked flows."""
    flow_overrides = {"process_text": AsyncMock(return_value=None)}
    return Router(mock_bot, flow_overrides)


@pytest.mark.asyncio
async def test_two_images_sequential(router):
    """Test that two image attachments are processed sequentially, not in parallel."""
    # Create mock message with two image attachments
    message = MagicMock(spec=Message)
    message.id = 123
    message.content = "Analyze these images"
    message.mentions = []
    message.attachments = [
        MagicMock(spec=Attachment, filename="image1.jpg", content_type="image/jpeg"),
        MagicMock(spec=Attachment, filename="image2.png", content_type="image/png"),
    ]
    message.embeds = []

    # Track processing order
    processing_order = []

    async def mock_handle_image(item):
        processing_order.append(item.payload.filename)
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Analysis of {item.payload.filename}: This is a test image."

    # Mock the handler
    router._handle_image = mock_handle_image

    # Mock text flow to capture final aggregated result
    final_prompt = None

    async def capture_text_flow(prompt, message, context):
        nonlocal final_prompt
        final_prompt = prompt

    router._invoke_text_flow = capture_text_flow

    # Process the message
    await router._process_multimodal_message_internal(message, "test context")

    # Verify sequential processing (not parallel)
    assert processing_order == ["image1.jpg", "image2.png"]

    # Verify single aggregated response
    assert final_prompt is not None
    assert "image1.jpg" in final_prompt
    assert "image2.png" in final_prompt
    assert "Analysis of image1.jpg" in final_prompt
    assert "Analysis of image2.png" in final_prompt
    assert "[1/2]" in final_prompt and "[2/2]" in final_prompt


@pytest.mark.asyncio
async def test_mixed_modalities_order(router):
    """Test that mixed modalities (image + PDF + URL + image) are processed in appearance order."""
    # Create mock message with mixed content
    message = MagicMock(spec=Message)
    message.id = 456
    message.content = "Process https://example.com/test and analyze everything"
    message.mentions = []
    message.attachments = [
        MagicMock(spec=Attachment, filename="doc.pdf", content_type="application/pdf"),
        MagicMock(spec=Attachment, filename="final.jpg", content_type="image/jpeg"),
    ]
    message.embeds = []

    # Track processing order and types
    processing_log = []

    async def mock_handle_general_url(item):
        processing_log.append(("URL", item.payload))
        return f"URL content from {item.payload}"

    async def mock_handle_pdf(item):
        processing_log.append(("PDF", item.payload.filename))
        return f"PDF content from {item.payload.filename}"

    async def mock_handle_image(item):
        processing_log.append(("IMAGE", item.payload.filename))
        return f"Image analysis of {item.payload.filename}"

    # Mock handlers
    router._handle_general_url = mock_handle_general_url
    router._handle_pdf = mock_handle_pdf
    router._handle_image = mock_handle_image

    # Mock text flow
    final_prompt = None

    async def capture_text_flow(prompt, message, context):
        nonlocal final_prompt
        final_prompt = prompt

    router._invoke_text_flow = capture_text_flow

    # Process the message
    await router._process_multimodal_message_internal(message, "test context")

    # Verify processing order matches appearance order: URL, PDF, Image
    expected_order = [
        ("URL", "https://example.com/test"),
        ("PDF", "doc.pdf"),
        ("IMAGE", "final.jpg"),
    ]
    assert processing_log == expected_order

    # Verify aggregated result contains all sections in sequence
    assert final_prompt is not None
    assert "[1/3]" in final_prompt
    assert "[2/3]" in final_prompt
    assert "[3/3]" in final_prompt
    assert "URL content from https://example.com/test" in final_prompt
    assert "PDF content from doc.pdf" in final_prompt
    assert "Image analysis of final.jpg" in final_prompt


@pytest.mark.asyncio
async def test_retries_then_success():
    """Test that retry logic works: first two attempts fail, third succeeds."""
    call_count = 0

    async def failing_handler(item):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError(f"Attempt {call_count} failed")
        return "Success on attempt 3"

    # Test the retry utility directly
    result = await run_with_retries(
        failing_handler,
        MagicMock(payload=MagicMock(filename="test.jpg")),
        retries=3,
        base_delay=0.01,  # Fast for testing
        max_delay=0.1,
        timeout=1.0,
    )

    assert call_count == 3
    assert result == "Success on attempt 3"


@pytest.mark.asyncio
async def test_retries_then_fail():
    """Test that after all retries fail, a clear failure message is returned."""

    async def always_failing_handler(item):
        raise TimeoutError("Handler always times out")

    # Test the retry utility
    result = await run_with_retries(
        always_failing_handler,
        MagicMock(payload=MagicMock(filename="test.jpg")),
        retries=3,
        base_delay=0.01,
        max_delay=0.1,
        timeout=0.1,
    )

    # Should return error message, not raise exception
    assert "❌ Processing failed after 3 attempts" in result
    assert "TimeoutError" in result


@pytest.mark.asyncio
async def test_no_parallelism():
    """Test that only one handler is active at any time."""
    active_handlers = set()
    max_concurrent = 0

    async def tracking_handler(item):
        handler_id = id(asyncio.current_task())
        active_handlers.add(handler_id)
        nonlocal max_concurrent
        max_concurrent = max(max_concurrent, len(active_handlers))

        await asyncio.sleep(0.1)  # Simulate work

        active_handlers.remove(handler_id)
        return f"Processed {item.payload.filename}"

    # Create message with multiple items
    message = MagicMock(spec=Message)
    message.id = 789
    message.content = ""
    message.mentions = []
    message.attachments = [
        MagicMock(spec=Attachment, filename=f"file{i}.jpg", content_type="image/jpeg")
        for i in range(5)
    ]
    message.embeds = []

    # Create router and mock handler
    mock_bot = MagicMock()
    mock_bot.user.id = 12345
    mock_bot.context_manager.get_context_string = AsyncMock(return_value="context")

    router = Router(mock_bot)
    router._handle_image = tracking_handler
    router._invoke_text_flow = AsyncMock()

    # Process message
    await router._process_multimodal_message_internal(message, "context")

    # Verify no parallelism occurred
    assert max_concurrent == 1, (
        f"Expected max 1 concurrent handler, got {max_concurrent}"
    )


@pytest.mark.asyncio
async def test_text_only_passthrough(router):
    """Test that plain text messages are routed to text flow unchanged."""
    message = MagicMock(spec=Message)
    message.id = 999
    message.content = "Just plain text, no attachments"
    message.mentions = []
    message.attachments = []
    message.embeds = []

    # Mock text flow to capture input
    captured_text = None

    async def capture_text_flow(text, message, context):
        nonlocal captured_text
        captured_text = text

    router._invoke_text_flow = capture_text_flow

    # Process message
    await router._process_multimodal_message_internal(message, "context")

    # Verify text passed through unchanged
    assert captured_text == "Just plain text, no attachments"


def test_result_aggregator_formatting():
    """Test that ResultAggregator formats results correctly."""
    aggregator = ResultAggregator()

    # Add some mock results
    mock_item1 = InputItem("attachment", MagicMock(filename="test.jpg"), 0)
    mock_item2 = InputItem("url", "https://example.com/doc.pdf", 1)

    aggregator.add_result(
        0,
        mock_item1,
        InputModality.SINGLE_IMAGE,
        "This is an image of a cat.",
        True,
        1.5,
        1,
    )
    aggregator.add_result(
        1,
        mock_item2,
        InputModality.PDF_DOCUMENT,
        "Document contains technical specs.",
        True,
        2.1,
        2,
    )

    # Generate aggregated prompt
    prompt = aggregator.get_aggregated_prompt("Original message text")

    # Verify formatting
    assert "I processed 2 inputs from your message:" in prompt
    assert "[1/2] ✅ Image: test.jpg (1.5s)" in prompt
    assert "[2/2] ✅ PDF Document: doc.pdf (2.1s, 2 attempts)" in prompt
    assert "This is an image of a cat." in prompt
    assert "Document contains technical specs." in prompt
    assert "### Original Message Text:" in prompt
    assert "Original message text" in prompt


def test_collect_input_items_order():
    """Test that collect_input_items preserves order correctly."""
    message = MagicMock(spec=Message)
    message.content = "Check https://example.com and https://test.org"
    message.attachments = [
        MagicMock(spec=Attachment, filename="file1.pdf"),
        MagicMock(spec=Attachment, filename="file2.jpg"),
    ]
    message.embeds = [MagicMock(spec=Embed)]

    items = collect_input_items(message)

    # Should be: URL1, URL2, attachment1, attachment2, embed1
    assert len(items) == 5
    assert items[0].source_type == "url" and items[0].payload == "https://example.com"
    assert items[1].source_type == "url" and items[1].payload == "https://test.org"
    assert (
        items[2].source_type == "attachment"
        and items[2].payload.filename == "file1.pdf"
    )
    assert (
        items[3].source_type == "attachment"
        and items[3].payload.filename == "file2.jpg"
    )
    assert items[4].source_type == "embed"

    # Verify order indices
    for i, item in enumerate(items):
        assert item.order_index == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
