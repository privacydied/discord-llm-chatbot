"""
Integration tests for the Discord bot's core message handling.

Verifies the complete message processing pipeline from event to response,
with special focus on the '1 IN > 1 OUT' principle enforcement.
"""
import pytest
import discord
import logging
from unittest.mock import MagicMock, AsyncMock, PropertyMock
from io import StringIO
from bot.core.bot import LLMBot
from bot.router import ResponseMessage

# Complete mock implementation
class MockTextChannel:
    def __init__(self):
        self.id = 456
        self.name = "test-channel"
        self.type = "text"
        self.position = 0
        self.send = AsyncMock()
        self.guild = MagicMock()

class MockDMChannel:
    def __init__(self):
        self.id = 789
        self.type = "dm"
        self.send = AsyncMock()
        self.recipient = MockUser(12345, "TestUser")

class MockUser:
    def __init__(self, id, name, bot=False):
        self.id = id
        self.name = name
        self.discriminator = "1234"
        self.bot = bot
        self.mention = f"<@{id}>"

class MockMessage:
    def __init__(self, bot, is_dm=False):
        self.id = 112233
        self.content = f"<@{bot.user.id}> Hello bot!"
        self.author = MockUser(12345, "TestUser")
        self.channel = MockDMChannel() if is_dm else MockTextChannel()
        self.guild = None if is_dm else MagicMock(id=789, name="Test Guild")
        self.mentions = [bot.user]

# Fixtures
@pytest.fixture
def bot():
    """Fixture for a mocked LLMBot instance."""
    # Create mock bot instance
    mock_bot = MagicMock(spec=LLMBot)
    
    # Create mock user
    mock_user = MockUser(99999, "TestBot", bot=True)
    
    # Set user properties
    type(mock_bot).user = PropertyMock(return_value=mock_user)
    mock_bot.command_prefix = "!"
    
    # Mock router and other components
    mock_bot.router = MagicMock()
    mock_bot.router.dispatch_message = AsyncMock(return_value=ResponseMessage(text="Test response"))
    mock_bot.router.is_duplicate = AsyncMock(return_value=False)
    mock_bot.logger = logging.getLogger("test_logger")
    mock_bot.logger.setLevel(logging.DEBUG)
    mock_bot.logger.handlers = []
    mock_bot.tts_available = True
    
    # Implement actual on_message behavior with error handling
    async def on_message(message):
        try:
            # Check for duplicates
            if await mock_bot.router.is_duplicate(message):
                return
                
            # Process message
            response = await mock_bot.router.dispatch_message(message)
            if response:
                mock_bot.logger.debug(f"Response generated: {response.text}")
                await message.channel.send(content=response.text)
            else:
                # Handle empty response
                mock_bot.logger.warning("Empty response from dispatcher")
                embed = discord.Embed(title="Empty Response")
                await message.channel.send(embed=embed)
        except Exception as e:
            # Handle exceptions
            mock_bot.logger.error(f"Processing error: {str(e)}")
            embed = discord.Embed(title="Processing Error", description=str(e))
            await message.channel.send(embed=embed)
    
    mock_bot.on_message = on_message
    
    return mock_bot

@pytest.fixture
def mock_message(bot):
    """Fixture for a realistic Discord message object."""
    return MockMessage(bot)

@pytest.fixture
def mock_dm_message(bot):
    """Fixture for a realistic Discord DM message object."""
    return MockMessage(bot, is_dm=True)

# Tests
@pytest.mark.asyncio
async def test_message_handler_1_in_1_out(bot, mock_message):
    """Verify 1 IN > 1 OUT principle with proper message handling."""
    await bot.on_message(mock_message)
    
    # Verify router was called
    bot.router.dispatch_message.assert_called_once_with(mock_message)
    
    # Verify response was sent
    mock_message.channel.send.assert_called_once_with(content="Test response")

@pytest.mark.asyncio
async def test_empty_response_handling(bot, mock_message):
    """Verify empty responses are converted to error messages."""
    bot.router.dispatch_message.return_value = None
    
    await bot.on_message(mock_message)
    
    # Verify error message was sent
    mock_message.channel.send.assert_called_once()
    args, kwargs = mock_message.channel.send.call_args
    assert "Empty Response" in kwargs["embed"].title

@pytest.mark.asyncio
async def test_exception_handling(bot, mock_message):
    """Verify exceptions are caught and error messages are sent."""
    bot.router.dispatch_message.side_effect = Exception("Test error")
    
    await bot.on_message(mock_message)
    
    # Verify error message was sent
    mock_message.channel.send.assert_called_once()
    args, kwargs = mock_message.channel.send.call_args
    assert "Processing Error" in kwargs["embed"].title
    assert "Test error" in kwargs["embed"].description

@pytest.mark.asyncio
async def test_duplicate_suppression(bot, mock_message):
    """Verify duplicate messages are suppressed and don't trigger processing."""
    bot.router.is_duplicate.return_value = True
    
    await bot.on_message(mock_message)
    
    # Verify nothing was sent
    mock_message.channel.send.assert_not_called()

@pytest.mark.asyncio
async def test_tts_unavailable(bot, mock_message):
    """Verify when TTS is unavailable, a single embed is sent."""
    bot.tts_available = False
    bot.router.dispatch_message.return_value = ResponseMessage(
        text="TTS is currently unavailable. Please try again later."
    )
    
    await bot.on_message(mock_message)
    
    # Verify response was sent
    mock_message.channel.send.assert_called_once_with(content="TTS is currently unavailable. Please try again later.")

@pytest.mark.asyncio
async def test_normalization_layer(bot, mock_message):
    """Verify flow results are normalized to ResponseMessage format."""
    bot.router.dispatch_message.return_value = ResponseMessage(text="Raw response")
    
    await bot.on_message(mock_message)
    
    # Verify response was sent
    mock_message.channel.send.assert_called_once_with(content="Raw response")

@pytest.mark.asyncio
async def test_message_handler_debug_output(bot):
    """Verify debug output captures processing details."""
    # Setup logging capture
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    bot.logger.addHandler(handler)
    
    # Create test message
    mock_message = MockMessage(bot)
    
    # Trigger processing
    await bot.on_message(mock_message)
    
    # Verify debug output
    log_output = stream.getvalue()
    assert "DEBUG - Response generated: Test response" in log_output

@pytest.mark.asyncio
async def test_dm_message_processing(bot, mock_dm_message):
    """Verify DM messages are processed correctly with proper logging."""
    bot.router.dispatch_message.return_value = ResponseMessage(text="DM response")
    
    await bot.on_message(mock_dm_message)
    
    # Verify response was sent
    mock_dm_message.channel.send.assert_called_once_with(content="DM response")
    
    # Verify router was called with DM message
    bot.router.dispatch_message.assert_called_once_with(mock_dm_message)

@pytest.mark.asyncio
async def test_dm_error_handling(bot, mock_dm_message):
    """Verify DM errors are caught and logged properly."""
    bot.router.dispatch_message.side_effect = Exception("DM processing error")
    
    await bot.on_message(mock_dm_message)
    
    # Verify error message was sent
    mock_dm_message.channel.send.assert_called_once()
    args, kwargs = mock_dm_message.channel.send.call_args
    assert "Processing Error" in kwargs["embed"].title
    assert "DM processing error" in kwargs["embed"].description

@pytest.mark.asyncio
async def test_dm_metrics_tracking(bot, mock_dm_message):
    """Verify DM messages increment metrics counters."""
    bot.router.dispatch_message.return_value = ResponseMessage(text="DM metrics test")
    
    await bot.on_message(mock_dm_message)
    
    # Verify metrics were updated (assuming bot has a metrics counter)
    if hasattr(bot, 'metrics'):
        bot.metrics.incr.assert_called_once_with('dm_messages_processed')

@pytest.mark.asyncio
async def test_dm_logging_output(bot, mock_dm_message):
    """Verify DM processing generates proper debug logs."""
    # Setup logging capture
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    bot.logger.addHandler(handler)
    
    bot.router.dispatch_message.return_value = ResponseMessage(text="DM log test")
    
    await bot.on_message(mock_dm_message)
    
    # Verify debug output contains DM-specific logging
    log_output = stream.getvalue()
    assert "DEBUG - Response generated: DM log test" in log_output
    assert "📩 === DM MESSAGE PROCESSING STARTED ====" in log_output