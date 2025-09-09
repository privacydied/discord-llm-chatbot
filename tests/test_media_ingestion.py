"""
Tests for unified media ingestion system.
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from bot.media_ingestion import (
    MediaIngestionManager,
    MediaIngestionResult,
    create_media_ingestion_manager
)
from bot.media_capability import ProbeResult
from bot.action import BotAction


class TestMediaIngestionResult:
    """Test MediaIngestionResult dataclass."""
    
    def test_init_success(self):
        """Test successful result initialization."""
        result = MediaIngestionResult(
            success=True,
            content="Test content",
            metadata={'title': 'Test Video'},
            source_type="media",
            processing_time_ms=150.0
        )
        
        assert result.success is True
        assert result.content == "Test content"
        assert result.metadata == {'title': 'Test Video'}
        assert result.error_message is None
        assert result.fallback_triggered is False
        assert result.source_type == "media"
        assert result.processing_time_ms == 150.0
    
    def test_init_failure(self):
        """Test failure result initialization."""
        result = MediaIngestionResult(
            success=False,
            error_message="Processing failed",
            fallback_triggered=True,
            source_type="scrape",
            processing_time_ms=50.0
        )
        
        assert result.success is False
        assert result.content is None
        assert result.metadata is None
        assert result.error_message == "Processing failed"
        assert result.fallback_triggered is True
        assert result.source_type == "scrape"
        assert result.processing_time_ms == 50.0


class TestMediaIngestionManager:
    """Test suite for MediaIngestionManager."""
    
    @pytest.fixture
    def mock_bot(self):
        """Create mock bot instance."""
        bot = MagicMock()
        bot.config = MagicMock()
        bot.context_manager = AsyncMock()
        bot.context_manager.get_context_string.return_value = "Previous conversation context"
        bot.system_prompts = {"VL_PROMPT_FILE": "Describe this image"}
        bot.enhanced_context_manager = None
        bot.metrics = None
        return bot
    
    @pytest.fixture
    def mock_message(self):
        """Create mock Discord message."""
        message = MagicMock()
        message.id = 12345
        return message
    
    @pytest.fixture
    def manager(self, mock_bot):
        """Create MediaIngestionManager instance."""
        return MediaIngestionManager(mock_bot)
    
    def test_init(self, manager, mock_bot):
        """Test manager initialization."""
        assert manager.bot == mock_bot
        assert manager.config == mock_bot.config
        assert manager.logger is not None
        assert manager._retry_delays == {}
    
    def test_sanitize_metadata_basic(self, manager):
        """Test basic metadata sanitization."""
        raw_metadata = {
            'title': 'Test Video Title',
            'uploader': 'Test Channel',
            'source': 'youtube',
            'duration_seconds': 120.5,
            'upload_date': '2023-01-01',
            'url': 'https://youtube.com/watch?v=test123',
            'unsafe_field': 'should be ignored'
        }
        
        sanitized = manager._sanitize_metadata(raw_metadata)
        
        assert sanitized['title'] == 'Test Video Title'
        assert sanitized['uploader'] == 'Test Channel'
        assert sanitized['source'] == 'youtube'
        assert sanitized['duration_seconds'] == 120.5
        assert sanitized['upload_date'] == '2023-01-01'
        assert sanitized['url'] == 'https://youtube.com/watch?v=test123'
        assert 'unsafe_field' not in sanitized
    
    def test_sanitize_metadata_length_limits(self, manager):
        """Test metadata length limiting."""
        raw_metadata = {
            'title': 'A' * 300,  # Too long
            'uploader': 'B' * 150,  # Too long
            'url': 'https://example.com/' + 'C' * 600  # Too long
        }
        
        sanitized = manager._sanitize_metadata(raw_metadata)
        
        assert len(sanitized['title']) <= 203  # 200 + "..."
        assert sanitized['title'].endswith('...')
        assert len(sanitized['uploader']) <= 103  # 100 + "..."
        assert sanitized['uploader'].endswith('...')
        assert len(sanitized['url']) <= 503  # 500 + "..."
        assert sanitized['url'].endswith('...')
    
    def test_sanitize_metadata_control_characters(self, manager):
        """Test control character removal."""
        raw_metadata = {
            'title': 'Test\x00Video\x01Title\x1f',  # Contains control chars
            'uploader': 'Channel\nName\tWith\rWhitespace'  # Contains allowed whitespace
        }
        
        sanitized = manager._sanitize_metadata(raw_metadata)
        
        assert sanitized['title'] == 'TestVideoTitle'  # Control chars removed
        assert sanitized['uploader'] == 'Channel\nName\tWith\rWhitespace'  # Whitespace preserved
    
    def test_sanitize_metadata_empty_or_none(self, manager):
        """Test handling of empty or None metadata."""
        assert manager._sanitize_metadata(None) == {}
        assert manager._sanitize_metadata({}) == {}
    
    @pytest.mark.asyncio
    async def test_extract_media_with_retry_success(self, manager):
        """Test successful media extraction."""
        url = "https://youtube.com/watch?v=test123"
        
        mock_result = {
            'transcription': 'Test transcription',
            'metadata': {'title': 'Test Video', 'source': 'youtube'}
        }
        
        with patch('bot.media_ingestion.hear_infer_from_url', return_value=mock_result):
            success, result, error = await manager._extract_media_with_retry(url)
        
        assert success is True
        assert result == mock_result
        assert error is None
    
    @pytest.mark.asyncio
    async def test_extract_media_with_retry_timeout(self, manager):
        """Test media extraction timeout."""
        url = "https://youtube.com/watch?v=test123"
        
        with patch('bot.media_ingestion.hear_infer_from_url', side_effect=asyncio.TimeoutError):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
                success, result, error = await manager._extract_media_with_retry(url)
        
        assert success is False
        assert result is None
        assert "timeout" in error.lower()
    
    @pytest.mark.asyncio
    async def test_extract_media_with_retry_max_attempts(self, manager):
        """Test retry logic reaches max attempts."""
        url = "https://youtube.com/watch?v=test123"
        
        with patch('bot.media_ingestion.hear_infer_from_url', side_effect=Exception("Download failed")):
            with patch('asyncio.sleep'):  # Speed up test by mocking sleep
                success, result, error = await manager._extract_media_with_retry(url)
        
        assert success is False
        assert result is None
        assert "Download failed" in error
    
    def test_build_media_context_full_metadata(self, manager):
        """Test media context building with full metadata."""
        transcription = "This is the video transcription."
        metadata = {
            'source': 'youtube',
            'title': 'Test Video',
            'uploader': 'Test Channel',
            'duration_seconds': 120.5,
            'speedup_factor': 1.5
        }
        url = "https://youtube.com/watch?v=test123"
        
        context = manager._build_media_context(transcription, metadata, url)
        
        assert "youtube video" in context.lower()
        assert "Test Video" in context
        assert "Test Channel" in context
        assert "120.5s" in context
        assert "1.5x speed" in context
        assert transcription in context
    
    def test_build_media_context_minimal_metadata(self, manager):
        """Test media context building with minimal metadata."""
        transcription = "This is the video transcription."
        metadata = {}
        url = "https://youtube.com/watch?v=test123"
        
        context = manager._build_media_context(transcription, metadata, url)
        
        assert url in context
        assert transcription in context
    
    def test_build_media_context_no_transcription(self, manager):
        """Test media context building with no transcription."""
        transcription = ""
        metadata = {'source': 'youtube', 'title': 'Test Video'}
        url = "https://youtube.com/watch?v=test123"
        
        context = manager._build_media_context(transcription, metadata, url)
        
        assert "youtube video" in context.lower()
        assert "Test Video" in context
        assert "No audio transcription was available" in context
    
    @pytest.mark.asyncio
    async def test_process_media_path_success(self, manager, mock_message):
        """Test successful media path processing."""
        url = "https://youtube.com/watch?v=test123"
        
        mock_extract_result = {
            'transcription': 'Test transcription',
            'metadata': {'title': 'Test Video', 'source': 'youtube'}
        }
        
        with patch.object(manager, '_extract_media_with_retry', return_value=(True, mock_extract_result, None)):
            result = await manager._process_media_path(url, mock_message)
        
        assert result.success is True
        assert result.content is not None
        assert "Test transcription" in result.content
        assert result.metadata is not None
        assert result.source_type == "media"
        assert result.processing_time_ms is not None
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_process_media_path_failure(self, manager, mock_message):
        """Test failed media path processing."""
        url = "https://youtube.com/watch?v=test123"
        
        with patch.object(manager, '_extract_media_with_retry', return_value=(False, None, "Download failed")):
            result = await manager._process_media_path(url, mock_message)
        
        assert result.success is False
        assert result.error_message == "Download failed"
        assert result.source_type == "media"
        assert result.processing_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_process_fallback_path_success_with_screenshot(self, manager, mock_message):
        """Test successful fallback processing with screenshot."""
        url = "https://twitter.com/user/status/123"
        fallback_reason = "media extraction failed"
        
        mock_web_result = {
            'screenshot_path': '/path/to/screenshot.png',
            'text': None,
            'error': None
        }
        
        with patch('bot.media_ingestion.web.process_url', return_value=mock_web_result):
            result = await manager._process_fallback_path(url, mock_message, fallback_reason)
        
        assert result.success is True
        assert "Screenshot available at:" in result.content
        assert result.fallback_triggered is True
        assert result.source_type == "scrape"
        assert result.metadata['fallback_reason'] == fallback_reason
    
    @pytest.mark.asyncio
    async def test_process_fallback_path_success_with_text(self, manager, mock_message):
        """Test successful fallback processing with text content."""
        url = "https://example.com/article"
        fallback_reason = "domain not whitelisted"
        
        mock_web_result = {
            'screenshot_path': None,
            'text': 'Article content here',
            'error': None
        }
        
        with patch('bot.media_ingestion.web.process_url', return_value=mock_web_result):
            result = await manager._process_fallback_path(url, mock_message, fallback_reason)
        
        assert result.success is True
        assert result.content == 'Article content here'
        assert result.fallback_triggered is True
        assert result.source_type == "scrape"
    
    @pytest.mark.asyncio
    async def test_process_fallback_path_web_error(self, manager, mock_message):
        """Test fallback processing with web error."""
        url = "https://example.com/broken"
        fallback_reason = "media extraction failed"
        
        mock_web_result = {
            'screenshot_path': None,
            'text': None,
            'error': 'Page not found'
        }
        
        with patch('bot.media_ingestion.web.process_url', return_value=mock_web_result):
            result = await manager._process_fallback_path(url, mock_message, fallback_reason)
        
        assert result.success is False
        assert result.error_message == 'Page not found'
        assert result.fallback_triggered is True
        assert result.source_type == "scrape"
    
    @pytest.mark.asyncio
    async def test_process_url_smart_media_capable_success(self, manager, mock_message):
        """Test smart URL processing with successful media path."""
        url = "https://youtube.com/watch?v=test123"
        
        mock_probe_result = ProbeResult(
            is_media_capable=True,
            reason='media available',
            cached=False,
            probe_duration_ms=100.0
        )
        
        mock_media_result = MediaIngestionResult(
            success=True,
            content="Media context content",
            metadata={'title': 'Test Video'},
            source_type="media",
            processing_time_ms=200.0
        )
        
        mock_bot_action = BotAction(content="Generated response")
        
        with patch('bot.media_ingestion.media_detector.is_media_capable', return_value=mock_probe_result):
            with patch.object(manager, '_process_media_path', return_value=mock_media_result):
                with patch.object(manager, '_create_bot_action_from_media', return_value=mock_bot_action):
                    result = await manager.process_url_smart(url, mock_message)
        
        assert result == mock_bot_action
    
    @pytest.mark.asyncio
    async def test_process_url_smart_media_capable_fallback(self, manager, mock_message):
        """Test smart URL processing with media failure and successful fallback."""
        url = "https://youtube.com/watch?v=test123"
        
        mock_probe_result = ProbeResult(
            is_media_capable=True,
            reason='media available',
            cached=False,
            probe_duration_ms=100.0
        )
        
        mock_media_result = MediaIngestionResult(
            success=False,
            error_message="Download failed",
            source_type="media"
        )
        
        mock_fallback_result = MediaIngestionResult(
            success=True,
            content="Fallback content",
            fallback_triggered=True,
            source_type="scrape"
        )
        
        mock_bot_action = BotAction(content="Fallback response")
        
        with patch('bot.media_ingestion.media_detector.is_media_capable', return_value=mock_probe_result):
            with patch.object(manager, '_process_media_path', return_value=mock_media_result):
                with patch.object(manager, '_process_fallback_path', return_value=mock_fallback_result):
                    with patch.object(manager, '_create_bot_action_from_fallback', return_value=mock_bot_action):
                        result = await manager.process_url_smart(url, mock_message)
        
        assert result == mock_bot_action
    
    @pytest.mark.asyncio
    async def test_process_url_smart_not_media_capable(self, manager, mock_message):
        """Test smart URL processing for non-media-capable URL."""
        url = "https://example.com/article"
        
        mock_probe_result = ProbeResult(
            is_media_capable=False,
            reason='domain not whitelisted',
            cached=False,
            probe_duration_ms=5.0
        )
        
        mock_fallback_result = MediaIngestionResult(
            success=True,
            content="Article content",
            fallback_triggered=True,
            source_type="scrape"
        )
        
        mock_bot_action = BotAction(content="Article response")
        
        with patch('bot.media_ingestion.media_detector.is_media_capable', return_value=mock_probe_result):
            with patch.object(manager, '_process_fallback_path', return_value=mock_fallback_result):
                with patch.object(manager, '_create_bot_action_from_fallback', return_value=mock_bot_action):
                    result = await manager.process_url_smart(url, mock_message)
        
        assert result == mock_bot_action
    
    @pytest.mark.asyncio
    async def test_process_url_smart_both_paths_fail(self, manager, mock_message):
        """Test smart URL processing when both media and fallback paths fail."""
        url = "https://youtube.com/watch?v=broken"
        
        mock_probe_result = ProbeResult(
            is_media_capable=True,
            reason='media available',
            cached=False
        )
        
        mock_media_result = MediaIngestionResult(
            success=False,
            error_message="Media extraction failed"
        )
        
        mock_fallback_result = MediaIngestionResult(
            success=False,
            error_message="Web scraping failed",
            fallback_triggered=True
        )
        
        with patch('bot.media_ingestion.media_detector.is_media_capable', return_value=mock_probe_result):
            with patch.object(manager, '_process_media_path', return_value=mock_media_result):
                with patch.object(manager, '_process_fallback_path', return_value=mock_fallback_result):
                    result = await manager.process_url_smart(url, mock_message)
        
        assert isinstance(result, BotAction)
        assert result.error is True
        assert "Could not process URL" in result.content
    
    @pytest.mark.asyncio
    async def test_create_bot_action_from_media_basic_brain(self, manager, mock_message):
        """Test creating bot action from media result using basic brain inference."""
        media_result = MediaIngestionResult(
            success=True,
            content="Media context content",
            metadata={'title': 'Test Video'}
        )
        
        mock_brain_result = BotAction(content="AI response to media")
        
        with patch('bot.media_ingestion.brain_infer', return_value=mock_brain_result):
            result = await manager._create_bot_action_from_media(media_result, mock_message)
        
        assert result == mock_brain_result
    
    @pytest.mark.asyncio
    async def test_create_bot_action_from_media_contextual_brain(self, manager, mock_message):
        """Test creating bot action from media result using contextual brain."""
        # Enable contextual brain
        manager.bot.enhanced_context_manager = MagicMock()
        
        media_result = MediaIngestionResult(
            success=True,
            content="Media context content",
            metadata={'title': 'Test Video'}
        )
        
        mock_contextual_response = "Contextual AI response"
        
        with patch.dict(os.environ, {'USE_ENHANCED_CONTEXT': 'true'}):
            with patch('bot.media_ingestion.contextual_brain_infer_simple', return_value=mock_contextual_response):
                result = await manager._create_bot_action_from_media(media_result, mock_message)
        
        assert isinstance(result, BotAction)
        assert result.content == mock_contextual_response
    
    @pytest.mark.asyncio
    async def test_create_bot_action_from_fallback_screenshot(self, manager, mock_message):
        """Test creating bot action from fallback result with screenshot."""
        fallback_result = MediaIngestionResult(
            success=True,
            content="Screenshot available at: /path/to/screenshot.png",
            fallback_triggered=True,
            source_type="scrape"
        )
        
        mock_vision_response = MagicMock()
        mock_vision_response.content = "Vision analysis of screenshot"
        mock_vision_response.error = False
        
        mock_brain_result = BotAction(content="Final response")
        
        with patch('bot.media_ingestion.see_infer', return_value=mock_vision_response):
            with patch('bot.media_ingestion.brain_infer', return_value=mock_brain_result):
                result = await manager._create_bot_action_from_fallback(fallback_result, mock_message)
        
        assert result == mock_brain_result
    
    @pytest.mark.asyncio
    async def test_create_bot_action_from_fallback_text(self, manager, mock_message):
        """Test creating bot action from fallback result with text content."""
        fallback_result = MediaIngestionResult(
            success=True,
            content="Article text content",
            fallback_triggered=True,
            source_type="scrape"
        )
        
        # Mock router with text flow
        mock_router = MagicMock()
        mock_router._invoke_text_flow = AsyncMock(return_value=BotAction(content="Text flow response"))
        manager.bot.router = mock_router
        
        result = await manager._create_bot_action_from_fallback(fallback_result, mock_message)
        
        assert isinstance(result, BotAction)
        assert result.content == "Text flow response"
        mock_router._invoke_text_flow.assert_called_once()


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_media_ingestion_manager(self):
        """Test factory function creates manager correctly."""
        mock_bot = MagicMock()
        
        manager = create_media_ingestion_manager(mock_bot)
        
        assert isinstance(manager, MediaIngestionManager)
        assert manager.bot == mock_bot


@pytest.mark.integration
class TestMediaIngestionIntegration:
    """Integration tests for media ingestion system."""
    
    @pytest.fixture
    def integration_bot(self):
        """Create bot-like object for integration testing."""
        bot = MagicMock()
        bot.config = MagicMock()
        bot.context_manager = AsyncMock()
        bot.context_manager.get_context_string.return_value = ""
        bot.system_prompts = {}
        bot.enhanced_context_manager = None
        bot.metrics = None
        return bot
    
    @pytest.fixture
    def integration_message(self):
        """Create message-like object for integration testing."""
        message = MagicMock()
        message.id = 99999
        return message
    
    @pytest.mark.asyncio
    async def test_end_to_end_youtube_url_mock(self, integration_bot, integration_message):
        """Test end-to-end processing of YouTube URL with mocked components."""
        manager = MediaIngestionManager(integration_bot)
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        # Mock the capability detector
        mock_probe_result = ProbeResult(
            is_media_capable=True,
            reason='media available',
            cached=False,
            probe_duration_ms=100.0
        )
        
        # Mock the media extraction
        mock_extraction_result = {
            'transcription': 'Never gonna give you up, never gonna let you down...',
            'metadata': {
                'title': 'Rick Astley - Never Gonna Give You Up',
                'uploader': 'Rick Astley',
                'source': 'youtube',
                'duration_seconds': 213.0
            }
        }
        
        # Mock brain inference
        mock_brain_result = BotAction(content="This is the famous Rick Roll video!")
        
        with patch('bot.media_ingestion.media_detector.is_media_capable', return_value=mock_probe_result):
            with patch.object(manager, '_extract_media_with_retry', return_value=(True, mock_extraction_result, None)):
                with patch('bot.media_ingestion.brain_infer', return_value=mock_brain_result):
                    result = await manager.process_url_smart(url, integration_message)
        
        assert isinstance(result, BotAction)
        assert result.content == "This is the famous Rick Roll video!"
        assert result.error is not True
    
    @pytest.mark.asyncio
    async def test_end_to_end_fallback_to_scraping(self, integration_bot, integration_message):
        """Test end-to-end fallback to web scraping for non-media URL."""
        manager = MediaIngestionManager(integration_bot)
        url = "https://example.com/article"
        
        # Mock capability detector saying not media-capable
        mock_probe_result = ProbeResult(
            is_media_capable=False,
            reason='domain not whitelisted',
            cached=False,
            probe_duration_ms=5.0
        )
        
        # Mock web scraping result
        mock_web_result = {
            'text': 'This is an example article with some content.',
            'screenshot_path': None,
            'error': None
        }
        
        # Mock brain inference
        mock_brain_result = BotAction(content="This article discusses...")
        
        with patch('bot.media_ingestion.media_detector.is_media_capable', return_value=mock_probe_result):
            with patch('bot.media_ingestion.web.process_url', return_value=mock_web_result):
                with patch('bot.media_ingestion.brain_infer', return_value=mock_brain_result):
                    result = await manager.process_url_smart(url, integration_message)
        
        assert isinstance(result, BotAction)
        assert result.content == "This article discusses..."
        assert result.error is not True
