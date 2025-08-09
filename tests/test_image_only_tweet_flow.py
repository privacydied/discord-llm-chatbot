"""
Comprehensive test suite for Twitter image-only tweet processing flow.
Tests cover unit, integration, and fault-injection scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
import json

from bot.router import Router
from bot.commands.image_upgrade_commands import ImageUpgradeCommands, ImageUpgradeManager


class TestImageOnlyTweetFlow:
    """Unit tests for Twitter image-only tweet processing."""
    
    @pytest.fixture
    def mock_bot(self):
        """Create mock bot instance with necessary attributes."""
        bot = Mock()
        bot.config = {
            'TWITTER_IMAGE_ONLY_ENABLE': 'true',
            'VISION_CAPTION_STYLE': 'neutral',
            'VISION_OCR_ENABLE': 'true',
            'VISION_OCR_MAX_CHARS': '500',
            'VISION_TAGS_ENABLE': 'true',
            'VISION_SAFETY_FILTER': 'true',
            'REPLY_TONE': 'neutral',
            'IMAGE_UPGRADE_REACTIONS': '🖼️,🔎,🏷️,🧠,↩️',
            'ECHO_TOXIC_USER_TERMS': 'false',
            'TWITTER_NORMALIZE_EMPTY_TEXT': 'true',
        }
        bot.logger = Mock()
        bot.get_channel = Mock()
        return bot
    
    @pytest.fixture 
    def router(self, mock_bot):
        """Create router instance with mocked dependencies."""
        router = Router(mock_bot)
        router.logger = Mock()
        router.metrics = Mock()
        return router
    
    @pytest.fixture
    def sample_image_only_tweet(self):
        """Sample syndication data for image-only tweet."""
        return {
            "text": "",  # Empty text indicates image-only
            "full_text": "",
            "photos": [
                {
                    "url": "https://pbs.twimg.com/media/test1.jpg",
                    "width": 1200,
                    "height": 800
                },
                {
                    "url": "https://pbs.twimg.com/media/test2.jpg", 
                    "width": 800,
                    "height": 600
                }
            ],
            "user": {
                "screen_name": "testuser",
                "name": "Test User"
            },
            "created_at": "2024-01-01T12:00:00Z"
        }
    
    @pytest.fixture
    def sample_mixed_content_tweet(self):
        """Sample syndication data for tweet with both text and images."""
        return {
            "text": "Check out this amazing view!",
            "full_text": "Check out this amazing view! #photography",
            "photos": [
                {
                    "url": "https://pbs.twimg.com/media/view.jpg",
                    "width": 1600,
                    "height": 900
                }
            ],
            "user": {
                "screen_name": "photographer",
                "name": "Photo Grapher"
            },
            "created_at": "2024-01-01T15:30:00Z"
        }

    def test_is_image_only_tweet_detection(self, router, sample_image_only_tweet, sample_mixed_content_tweet):
        """Test detection of image-only tweets vs mixed content."""
        # Test image-only tweet (empty text + photos)
        assert router._is_image_only_tweet(sample_image_only_tweet) == True
        
        # Test mixed content tweet (text + photos)
        assert router._is_image_only_tweet(sample_mixed_content_tweet) == False
        
        # Test text-only tweet (no photos)
        text_only = {"text": "Just text", "photos": []}
        assert router._is_image_only_tweet(text_only) == False
        
        # Test edge cases
        whitespace_only = {"text": "   \n  ", "photos": [{"url": "test.jpg"}]}
        assert router._is_image_only_tweet(whitespace_only) == True
        
        no_photos = {"text": "", "photos": None}
        assert router._is_image_only_tweet(no_photos) == False

    @pytest.mark.asyncio
    async def test_handle_image_only_tweet_single_image(self, router, sample_image_only_tweet):
        """Test processing single image in image-only tweet."""
        # Mock vision API response
        vision_response = {
            "alt_text": "A serene mountain landscape with snow-capped peaks",
            "ocr_text": "PEAK SUMMIT 2024",
            "safety_flags": [],
            "confidence": 0.95
        }
        
        with patch.object(router, '_vl_describe_image_from_url', new_callable=AsyncMock) as mock_vision:
            mock_vision.return_value = json.dumps(vision_response)
            
            with patch.object(router, '_build_neutral_vision_prompt') as mock_prompt:
                mock_prompt.return_value = "Describe this image neutrally"
                
                with patch.object(router, '_parse_vision_analysis') as mock_parse:
                    mock_parse.return_value = (
                        "A serene mountain landscape with snow-capped peaks",
                        "PEAK SUMMIT 2024", 
                        []
                    )
                    
                    # Test single image processing
                    single_image_tweet = {**sample_image_only_tweet, "photos": [sample_image_only_tweet["photos"][0]]}
                    
                    result = await router._handle_image_only_tweet(
                        "https://twitter.com/testuser/status/123456789",
                        single_image_tweet
                    )
                    
                    # Verify result structure
                    assert "📷 Image Analysis" in result
                    assert "A serene mountain landscape with snow-capped peaks" in result
                    assert "PEAK SUMMIT 2024" in result
                    assert "@testuser" in result
                    assert "https://twitter.com/testuser/status/123456789" in result
                    
                    # Verify vision API was called correctly
                    mock_vision.assert_called_once()
                    call_args = mock_vision.call_args
                    assert "https://pbs.twimg.com/media/test1.jpg" in str(call_args)

    @pytest.mark.asyncio
    async def test_handle_image_only_tweet_multiple_images(self, router, sample_image_only_tweet):
        """Test processing multiple images in image-only tweet."""
        vision_responses = [
            {
                "alt_text": "Mountain landscape with snow-capped peaks",
                "ocr_text": "PEAK SUMMIT 2024",
                "safety_flags": [],
                "confidence": 0.95
            },
            {
                "alt_text": "Forest trail with autumn foliage",
                "ocr_text": "",
                "safety_flags": [],
                "confidence": 0.88
            }
        ]
        
        with patch.object(router, '_vl_describe_image_from_url', new_callable=AsyncMock) as mock_vision:
            mock_vision.side_effect = [json.dumps(resp) for resp in vision_responses]
            
            with patch.object(router, '_build_neutral_vision_prompt') as mock_prompt:
                mock_prompt.return_value = "Describe this image neutrally"
                
                with patch.object(router, '_parse_vision_analysis') as mock_parse:
                    mock_parse.side_effect = [
                        ("Mountain landscape with snow-capped peaks", "PEAK SUMMIT 2024", []),
                        ("Forest trail with autumn foliage", "", [])
                    ]
                    
                    result = await router._handle_image_only_tweet(
                        "https://twitter.com/testuser/status/123456789",
                        sample_image_only_tweet
                    )
                    
                    # Verify both images were processed
                    assert "📷 Images Analysis (2)" in result
                    assert "Mountain landscape with snow-capped peaks" in result
                    assert "Forest trail with autumn foliage" in result
                    assert "PEAK SUMMIT 2024" in result
                    
                    # Verify vision API called for each image
                    assert mock_vision.call_count == 2

    @pytest.mark.asyncio
    async def test_toxic_content_filtering(self, router):
        """Test that toxic content is filtered from vision responses."""
        toxic_tweet = {
            "text": "",
            "photos": [{"url": "https://pbs.twimg.com/media/toxic.jpg"}],
            "user": {"screen_name": "baduser"},
            "created_at": "2024-01-01T12:00:00Z"
        }
        
        # Mock vision response with toxic content
        toxic_response = {
            "alt_text": "Offensive content that should be filtered",
            "ocr_text": "HATE SPEECH TEXT",
            "safety_flags": ["toxic", "offensive"],
            "confidence": 0.92
        }
        
        with patch.object(router, '_vl_describe_image_from_url', new_callable=AsyncMock) as mock_vision:
            mock_vision.return_value = json.dumps(toxic_response)
            
            with patch.object(router, '_parse_vision_analysis') as mock_parse:
                # Mock filtered response
                mock_parse.return_value = (
                    "⚠️ Content flagged by safety filters",
                    "",
                    ["toxic", "offensive"]
                )
                
                result = await router._handle_image_only_tweet(
                    "https://twitter.com/baduser/status/987654321",
                    toxic_tweet
                )
                
                # Verify toxic content was filtered
                assert "⚠️ Content flagged by safety filters" in result
                assert "HATE SPEECH TEXT" not in result
                assert "Offensive content" not in result


class TestImageUpgradeSystem:
    """Tests for emoji-driven image upgrade functionality."""
    
    @pytest.fixture
    def mock_bot(self):
        bot = Mock()
        bot.config = {
            'IMAGE_UPGRADE_REACTIONS': '🖼️,🔎,🏷️,🧠,↩️',
            'VISION_CAPTION_STYLE': 'detailed',
        }
        bot.logger = Mock()
        bot.get_channel = Mock(return_value=Mock())
        return bot
    
    @pytest.fixture
    def upgrade_manager(self, mock_bot):
        return ImageUpgradeManager(mock_bot)
    
    @pytest.fixture
    def upgrade_commands(self, mock_bot):
        return ImageUpgradeCommands(mock_bot)
    
    @pytest.fixture
    def mock_reaction_payload(self):
        payload = Mock()
        payload.message_id = 123456789
        payload.channel_id = 987654321
        payload.user_id = 111222333
        payload.emoji = Mock()
        payload.emoji.name = "🖼️"
        return payload

    @pytest.mark.asyncio
    async def test_upgrade_context_caching(self, upgrade_manager):
        """Test upgrade context is properly cached and retrieved."""
        message_id = 123456789
        url = "https://twitter.com/test/status/123"
        syn_data = {"photos": [{"url": "test.jpg"}]}
        source = "syndication"
        original_analysis = ["Test analysis"]
        
        # Store context
        await upgrade_manager.store_upgrade_context(message_id, url, syn_data, source, original_analysis)
        
        # Retrieve context
        retrieved = upgrade_manager.get_upgrade_context(message_id)
        assert retrieved is not None
        assert retrieved["url"] == url
        assert retrieved["syn_data"] == syn_data
        assert retrieved["source"] == source
        assert retrieved["original_analysis"] == original_analysis

    @pytest.mark.asyncio
    async def test_detailed_caption_upgrade(self, upgrade_manager, mock_reaction_payload):
        """Test 🖼️ detailed caption upgrade."""
        # Setup upgrade context
        context = {
            "url": "https://twitter.com/test/status/123",
            "syndication_data": {
                "photos": [{"url": "https://pbs.twimg.com/media/test.jpg"}]
            },
            "source": "syndication"
        }
        upgrade_manager.store_upgrade_context(mock_reaction_payload.message_id, context)
        
        # Mock detailed vision response
        detailed_response = {
            "alt_text": "Detailed analysis: A majestic snow-covered mountain peak rising against a clear blue sky, with ancient pine trees in the foreground and wisps of clouds around the summit",
            "ocr_text": "MOUNT EVEREST EXPEDITION 2024",
            "tags": ["mountain", "snow", "landscape", "nature"],
            "confidence": 0.97
        }
        
        with patch.object(upgrade_manager, '_get_detailed_vision_analysis', new_callable=AsyncMock) as mock_detailed:
            mock_detailed.return_value = detailed_response
            
            result = await upgrade_manager.handle_upgrade_reaction(mock_reaction_payload)
            
            assert "🖼️ **Detailed Caption**" in result
            assert "majestic snow-covered mountain peak" in result
            assert "MOUNT EVEREST EXPEDITION 2024" in result
            mock_detailed.assert_called_once()

    @pytest.mark.asyncio 
    async def test_ocr_upgrade(self, upgrade_manager, mock_reaction_payload):
        """Test 🔎 OCR details upgrade."""
        mock_reaction_payload.emoji.name = "🔎"
        
        context = {
            "url": "https://twitter.com/test/status/123",
            "syndication_data": {
                "photos": [{"url": "https://pbs.twimg.com/media/document.jpg"}]
            },
            "source": "syndication"
        }
        upgrade_manager.store_upgrade_context(mock_reaction_payload.message_id, context)
        
        ocr_response = {
            "ocr_text": "INVOICE #INV-2024-001\nDate: January 15, 2024\nAmount: $1,250.00\nDue: February 15, 2024",
            "confidence": 0.94,
            "languages": ["en"]
        }
        
        with patch.object(upgrade_manager, '_get_ocr_analysis', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = ocr_response
            
            result = await upgrade_manager.handle_upgrade_reaction(mock_reaction_payload)
            
            assert "🔎 **OCR Text Details**" in result
            assert "INVOICE #INV-2024-001" in result
            assert "$1,250.00" in result
            mock_ocr.assert_called_once()

    @pytest.mark.asyncio
    async def test_thread_context_upgrade(self, upgrade_manager, mock_reaction_payload):
        """Test ↩️ thread context upgrade."""
        mock_reaction_payload.emoji.name = "↩️"
        
        context = {
            "url": "https://twitter.com/test/status/123", 
            "syndication_data": {"photos": [{"url": "test.jpg"}]},
            "source": "syndication"
        }
        upgrade_manager.store_upgrade_context(mock_reaction_payload.message_id, context)
        
        thread_context = {
            "thread_tweets": [
                {"text": "Starting a new photography series...", "user": "photographer"},
                {"text": "This is the first shot from yesterday's hike", "user": "photographer"}
            ],
            "total_tweets": 5,
            "thread_start": "2024-01-01T10:00:00Z"
        }
        
        with patch.object(upgrade_manager, '_get_thread_context', new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = thread_context
            
            result = await upgrade_manager.handle_upgrade_reaction(mock_reaction_payload)
            
            assert "↩️ **Thread Context**" in result
            assert "photography series" in result
            assert "first shot from yesterday's hike" in result
            mock_thread.assert_called_once()


class TestFaultInjection:
    """Fault injection tests for robustness."""
    
    @pytest.fixture
    def router(self, mock_bot):
        with patch('bot.router.load_config', return_value=mock_bot.config):
            router = Router(mock_bot)
            router.logger = Mock()
            router.metrics = Mock()
            return router
    
    @pytest.fixture
    def mock_bot(self):
        bot = Mock()
        bot.config = {
            'TWITTER_IMAGE_ONLY_ENABLE': 'true',
            'VISION_CAPTION_STYLE': 'neutral',
            'VISION_OCR_ENABLE': 'true',
        }
        bot.logger = Mock()
        return bot

    @pytest.mark.asyncio
    async def test_vision_api_failure_handling(self, router):
        """Test graceful handling of vision API failures."""
        failing_tweet = {
            "text": "",
            "photos": [{"url": "https://pbs.twimg.com/media/broken.jpg"}],
            "user": {"screen_name": "testuser"},
            "created_at": "2024-01-01T12:00:00Z"
        }
        
        # Mock vision API failure
        with patch.object(router, '_vl_describe_image_from_url', new_callable=AsyncMock) as mock_vision:
            mock_vision.side_effect = Exception("Vision API unavailable")
            
            result = await router._handle_image_only_tweet(
                "https://twitter.com/testuser/status/123",
                failing_tweet
            )
            
            # Should return user-friendly error message
            assert "⚠️ Could not process images from this tweet right now" in result
            assert "Please try again later" in result
            
            # Verify error was logged
            router.logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_malformed_syndication_data(self, router):
        """Test handling of malformed syndication data."""
        malformed_cases = [
            # Missing photos array
            {"text": "", "user": {"screen_name": "test"}},
            # Photos is None
            {"text": "", "photos": None, "user": {"screen_name": "test"}},
            # Empty photos with missing URLs
            {"text": "", "photos": [{"width": 800}], "user": {"screen_name": "test"}},
            # Corrupt user data
            {"text": "", "photos": [{"url": "test.jpg"}], "user": None},
        ]
        
        for malformed_data in malformed_cases:
            result = await router._handle_image_only_tweet(
                "https://twitter.com/test/status/123",
                malformed_data
            )
            
            # Should handle gracefully without crashing
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio 
    async def test_concurrent_upgrade_requests(self, upgrade_manager):
        """Test handling of concurrent upgrade requests on same message."""
        message_id = 123456789
        context = {
            "url": "https://twitter.com/test/status/123",
            "syndication_data": {"photos": [{"url": "test.jpg"}]},
            "source": "test"
        }
        upgrade_manager.store_upgrade_context(message_id, context)
        
        # Create multiple concurrent upgrade requests
        payloads = []
        for emoji in ["🖼️", "🔎", "🏷️"]:
            payload = Mock()
            payload.message_id = message_id
            payload.emoji = Mock()
            payload.emoji.name = emoji
            payloads.append(payload)
        
        with patch.object(upgrade_manager, '_get_detailed_vision_analysis', new_callable=AsyncMock) as mock_vision:
            mock_vision.return_value = {"alt_text": "Test response", "confidence": 0.9}
            
            # Execute concurrent requests
            tasks = [upgrade_manager.handle_upgrade_reaction(p) for p in payloads]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should complete without exceptions
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_rate_limit_simulation(self, router):
        """Test behavior under simulated rate limiting."""
        rate_limited_tweet = {
            "text": "",
            "photos": [{"url": "https://pbs.twimg.com/media/test.jpg"}],
            "user": {"screen_name": "testuser"},
            "created_at": "2024-01-01T12:00:00Z"
        }
        
        # Mock rate limit error
        with patch.object(router, '_vl_describe_image_from_url', new_callable=AsyncMock) as mock_vision:
            mock_vision.side_effect = Exception("Rate limit exceeded")
            
            result = await router._handle_image_only_tweet(
                "https://twitter.com/testuser/status/123",
                rate_limited_tweet
            )
            
            # Should handle rate limits gracefully
            assert "⚠️ Could not process images" in result
            assert "try again later" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
