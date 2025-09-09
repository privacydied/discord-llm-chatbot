"""
Integration tests for Twitter/X syndication to VL flow.
Tests the complete pipeline from syndication data to VL processing with full-res images.
"""
import pytest
import asyncio
from bot.syndication.handler import handle_twitter_syndication_to_vl


class TestSyndicationIntegration:
    """Integration tests for complete syndication to VL flow."""

    @pytest.fixture
    def mock_vl_handler(self):
        """Mock VL handler function that returns predictable results."""
        async def mock_handler(image_url, prompt=None):
            # Simulate different responses based on image URL
            if "SUCCESS" in image_url:
                return f"Analysis of {image_url}: This is a test image."
            elif "FAIL" in image_url:
                return None  # Simulate VL failure
            else:
                return f"Analysis of {image_url}: Standard result."
        
        return mock_handler

    @pytest.fixture
    def sample_syndication_data(self):
        """Sample syndication JSON data for testing."""
        return {
            "text": "Check out these amazing photos from my trip!",
            "photos": [
                {"url": "https://pbs.twimg.com/media/SUCCESS1?format=jpg&name=small"},
                {"url": "https://pbs.twimg.com/media/SUCCESS2?format=png&name=240x240"},
                {"url": "https://pbs.twimg.com/media/FAIL3?format=jpg&name=large"}
            ],
            "user": {"screen_name": "testuser"},
            "created_at": "2024-01-01T00:00:00Z"
        }

    @pytest.mark.asyncio
    async def test_multiple_images_vl_flow(self, mock_vl_handler, sample_syndication_data):
        """Test processing multiple images through VL with full-res URLs."""
        url = "https://twitter.com/testuser/status/123456789"
        
        result = await handle_twitter_syndication_to_vl(
            sample_syndication_data, url, mock_vl_handler
        )
        
        # Verify result contains tweet text
        assert "Check out these amazing photos from my trip!" in result
        
        # Verify all 3 photos were processed (2 success, 1 fail based on mock handler)
        assert "Photos analyzed: 2/3" in result  # 2 success, 1 fail
        
        # Verify individual photo results
        assert "📷 Photo 1/3" in result
        assert "📷 Photo 2/3" in result
        assert "📷 Photo 3/3" in result
        
        # Verify successful analyses contain content (with upgraded URLs)
        assert "Analysis of https://pbs.twimg.com/media/SUCCESS1?format=jpg&name=orig" in result
        assert "Analysis of https://pbs.twimg.com/media/SUCCESS2?format=png&name=orig" in result
        
        # Verify failed analysis shows unavailable message
        assert "analysis unavailable" in result

    @pytest.mark.asyncio
    async def test_single_image_tweet(self, mock_vl_handler):
        """Test processing single image tweet."""
        syndication_data = {
            "text": "Single photo tweet",
            "photos": [
                {"url": "https://pbs.twimg.com/media/SUCCESS1?format=jpg&name=small"}
            ]
        }
        
        url = "https://twitter.com/user/status/123"
        
        result = await handle_twitter_syndication_to_vl(
            syndication_data, url, mock_vl_handler
        )
        
        assert "Single photo tweet" in result
        assert "Photos analyzed: 1/1" in result
        assert "📷 Photo 1/1" in result
        assert "Analysis of https://pbs.twimg.com/media/SUCCESS1?format=jpg&name=orig" in result

    @pytest.mark.asyncio
    async def test_no_images_fallback(self, mock_vl_handler):
        """Test handling of tweets with no images."""
        syndication_data = {
            "text": "Text-only tweet with no photos",
            "photos": []
        }
        
        url = "https://twitter.com/user/status/123"
        
        result = await handle_twitter_syndication_to_vl(
            syndication_data, url, mock_vl_handler
        )
        
        # Should return text-only content
        assert result == "Text-only tweet with no photos"

    @pytest.mark.asyncio
    async def test_empty_text_with_images(self, mock_vl_handler):
        """Test tweets with images but no text content."""
        syndication_data = {
            "text": "",  # Empty text
            "photos": [
                {"url": "https://pbs.twimg.com/media/SUCCESS1?format=jpg&name=small"}
            ]
        }
        
        url = "https://twitter.com/user/status/123"
        
        result = await handle_twitter_syndication_to_vl(
            syndication_data, url, mock_vl_handler
        )
        
        # Should use URL as fallback text
        assert "Tweet from https://twitter.com/user/status/123" in result
        assert "Photos analyzed: 1/1" in result
        assert "Analysis of https://pbs.twimg.com/media/SUCCESS1?format=jpg&name=orig" in result

    @pytest.mark.asyncio
    async def test_vl_handler_exceptions(self, sample_syndication_data):
        """Test resilience when VL handler raises exceptions."""
        
        async def failing_vl_handler(image_url, prompt=None):
            if "SUCCESS1" in image_url:
                return "Successful analysis"
            elif "SUCCESS2" in image_url:
                raise Exception("VL processing error")
            else:
                return None
        
        url = "https://twitter.com/testuser/status/123456789"
        
        result = await handle_twitter_syndication_to_vl(
            sample_syndication_data, url, failing_vl_handler
        )
        
        # Should handle exception gracefully
        assert "Check out these amazing photos from my trip!" in result
        assert "Photos analyzed: 1/3" in result  # Only 1 success, 2 failures
        assert "analysis failed" in result  # Exception should be logged as failure

    @pytest.mark.asyncio
    async def test_all_vl_failures(self, sample_syndication_data):
        """Test handling when all VL analyses fail."""
        
        async def always_failing_vl_handler(image_url, prompt=None):
            return None  # Always fail
        
        url = "https://twitter.com/testuser/status/123456789"
        
        result = await handle_twitter_syndication_to_vl(
            sample_syndication_data, url, always_failing_vl_handler
        )
        
        assert "Check out these amazing photos from my trip!" in result
        assert "Photos analyzed: 0/3" in result
        assert "analysis unavailable" in result

    @pytest.mark.asyncio
    async def test_high_res_url_upgrade_integration(self, mock_vl_handler):
        """Test that URLs are properly upgraded to name=orig in the full flow."""
        syndication_data = {
            "text": "Testing URL upgrades",
            "photos": [
                {"url": "https://pbs.twimg.com/media/TEST1?format=jpg&name=small"},
                {"url": "https://pbs.twimg.com/media/TEST2?format=png&name=240x240"},
                {"url": "https://example.com/external.jpg"}  # Non-pbs URL
            ]
        }
        
        url = "https://twitter.com/user/status/123"
        
        # Track which URLs were passed to VL handler
        passed_urls = []
        
        async def tracking_vl_handler(image_url, prompt=None):
            passed_urls.append(image_url)
            return f"Analysis of {image_url}"
        
        result = await handle_twitter_syndication_to_vl(
            syndication_data, url, tracking_vl_handler
        )
        
        # Verify all URLs were processed
        assert len(passed_urls) == 3
        
        # Verify pbs URLs were upgraded to name=orig
        assert "https://pbs.twimg.com/media/TEST1?format=jpg&name=orig" in passed_urls
        assert "https://pbs.twimg.com/media/TEST2?format=png&name=orig" in passed_urls
        
        # Verify non-pbs URL was unchanged
        assert "https://example.com/external.jpg" in passed_urls
        
        # Verify result contains all analyses
        assert "Photos analyzed: 3/3" in result

    @pytest.mark.asyncio
    async def test_fallback_image_processing(self, mock_vl_handler):
        """Test processing of fallback card images when no photos present."""
        syndication_data = {
            "text": "Card image only",
            "photos": [],
            "image": {"url": "https://pbs.twimg.com/card_img/CARD123?name=small"}
        }
        
        url = "https://twitter.com/user/status/123"
        
        result = await handle_twitter_syndication_to_vl(
            syndication_data, url, mock_vl_handler
        )
        
        assert "Card image only" in result
        assert "Photos analyzed: 1/1" in result
        assert "Analysis of https://pbs.twimg.com/card_img/CARD123?name=orig: Standard result." in result

    @pytest.mark.asyncio
    async def test_concurrent_processing_simulation(self, sample_syndication_data):
        """Test that processing works correctly under concurrent conditions."""
        
        async def slow_vl_handler(image_url, prompt=None):
            # Simulate variable processing times
            if "SUCCESS1" in image_url:
                await asyncio.sleep(0.1)
                return "Slow analysis 1"
            elif "SUCCESS2" in image_url:
                await asyncio.sleep(0.05)
                return "Fast analysis 2"
            else:
                await asyncio.sleep(0.15)
                return "Slowest analysis 3"
        
        url = "https://twitter.com/testuser/status/123456789"
        
        # Process multiple requests concurrently
        tasks = [
            handle_twitter_syndication_to_vl(sample_syndication_data, url, slow_vl_handler)
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All results should be successful and consistent
        for result in results:
            assert "Check out these amazing photos from my trip!" in result
            assert "Photos analyzed: 3/3" in result
            assert "Slow analysis 1" in result
            assert "Fast analysis 2" in result
            assert "Slowest analysis 3" in result
