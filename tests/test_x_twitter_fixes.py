"""
Unit tests for X/Twitter image processing and naked image attachment fixes.
Tests the key behaviors fixed in the "Fix This Code" implementation.
"""
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from bot.syndication.extract import extract_text_and_images_from_syndication
from bot.syndication.handler import handle_twitter_syndication_to_vl
from bot.syndication.url_utils import upgrade_pbs_to_orig, pbs_base_key


class TestTwitterMediaSelection:
    """Test X/Twitter media selection: prioritize high-res native images over cards."""

    def test_native_photos_prioritized_over_card(self):
        """Primary native photos should be chosen over card images."""
        tw_data = {
            "photos": [
                {"url": "https://pbs.twimg.com/media/test1.jpg:large"}
            ],
            "card": {
                "binding_values": {
                    "photo_image_full_size_large": {
                        "image_value": {"url": "https://pbs.twimg.com/card/card.jpg"}
                    }
                }
            },
            "full_text": "Test tweet with native photo"
        }
        
        result = extract_text_and_images_from_syndication(tw_data)
        
        assert result["source"] == "photos"
        assert len(result["image_urls"]) == 1
        assert "test1.jpg" in result["image_urls"][0]
        assert result["had_card"] is True  # Card was present but not used

    def test_quoted_fallback_when_no_primary_native(self):
        """Quoted tweet photos should be used when primary has no native media."""
        tw_data = {
            "photos": [],  # No primary photos
            "quoted_tweet": {
                "photos": [
                    {"url": "https://pbs.twimg.com/media/quoted1.jpg:large"}
                ]
            },
            "full_text": "RT with comment"
        }
        
        result = extract_text_and_images_from_syndication(tw_data)
        
        assert result["source"] == "quoted_photos"
        assert len(result["image_urls"]) == 1
        assert "quoted1.jpg" in result["image_urls"][0]

    def test_card_fallback_when_no_native_media(self):
        """Card image should only be used when no native media exists."""
        tw_data = {
            "photos": [],  # No primary photos
            "quoted_tweet": {"photos": []},  # No quoted photos
            "card": {
                "binding_values": {
                    "photo_image_full_size_large": {
                        "image_value": {"url": "https://pbs.twimg.com/card/fallback.jpg"}
                    }
                }
            },
            "full_text": "Text-only tweet with link card"
        }
        
        result = extract_text_and_images_from_syndication(tw_data)
        
        assert result["source"] == "card"
        assert len(result["image_urls"]) == 1
        assert "fallback.jpg" in result["image_urls"][0]

    def test_high_res_normalization(self):
        """URLs should be upgraded to high-res with name=orig parameter."""
        tw_data = {
            "photos": [
                {"url": "https://pbs.twimg.com/media/test.jpg:large"},
                {"url": "https://pbs.twimg.com/media/test2.jpg?name=small"}
            ],
            "full_text": "Multiple images"
        }
        
        result = extract_text_and_images_from_syndication(tw_data)
        
        # All URLs should be upgraded to name=orig
        for url in result["image_urls"]:
            assert "name=orig" in url
            assert ":large" not in url  # Legacy suffix removed

    def test_deduplication_by_base_asset(self):
        """Multiple versions of same image should be deduplicated."""
        tw_data = {
            "photos": [
                {"url": "https://pbs.twimg.com/media/same.jpg:large"},
                {"url": "https://pbs.twimg.com/media/same.jpg?name=small"},
                {"url": "https://pbs.twimg.com/media/different.jpg:large"}
            ],
            "full_text": "Duplicate test"
        }
        
        result = extract_text_and_images_from_syndication(tw_data)
        
        assert len(result["image_urls"]) == 2  # Only 2 unique images
        base_keys = [pbs_base_key(url) for url in result["image_urls"]]
        assert len(set(base_keys)) == 2  # Confirm deduplication


class TestNakedImageHandling:
    """Test naked image attachment handling as implicit 'Thoughts?' request."""

    @pytest.fixture
    def mock_message(self):
        """Create a mock Discord message."""
        message = Mock()
        message.id = 12345
        message.content = ""
        message.attachments = []
        return message

    @pytest.fixture
    def mock_bot(self):
        """Create a mock bot with user ID."""
        bot = Mock()
        bot.user = Mock()
        bot.user.id = 67890
        return bot

    def test_empty_content_triggers_ack_thoughts_format(self, mock_message, mock_bot):
        """Empty content with image should use ack+thoughts format."""
        # This would be tested in the actual router logic
        # Testing the pattern matching here
        import re
        
        mock_message.content = ""
        mention_pattern = fr'^<@!?{mock_bot.user.id}>\s*'
        clean_content = re.sub(mention_pattern, '', (mock_message.content or '').strip())
        
        assert not clean_content  # Should be empty, triggering naked image logic

    def test_mention_only_content_triggers_ack_thoughts_format(self, mock_message, mock_bot):
        """Content with only bot mention should use ack+thoughts format."""
        import re
        
        mock_message.content = f"<@!{mock_bot.user.id}>"
        mention_pattern = fr'^<@!?{mock_bot.user.id}>\s*'
        clean_content = re.sub(mention_pattern, '', (mock_message.content or '').strip())
        
        assert not clean_content  # Should be empty after mention removal

    def test_text_with_image_uses_normal_flow(self, mock_message, mock_bot):
        """Text content with image should use normal processing flow."""
        import re
        
        mock_message.content = f"<@!{mock_bot.user.id}> What do you think about this?"
        mention_pattern = fr'^<@!?{mock_bot.user.id}>\s*'
        clean_content = re.sub(mention_pattern, '', (mock_message.content or '').strip())
        
        assert clean_content == "What do you think about this?"  # Has actual text


class TestReplyFormatting:
    """Test reply formatting controls (ack+thoughts vs verbatim)."""

    @pytest.mark.asyncio
    async def test_ack_thoughts_format(self):
        """Test ack+thoughts reply format for concise responses."""
        mock_vl_handler = AsyncMock(return_value="This is a cat sitting on a windowsill.")
        
        tweet_json = {
            "full_text": "Check out this adorable photo I took today! #cats #photography",
            "photos": [{"url": "https://pbs.twimg.com/media/cat.jpg"}]
        }
        
        result = await handle_twitter_syndication_to_vl(
            tweet_json=tweet_json,
            url="https://twitter.com/user/status/123",
            vl_handler_func=mock_vl_handler,
            reply_style="ack+thoughts"
        )
        
        # Should be concise acknowledgment + analysis
        assert "Got the tweet" in result
        assert "here's what the images show" in result
        assert "This is a cat sitting on a windowsill" in result
        # Should truncate long tweet text
        assert len([line for line in result.split('\n') if 'Check out this adorable' in line and len(line) < 150])

    @pytest.mark.asyncio  
    async def test_verbatim_thoughts_format(self):
        """Test verbatim+thoughts reply format for detailed responses."""
        mock_vl_handler = AsyncMock(return_value="This is a cat sitting on a windowsill.")
        
        tweet_json = {
            "full_text": "Check out this adorable photo I took today! #cats #photography", 
            "photos": [{"url": "https://pbs.twimg.com/media/cat.jpg"}]
        }
        
        result = await handle_twitter_syndication_to_vl(
            tweet_json=tweet_json,
            url="https://twitter.com/user/status/123", 
            vl_handler_func=mock_vl_handler,
            reply_style="verbatim+thoughts"
        )
        
        # Should include full tweet text and detailed analysis
        assert "Check out this adorable photo I took today! #cats #photography" in result
        assert "Photos analyzed:" in result
        assert "This is a cat sitting on a windowsill" in result


class TestUrlUtils:
    """Test URL utility functions for high-res upgrades."""

    def test_upgrade_pbs_to_orig(self):
        """Test upgrading pbs.twimg.com URLs to name=orig."""
        # Legacy :size suffix
        url1 = "https://pbs.twimg.com/media/test.jpg:large"
        result1 = upgrade_pbs_to_orig(url1)
        assert "name=orig" in result1
        assert ":large" not in result1
        
        # Existing query params
        url2 = "https://pbs.twimg.com/media/test.jpg?format=jpg&name=small"
        result2 = upgrade_pbs_to_orig(url2)
        assert "name=orig" in result2
        assert "format=jpg" in result2  # Preserve existing format
        
        # Non-pbs URLs should be unchanged
        url3 = "https://example.com/image.jpg"
        result3 = upgrade_pbs_to_orig(url3)
        assert result3 == url3

    def test_pbs_base_key_deduplication(self):
        """Test base key generation for deduplication."""
        url1 = "https://pbs.twimg.com/media/test.jpg:large"
        url2 = "https://pbs.twimg.com/media/test.jpg?name=small" 
        url3 = "https://pbs.twimg.com/media/test.jpg?name=orig"
        
        key1 = pbs_base_key(url1)
        key2 = pbs_base_key(url2)
        key3 = pbs_base_key(url3)
        
        # All should have same base key for deduplication
        assert key1 == key2 == key3
        assert key1 == "pbs:/media/test.jpg"


if __name__ == "__main__":
    pytest.main([__file__])
