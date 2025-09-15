"""
Unit tests for syndication content extraction.
Tests for extract_text_and_images_from_syndication function.
"""

from bot.syndication.extract import extract_text_and_images_from_syndication


class TestSyndicationExtract:
    """Test cases for syndication content extraction."""

    def test_extract_photos_multiple(self):
        """extract_photos_multiple: Extract multiple photos with name=orig upgrade."""
        syndication_json = {
            "text": "Check out these photos!",
            "photos": [
                {"url": "https://pbs.twimg.com/media/ABC123?format=jpg&name=small"},
                {
                    "media_url_https": "https://pbs.twimg.com/media/DEF456?format=png&name=240x240"
                },
                {"url": "https://pbs.twimg.com/media/GHI789?name=large"},
            ],
        }

        result = extract_text_and_images_from_syndication(syndication_json)

        assert result["text"] == "Check out these photos!"
        assert len(result["image_urls"]) == 3
        assert (
            "https://pbs.twimg.com/media/ABC123?format=jpg&name=orig"
            in result["image_urls"]
        )
        assert (
            "https://pbs.twimg.com/media/DEF456?format=png&name=orig"
            in result["image_urls"]
        )
        assert "https://pbs.twimg.com/media/GHI789?name=orig" in result["image_urls"]

    def test_extract_fallback_card(self):
        """extract_fallback_card: Use card image when no photos present."""
        syndication_json = {
            "text": "No photos, just a card",
            "photos": [],
            "image": {"url": "https://pbs.twimg.com/card_img/ABC123?name=small"},
        }

        result = extract_text_and_images_from_syndication(syndication_json)

        assert result["text"] == "No photos, just a card"
        assert len(result["image_urls"]) == 1
        assert (
            result["image_urls"][0] == "https://pbs.twimg.com/card_img/ABC123?name=orig"
        )

    def test_extract_fallback_card_string_format(self):
        """Handle card image as direct string URL."""
        syndication_json = {
            "text": "Card as string",
            "photos": [],
            "image": "https://pbs.twimg.com/card_img/ABC123?name=small",
        }

        result = extract_text_and_images_from_syndication(syndication_json)

        assert result["text"] == "Card as string"
        assert len(result["image_urls"]) == 1
        assert (
            result["image_urls"][0] == "https://pbs.twimg.com/card_img/ABC123?name=orig"
        )

    def test_extract_quoted_only_when_none(self):
        """extract_quoted_only_when_none: Use quoted tweet photos only when main has none."""
        # Case 1: Main has photos, ignore quoted
        syndication_json_main_photos = {
            "text": "Main tweet with photos",
            "photos": [{"url": "https://pbs.twimg.com/media/MAIN123?name=small"}],
            "quoted_tweet": {
                "photos": [{"url": "https://pbs.twimg.com/media/QUOTED456?name=small"}]
            },
        }

        result = extract_text_and_images_from_syndication(syndication_json_main_photos)

        assert len(result["image_urls"]) == 1
        assert "MAIN123" in result["image_urls"][0]
        assert "QUOTED456" not in str(result["image_urls"])

        # Case 2: No main photos, use quoted
        syndication_json_quoted_only = {
            "text": "Quoted tweet content",
            "photos": [],
            "quoted_tweet": {
                "photos": [
                    {"url": "https://pbs.twimg.com/media/QUOTED456?name=small"},
                    {
                        "media_url_https": "https://pbs.twimg.com/media/QUOTED789?name=large"
                    },
                ]
            },
        }

        result = extract_text_and_images_from_syndication(syndication_json_quoted_only)

        assert len(result["image_urls"]) == 2
        assert "https://pbs.twimg.com/media/QUOTED456?name=orig" in result["image_urls"]
        assert "https://pbs.twimg.com/media/QUOTED789?name=orig" in result["image_urls"]

    def test_extract_deduplication(self):
        """Ensure duplicate URLs are removed while preserving order."""
        syndication_json = {
            "text": "Duplicate images",
            "photos": [
                {"url": "https://pbs.twimg.com/media/ABC123?name=small"},
                {
                    "url": "https://pbs.twimg.com/media/ABC123?name=large"
                },  # Same base URL
                {"url": "https://pbs.twimg.com/media/DEF456?name=small"},
            ],
        }

        result = extract_text_and_images_from_syndication(syndication_json)

        # Should have 2 unique URLs (ABC123 and DEF456), both upgraded to name=orig
        assert len(result["image_urls"]) == 2
        assert "https://pbs.twimg.com/media/ABC123?name=orig" in result["image_urls"]
        assert "https://pbs.twimg.com/media/DEF456?name=orig" in result["image_urls"]

        # Verify order preservation (ABC123 should come first)
        assert result["image_urls"][0] == "https://pbs.twimg.com/media/ABC123?name=orig"
        assert result["image_urls"][1] == "https://pbs.twimg.com/media/DEF456?name=orig"

    def test_extract_text_fallbacks(self):
        """Test text extraction with fallbacks (full_text vs text)."""
        # Test full_text priority
        syndication_json_full = {
            "full_text": "Full text content",
            "text": "Truncated text...",
            "photos": [],
        }

        result = extract_text_and_images_from_syndication(syndication_json_full)
        assert result["text"] == "Full text content"

        # Test text fallback when no full_text
        syndication_json_text = {"text": "Regular text content", "photos": []}

        result = extract_text_and_images_from_syndication(syndication_json_text)
        assert result["text"] == "Regular text content"

        # Test empty when both missing
        syndication_json_empty = {"photos": []}

        result = extract_text_and_images_from_syndication(syndication_json_empty)
        assert result["text"] == ""

    def test_extract_missing_photo_urls(self):
        """Handle photos with missing or empty URLs gracefully."""
        syndication_json = {
            "text": "Some photos missing URLs",
            "photos": [
                {"url": "https://pbs.twimg.com/media/GOOD123?name=small"},
                {"url": None},  # Missing URL
                {"media_url_https": ""},  # Empty URL
                {},  # No URL fields
                {"url": "https://pbs.twimg.com/media/GOOD456?name=small"},
            ],
        }

        result = extract_text_and_images_from_syndication(syndication_json)

        # Should only extract the 2 valid URLs
        assert len(result["image_urls"]) == 2
        assert "https://pbs.twimg.com/media/GOOD123?name=orig" in result["image_urls"]
        assert "https://pbs.twimg.com/media/GOOD456?name=orig" in result["image_urls"]

    def test_extract_non_pbs_urls_passthrough(self):
        """Non-pbs URLs should pass through unchanged."""
        syndication_json = {
            "text": "Mixed URLs",
            "photos": [
                {"url": "https://pbs.twimg.com/media/PBS123?name=small"},
                {"url": "https://example.com/image.jpg"},
                {"url": "https://cdn.twitter.com/media/other.png"},
            ],
        }

        result = extract_text_and_images_from_syndication(syndication_json)

        assert len(result["image_urls"]) == 3
        assert "https://pbs.twimg.com/media/PBS123?name=orig" in result["image_urls"]
        assert "https://example.com/image.jpg" in result["image_urls"]  # Unchanged
        assert (
            "https://cdn.twitter.com/media/other.png" in result["image_urls"]
        )  # Unchanged

    def test_extract_metrics_integration(self):
        """Test that metrics integration doesn't break extraction."""
        syndication_json = {
            "text": "Test metrics",
            "photos": [
                {"url": "https://pbs.twimg.com/media/ABC123?name=small"},
                {"url": "https://pbs.twimg.com/media/DEF456?name=small"},
            ],
        }

        result = extract_text_and_images_from_syndication(syndication_json)

        # Function should work regardless of metrics availability
        assert len(result["image_urls"]) == 2
        assert result["text"] == "Test metrics"

    def test_extract_metrics_failure_resilience(self):
        """Ensure extraction continues even if metrics fail."""
        syndication_json = {
            "text": "Test resilience",
            "photos": [{"url": "https://pbs.twimg.com/media/ABC123?name=small"}],
        }

        result = extract_text_and_images_from_syndication(syndication_json)

        # Should work despite any metrics issues
        assert result["text"] == "Test resilience"
        assert len(result["image_urls"]) == 1
        assert result["image_urls"][0] == "https://pbs.twimg.com/media/ABC123?name=orig"
