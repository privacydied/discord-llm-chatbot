#!/usr/bin/env python3
"""Test script to verify Twitter/X fallback logic implementation.
Tests that video tweets are processed via yt-dlp and non-video tweets fallback to screenshot + VL.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

# Import the modules we need to test
from bot.modality import InputItem, InputModality, map_item_to_modality
from bot.router import Router
from bot.video_ingest import VideoIngestError


async def test_twitter_fallback_behavior() -> bool:
    """Test Twitter URL fallback behavior: yt-dlp first, then screenshot + VL."""
    # Test URLs
    test_cases = [
        {
            "url": "https://x.com/RpsAgainstTrump/status/1953578643042840689",  # User's video example
            "description": "Tweet with actual video content",
            "expected_first_attempt": "video_processing",
            "should_fallback": False,
        },
        {
            "url": "https://x.com/avaricum777/status/1953657907964477640",  # User's problematic example
            "description": "Tweet with text/images only (no video)",
            "expected_first_attempt": "video_processing",
            "should_fallback": True,
        },
        {
            "url": "https://twitter.com/user/status/1234567890",
            "description": "Regular text tweet",
            "expected_first_attempt": "video_processing",
            "should_fallback": True,
        },
    ]

    # Test modality detection - all should be VIDEO_URL initially
    for case in test_cases:
        url = case["url"]
        modality = await map_item_to_modality(InputItem(source_type="url", payload=url, order_index=0))

        if modality != InputModality.VIDEO_URL:
            return False

    # Test fallback logic behavior

    # Create mock bot and router
    mock_bot = Mock()
    mock_bot.config = Mock()
    mock_bot.tts_manager = Mock()
    mock_bot.loop = Mock()

    router = Router(mock_bot)

    for case in test_cases:
        url = case["url"]

        # Create InputItem
        item = InputItem(source_type="url", payload=url, order_index=0)

        # Test the video processing with mocked failures
        if case["should_fallback"]:
            # Mock hear_infer_from_url to simulate "No video found" error
            with patch("bot.router.hear_infer_from_url") as mock_hear:
                mock_hear.side_effect = VideoIngestError("yt-dlp metadata extraction failed: ERROR: [twitter] 1953657907964477640: No video could be found in this tweet")

                # Mock _handle_image to simulate successful screenshot processing
                with patch.object(router, "_handle_image", new_callable=AsyncMock) as mock_handle_image:
                    mock_handle_image.return_value = "Image analysis of tweet: This appears to be a text/image tweet with no video content."

                    try:
                        result = await router._handle_video_url(item)

                        # Verify that _handle_image was called (fallback happened)
                        mock_handle_image.assert_called_once_with(item)

                    except Exception as e:
                        return False
        else:
            # Mock successful video processing
            with patch("bot.router.hear_infer_from_url") as mock_hear:
                mock_hear.return_value = {
                    "transcription": "This is a video with actual spoken content that was transcribed.",
                    "metadata": {"title": "Example Video Tweet", "duration": 30.0},
                }

                try:
                    result = await router._handle_video_url(item)

                except Exception as e:
                    return False

    return True


async def test_error_patterns() -> bool:
    """Test specific error pattern matching for fallback trigger."""
    # Test error patterns that should trigger fallback
    fallback_triggers = [
        "yt-dlp metadata extraction failed: ERROR: [twitter] 1953657907964477640: No video could be found in this tweet",
        "VideoIngestError: Failed to download video: No video found",
        "Video extraction failed for this URL",
        "ERROR: No video could be found in this tweet",
    ]

    # Test error patterns that should NOT trigger fallback (other video errors)
    non_fallback_errors = [
        "Connection timeout",
        "Private video unavailable",
        "Video too long to process",
        "Audio processing failed",
    ]

    for error_msg in fallback_triggers:
        error_str = error_msg.lower()
        no_video_found = "no video could be found" in error_str or "no video" in error_str or "video extraction failed" in error_str

        if no_video_found:
            pass
        else:
            return False

    for error_msg in non_fallback_errors:
        error_str = error_msg.lower()
        no_video_found = "no video could be found" in error_str or "no video" in error_str or "video extraction failed" in error_str

        if not no_video_found:
            pass
        else:
            return False

    return True


if __name__ == "__main__":

    async def main() -> None:
        success = await test_twitter_fallback_behavior()
        if success:
            success = await test_error_patterns()

        if success:
            pass
        else:
            pass

    asyncio.run(main())
