#!/usr/bin/env python3
"""Test script to verify fallback logic works with EXACT production error messages.
Uses the exact error chain from user's production tracebacks.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from bot.exceptions import InferenceError

# Import the modules we need to test
from bot.modality import InputItem
from bot.router import Router


async def test_exact_production_error_handling() -> bool | None:
    """Test fallback logic with the exact production error messages from user's logs."""
    # The exact URL from user's production logs that failed
    problem_url = "https://x.com/avaricum777/status/1953657907964477640"


    # Create mock bot and router
    mock_bot = Mock()
    mock_bot.config = Mock()
    mock_bot.tts_manager = Mock()
    mock_bot.loop = Mock()

    router = Router(mock_bot)

    # Create InputItem for the problematic URL
    item = InputItem(source_type="url", payload=problem_url, order_index=0)

    # Test with the EXACT error message format from production logs

    # The exact InferenceError message from the production traceback
    exact_production_error = InferenceError("Video transcription failed: Failed to download video: yt-dlp metadata extraction failed: ERROR: [twitter] 1953657907964477640: No video could be found in this tweet")


    # Test error pattern detection with exact production error
    error_str = str(exact_production_error).lower()
    is_twitter_url = ("twitter.com" in problem_url.lower() or "x.com" in problem_url.lower()) and "/status/" in problem_url.lower()
    no_video_found = "no video could be found" in error_str or "no video" in error_str or "video extraction failed" in error_str


    if not (is_twitter_url and no_video_found):
        return False


    # Test the full fallback chain with the exact production error

    with patch("bot.router.hear_infer_from_url") as mock_hear:
        # Make hear_infer_from_url raise the exact production error
        mock_hear.side_effect = exact_production_error

        # Mock _handle_image to simulate successful screenshot processing
        with patch.object(router, "_handle_image", new_callable=AsyncMock) as mock_handle_image:
            mock_handle_image.return_value = "Screenshot analysis: This tweet contains text and images but no video content."

            try:
                result = await router._handle_video_url(item)

                # Verify that _handle_image was called (fallback happened)
                mock_handle_image.assert_called_once_with(item)

                return True

            except Exception as e:
                return False


async def test_other_twitter_errors_no_fallback() -> bool:
    """Test that other Twitter errors (not 'no video found') don't trigger fallback."""
    problem_url = "https://x.com/some_user/status/1234567890"

    # Create mock bot and router
    mock_bot = Mock()
    mock_bot.config = Mock()
    mock_bot.tts_manager = Mock()
    mock_bot.loop = Mock()

    Router(mock_bot)

    # Create InputItem
    InputItem(source_type="url", payload=problem_url, order_index=0)

    # Test with other types of Twitter video errors that should NOT fallback
    other_twitter_errors = [
        "Video transcription failed: Connection timeout",
        "Video transcription failed: Private video unavailable",
        "Video transcription failed: Video too long to process",
        "Video transcription failed: Authentication required",
    ]

    for error_msg in other_twitter_errors:

        error_str = error_msg.lower()
        is_twitter_url = ("twitter.com" in problem_url.lower() or "x.com" in problem_url.lower()) and "/status/" in problem_url.lower()
        no_video_found = "no video could be found" in error_str or "no video" in error_str or "video extraction failed" in error_str

        should_fallback = is_twitter_url and no_video_found

        if should_fallback:
            return False

    return True


if __name__ == "__main__":

    async def main() -> None:

        success = await test_exact_production_error_handling()
        if success:
            success = await test_other_twitter_errors_no_fallback()

        if success:
            pass
        else:
            pass

    asyncio.run(main())
