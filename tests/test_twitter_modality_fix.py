#!/usr/bin/env python3
"""Test script to verify Twitter/X modality detection fix.
Ensures text/image tweets are routed to screenshot flow, not video/STT flow.
"""

import asyncio
import re

# Import the modules we need to test
from bot.modality import (
    InputItem,
    InputModality,
    _map_url_to_modality,
    map_item_to_modality,
)
from bot.video_ingest import SUPPORTED_PATTERNS


async def test_twitter_url_routing() -> bool:
    """Test that Twitter URLs are correctly routed based on content type."""
    # Test URLs from the original error log
    test_urls = [
        # The problematic URL from user's error - should NOT be VIDEO_URL
        "https://x.com/avaricum777/status/1953657907964477640",
        # Other regular tweet URLs - should NOT be VIDEO_URL
        "https://twitter.com/user/status/1234567890",
        "https://x.com/user/status/9876543210",
        "https://www.twitter.com/someuser/status/1111111111",
        "https://www.x.com/someuser/status/2222222222",
        # Twitter Spaces/Live broadcasts - SHOULD be VIDEO_URL
        "https://twitter.com/i/broadcasts/1BdGYYpjoDkGX",
        "https://x.com/i/broadcasts/1BdGYYpjoDkGX",
        # YouTube URLs for comparison - SHOULD be VIDEO_URL
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        # TikTok URLs - SHOULD be VIDEO_URL
        "https://www.tiktok.com/@user/video/7123456789",
        "https://vm.tiktok.com/ZM8abc123/",
    ]

    # Test each URL
    for url in test_urls:
        # Check if URL matches video patterns
        video_match = False
        for pattern in SUPPORTED_PATTERNS:
            if re.search(pattern, url):
                video_match = True
                break

        # Test modality detection
        try:
            modality = await _map_url_to_modality(url)

            # Check expectations
            is_expected_video = (
                "youtube.com" in url or "youtu.be" in url or "tiktok.com" in url or "/broadcasts/" in url  # Twitter Spaces/Live only
            )

            is_regular_tweet = ("twitter.com" in url or "x.com" in url) and "/status/" in url and "/broadcasts/" not in url

            if is_regular_tweet:
                # Regular tweets should NOT be VIDEO_URL
                if modality == InputModality.VIDEO_URL:
                    return False

            elif is_expected_video:
                # These should be VIDEO_URL
                if modality == InputModality.VIDEO_URL:
                    pass
                else:
                    return False

        except Exception as e:
            return False

    return True


async def test_inputitem_routing() -> bool:
    """Test InputItem routing to ensure end-to-end flow works."""
    # Create InputItem for the problematic URL
    problem_url = "https://x.com/avaricum777/status/1953657907964477640"
    item = InputItem(source_type="url", payload=problem_url, order_index=0)

    # Test modality mapping
    modality = await map_item_to_modality(item)

    return modality != InputModality.VIDEO_URL


if __name__ == "__main__":

    async def main() -> None:
        success = await test_twitter_url_routing()
        if success:
            success = await test_inputitem_routing()

        if success:
            pass
        else:
            pass

    asyncio.run(main())
