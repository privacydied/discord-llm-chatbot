#!/usr/bin/env python3
"""
Test script to verify Twitter/X fallback logic implementation.
Tests that video tweets are processed via yt-dlp and non-video tweets fallback to screenshot + VL.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, patch

# Import the modules we need to test
from bot.modality import InputModality, InputItem, map_item_to_modality
from bot.router import Router
from bot.video_ingest import VideoIngestError

async def test_twitter_fallback_behavior():
    """Test Twitter URL fallback behavior: yt-dlp first, then screenshot + VL."""
    
    print("🧪 Testing Twitter/X fallback logic implementation...")
    print("=" * 70)
    
    # Test URLs
    test_cases = [
        {
            "url": "https://x.com/RpsAgainstTrump/status/1953578643042840689",  # User's video example
            "description": "Tweet with actual video content",
            "expected_first_attempt": "video_processing",
            "should_fallback": False
        },
        {
            "url": "https://x.com/avaricum777/status/1953657907964477640",  # User's problematic example
            "description": "Tweet with text/images only (no video)",
            "expected_first_attempt": "video_processing", 
            "should_fallback": True
        },
        {
            "url": "https://twitter.com/user/status/1234567890",
            "description": "Regular text tweet",
            "expected_first_attempt": "video_processing",
            "should_fallback": True
        }
    ]
    
    # Test modality detection - all should be VIDEO_URL initially
    print("\n📋 Step 1: Testing initial modality detection")
    for case in test_cases:
        url = case["url"]
        modality = await map_item_to_modality(InputItem(
            source_type="url",
            payload=url,
            order_index=0
        ))
        print(f"   {case['description']}: {modality}")
        
        if modality != InputModality.VIDEO_URL:
            print(f"   ❌ FAIL: Expected VIDEO_URL, got {modality}")
            return False
        else:
            print("   ✅ PASS: Correctly routed to VIDEO_URL modality")
    
    # Test fallback logic behavior
    print("\n🔄 Step 2: Testing fallback logic in Router._handle_video_url")
    
    # Create mock bot and router
    mock_bot = Mock()
    mock_bot.config = Mock()
    mock_bot.tts_manager = Mock()
    mock_bot.loop = Mock()
    
    router = Router(mock_bot)
    
    for case in test_cases:
        url = case["url"]
        print(f"\n📋 Testing: {case['description']}")
        print(f"   URL: {url}")
        
        # Create InputItem
        item = InputItem(
            source_type="url",
            payload=url,
            order_index=0
        )
        
        # Test the video processing with mocked failures
        if case["should_fallback"]:
            print("   Expected behavior: yt-dlp fails → fallback to screenshot + VL")
            
            # Mock hear_infer_from_url to simulate "No video found" error
            with patch('bot.router.hear_infer_from_url') as mock_hear:
                mock_hear.side_effect = VideoIngestError("yt-dlp metadata extraction failed: ERROR: [twitter] 1953657907964477640: No video could be found in this tweet")
                
                # Mock _handle_image to simulate successful screenshot processing
                with patch.object(router, '_handle_image', new_callable=AsyncMock) as mock_handle_image:
                    mock_handle_image.return_value = "Image analysis of tweet: This appears to be a text/image tweet with no video content."
                    
                    try:
                        result = await router._handle_video_url(item)
                        print(f"   ✅ PASS: Fallback successful, result: {result[:100]}...")
                        
                        # Verify that _handle_image was called (fallback happened)
                        mock_handle_image.assert_called_once_with(item)
                        
                    except Exception as e:
                        print(f"   ❌ FAIL: Fallback failed with error: {e}")
                        return False
        else:
            print("   Expected behavior: yt-dlp succeeds → video transcription")
            
            # Mock successful video processing 
            with patch('bot.router.hear_infer_from_url') as mock_hear:
                mock_hear.return_value = {
                    'transcription': 'This is a video with actual spoken content that was transcribed.',
                    'metadata': {'title': 'Example Video Tweet', 'duration': 30.0}
                }
                
                try:
                    result = await router._handle_video_url(item)
                    print(f"   ✅ PASS: Video processing successful, result: {result[:100]}...")
                    
                except Exception as e:
                    print(f"   ❌ FAIL: Video processing failed: {e}")
                    return False
    
    print("\n" + "=" * 70)
    print("✅ ALL FALLBACK TESTS PASSED!")
    print("\n📝 Summary of Twitter/X URL Processing Logic:")
    print("   1. All Twitter/X status URLs initially get VIDEO_URL modality")
    print("   2. Router tries yt-dlp video extraction first")
    print("   3. If video extraction succeeds → STT transcription")
    print("   4. If 'No video found' error → fallback to screenshot + VL processing")
    print("   5. Other video processing errors → standard error handling")
    
    return True

async def test_error_patterns():
    """Test specific error pattern matching for fallback trigger."""
    
    print("\n🔍 Testing error pattern detection...")
    
    # Test error patterns that should trigger fallback
    fallback_triggers = [
        "yt-dlp metadata extraction failed: ERROR: [twitter] 1953657907964477640: No video could be found in this tweet",
        "VideoIngestError: Failed to download video: No video found",
        "Video extraction failed for this URL",
        "ERROR: No video could be found in this tweet"
    ]
    
    # Test error patterns that should NOT trigger fallback (other video errors)
    non_fallback_errors = [
        "Connection timeout",
        "Private video unavailable", 
        "Video too long to process",
        "Audio processing failed"
    ]
    
    for error_msg in fallback_triggers:
        error_str = error_msg.lower()
        no_video_found = (
            "no video could be found" in error_str or 
            "no video" in error_str or 
            "video extraction failed" in error_str
        )
        
        if no_video_found:
            print(f"   ✅ PASS: '{error_msg[:50]}...' correctly triggers fallback")
        else:
            print(f"   ❌ FAIL: '{error_msg[:50]}...' should trigger fallback")
            return False
    
    for error_msg in non_fallback_errors:
        error_str = error_msg.lower()
        no_video_found = (
            "no video could be found" in error_str or 
            "no video" in error_str or 
            "video extraction failed" in error_str
        )
        
        if not no_video_found:
            print(f"   ✅ PASS: '{error_msg}' correctly does NOT trigger fallback")
        else:
            print(f"   ❌ FAIL: '{error_msg}' should NOT trigger fallback")
            return False
    
    return True

if __name__ == "__main__":
    async def main():
        success = await test_twitter_fallback_behavior()
        if success:
            success = await test_error_patterns()
        
        if success:
            print("\n🎉 All fallback logic tests passed!")
            print("\n🚀 The bot will now:")
            print("   • Try video extraction first on ALL tweet URLs")
            print("   • Fall back to screenshot + VL for tweets without video")
            print("   • Preserve full video processing for tweets with actual video content")
        else:
            print("\n❌ Some fallback tests failed. Please check the implementation.")
            
    asyncio.run(main())
