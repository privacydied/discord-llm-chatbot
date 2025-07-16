#!/usr/bin/env python3
"""Test that TTS manager always returns valid Path objects."""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the bot module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.tts import TTSManager
from bot.config import load_config


async def test_tts_returns_path():
    """Test that generate_tts always returns a Path object or raises an exception."""
    print("Testing TTSManager.generate_tts return type...")
    
    # Load config from environment
    config = load_config()
    
    # Initialize TTSManager
    tts_manager = TTSManager(config)
    # TTSManager is initialized in the constructor, no need for explicit initialize() call
    
    if not tts_manager.is_available():
        print("TTS is not available, skipping test")
        return
    
    try:
        # Test with valid text
        result = await tts_manager.generate_tts("Hello world")
        print(f"Result type: {type(result)}")
        assert isinstance(result, Path), f"Expected Path, got {type(result)}"
        assert result.exists(), f"Path {result} does not exist"
        assert result.stat().st_size > 0, f"File {result} is empty"
        print(f"Success! TTS returned a valid Path: {result}")
        print(f"File size: {result.stat().st_size} bytes")
        
        # Test with empty text - should raise ValueError
        try:
            await tts_manager.generate_tts("")
            print("ERROR: Empty text did not raise an exception!")
        except ValueError as e:
            print(f"Success! Empty text raised ValueError: {e}")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        await tts_manager.close()


if __name__ == "__main__":
    asyncio.run(test_tts_returns_path())
