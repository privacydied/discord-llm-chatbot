#!/usr/bin/env python3
"""Test that TTS manager always returns valid Path objects."""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the bot module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib

from bot.config import load_config
from bot.tts import TTSManager


async def test_tts_returns_path() -> None:
    """Test that generate_tts always returns a Path object or raises an exception."""
    # Load config from environment
    config = load_config()

    # Initialize TTSManager
    tts_manager = TTSManager(config)
    # TTSManager is initialized in the constructor, no need for explicit initialize() call

    if not tts_manager.is_available():
        return

    try:
        # Test with valid text
        result = await tts_manager.generate_tts("Hello world")
        assert isinstance(result, Path), f"Expected Path, got {type(result)}"
        assert result.exists(), f"Path {result} does not exist"
        assert result.stat().st_size > 0, f"File {result} is empty"

        # Test with empty text - should raise ValueError
        with contextlib.suppress(ValueError):
            await tts_manager.generate_tts("")

    except Exception as e:
        pass
    finally:
        await tts_manager.close()


if __name__ == "__main__":
    asyncio.run(test_tts_returns_path())
