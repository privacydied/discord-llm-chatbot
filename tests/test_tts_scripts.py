#!/usr/bin/env python3
"""Test the TTS functionality directly to verify our fixes."""

import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path so we can import bot modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

TTSManager = importlib.import_module("bot.tts").TTSManager


async def main() -> bool | None:
    """Test TTS functionality."""
    # Load environment variables
    load_dotenv()

    # Create config dict from env vars
    config = {
        "TTS_BACKEND": os.getenv("TTS_BACKEND", "kokoro-onnx"),
        "TTS_VOICE": os.getenv("TTS_VOICE", "am_michael"),
        "TTS_VOICE_FILE": os.getenv("TTS_VOICE_FILE", "tts/voices.json"),
        "TTS_MODEL_FILE": os.getenv("TTS_MODEL_FILE", "tts/kokoro-v1.0.onnx"),
        "TTS_CACHE_DIR": os.getenv("TTS_CACHE_DIR", "cache/tts"),
    }


    # Initialize TTS manager
    tts_manager = TTSManager(config)

    # Check if TTS is available
    if not tts_manager.available:
        return False


    # Generate TTS for a test sentence
    text = "Hello! This is a test of the Kokoro-ONNX text-to-speech system."

    try:
        # Generate TTS
        output_path = await tts_manager.generate_tts(text, tts_manager.voice)


        # Optional: Play the audio if on Linux with aplay
        try:
            import subprocess

            subprocess.run(["aplay", str(output_path)], check=True)
        except Exception as e:
            pass

        return True
    except Exception as e:
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(main())
