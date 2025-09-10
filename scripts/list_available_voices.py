#!/usr/bin/env python3
"""
List all available voices from the Kokoro-ONNX voices.json file.
This helps users select a valid TTS_VOICE for their .env file.

Run with:
uv run python scripts/list_available_voices.py
"""

import json
import sys
from pathlib import Path


def list_voices():
    """List all available voices from the voices.json file."""
    try:
        tts_dir = Path("tts")
        voices_path = tts_dir / "voices.json"

        if not voices_path.exists():
            print(f"❌ Voices file not found at {voices_path}")
            print("Run the convert_voices_to_json_fixed.py script first")
            return False

        print(f"📚 Reading voices from {voices_path}")
        with open(voices_path, "r") as f:
            voices_data = json.load(f)

        print(f"\n✅ Found {len(voices_data)} available voices:\n")

        # List all voices in a readable format
        for i, voice_name in enumerate(voices_data.keys(), 1):
            print(f"{i:2d}. {voice_name}")

        print("\n📝 To use a specific voice, set TTS_VOICE in your .env file:")
        print("TTS_VOICE=voice_name_here  # Replace with a voice from the list above")

        return True
    except Exception as e:
        print(f"❌ Error listing voices: {str(e)}")
        return False


if __name__ == "__main__":
    sys.exit(0 if list_voices() else 1)
