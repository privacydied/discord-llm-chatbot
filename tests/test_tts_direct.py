#!/usr/bin/env python
"""Simple test script for KokoroDirect TTS functionality."""

import os
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv

# Import the fixed KokoroDirect implementation
from bot.tts.kokoro_direct import KokoroDirect


def main() -> None:
    # Load environment variables
    load_dotenv()

    # Get model and voice paths from environment or use defaults
    model_path = os.environ.get("TTS_MODEL_FILE", "tts/onnx/kokoro-v1.0.onnx")
    voice_path = os.environ.get("TTS_VOICE_FILE", "tts/voices/voices-v1.0.bin")

    # Initialize KokoroDirect
    kokoro = KokoroDirect(model_path=model_path, voices_path=voice_path)

    # Get available voices
    voices = kokoro.get_voice_names()
    if not voices:
        return

    # Use the first available voice
    test_voice = voices[0]

    # Create a temporary output path
    output_path = Path("test_output.wav")

    # Generate audio for a test phrase
    text = "Hello world, this is a test of the TTS system."

    result = kokoro.create(text, test_voice, out_path=output_path)

    # Verify the result

    if isinstance(result, Path) and result.exists():
        file_size = result.stat().st_size

        # Get audio duration
        audio_info = sf.info(result)
        duration = audio_info.duration

    else:
        pass


if __name__ == "__main__":
    main()
