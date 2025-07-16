#!/usr/bin/env python
"""
Simple test script for KokoroDirect TTS functionality.
"""
import os
from pathlib import Path
import soundfile as sf
from dotenv import load_dotenv

# Import the fixed KokoroDirect implementation
from bot.kokoro_direct_fixed import KokoroDirect

def main():
    # Load environment variables
    load_dotenv()
    
    # Get model and voice paths from environment or use defaults
    model_path = os.environ.get("TTS_MODEL_FILE", "tts/onnx/kokoro-v1.0.onnx")
    voice_path = os.environ.get("TTS_VOICE_FILE", "tts/voices/voices-v1.0.bin")
    
    print(f"Using model path: {model_path}")
    print(f"Using voice path: {voice_path}")
    
    # Initialize KokoroDirect
    kokoro = KokoroDirect(
        model_path=model_path,
        voices_path=voice_path
    )
    
    # Get available voices
    voices = kokoro.get_voice_names()
    if not voices:
        print("Error: No voices available")
        return
    
    print(f"Available voices: {', '.join(voices[:5])}{'...' if len(voices) > 5 else ''}")
    
    # Use the first available voice
    test_voice = voices[0]
    print(f"Using voice: {test_voice}")
    
    # Create a temporary output path
    output_path = Path("test_output.wav")
    
    # Generate audio for a test phrase
    text = "Hello world, this is a test of the TTS system."
    print(f"Generating audio for text: '{text}'")
    
    result = kokoro.create(text, test_voice, out_path=output_path)
    
    # Verify the result
    print(f"Result type: {type(result)}")
    print(f"Result path: {result}")
    
    if isinstance(result, Path) and result.exists():
        file_size = result.stat().st_size
        print(f"File size: {file_size} bytes")
        
        # Get audio duration
        audio_info = sf.info(result)
        duration = audio_info.duration
        print(f"Audio duration: {duration:.2f} seconds")
        
        print("✅ TTS test successful!")
    else:
        print("❌ TTS test failed: Invalid result or file does not exist")

if __name__ == "__main__":
    main()
