#!/usr/bin/env python3
"""
Test the TTS functionality directly to verify our fixes.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import bot modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import TTS manager
from bot.tts import TTSManager
from dotenv import load_dotenv

async def main():
    """Test TTS functionality."""
    print("🔍 Testing Kokoro-ONNX TTS functionality...")
    
    # Load environment variables
    load_dotenv()
    
    # Create config dict from env vars
    config = {
        "TTS_BACKEND": os.getenv("TTS_BACKEND", "kokoro-onnx"),
        "TTS_VOICE": os.getenv("TTS_VOICE", "am_michael"),
        "TTS_VOICE_FILE": os.getenv("TTS_VOICE_FILE", "tts/voices.json"),
        "TTS_MODEL_FILE": os.getenv("TTS_MODEL_FILE", "tts/kokoro-v1.0.onnx"),
        "TTS_CACHE_DIR": os.getenv("TTS_CACHE_DIR", "tts_cache"),
    }
    
    print(f"Config: {config}")
    
    # Initialize TTS manager
    tts_manager = TTSManager(config)
    
    # Check if TTS is available
    if not tts_manager.available:
        print(f"❌ TTS is not available: {tts_manager.backend}")
        return False
    
    print(f"✅ TTS is available: {tts_manager.backend}")
    print(f"Using voice: {tts_manager.voice}")
    
    # Generate TTS for a test sentence
    text = "Hello! This is a test of the Kokoro-ONNX text-to-speech system."
    print(f"Generating TTS for: '{text}'")
    
    try:
        # Generate TTS
        output_path = await tts_manager.generate_tts(text, tts_manager.voice)
        
        print(f"✅ TTS generated successfully: {output_path}")
        print(f"File size: {output_path.stat().st_size} bytes")
        
        # Optional: Play the audio if on Linux with aplay
        try:
            import subprocess
            print("Playing audio...")
            subprocess.run(["aplay", str(output_path)], check=True)
            print("Audio playback complete")
        except Exception as e:
            print(f"Note: Audio playback failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())
