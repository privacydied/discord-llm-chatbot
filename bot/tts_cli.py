#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI script for testing TTS functionality.

This script verifies that:
1. The TTS pipeline always returns a valid Path object
2. The audio file exists and is not empty
3. The audio file is large enough to be playable (>10kB for 'ping')
4. The audio file has a duration of at least 1 second

Usage: python -m bot.tts_cli "hello world"
"""
import argparse
import os
import sys
from pathlib import Path

import dotenv

from bot.kokoro_direct import KokoroDirect
from bot.tts_errors import TTSWriteError
from bot.logger import get_logger

logger = get_logger(__name__)

def main():
    """Main entry point for the CLI script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate TTS audio from text")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--voice", default="am_michael", help="Voice ID to use")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("--model", help="Path to ONNX model file (optional)")
    parser.add_argument("--voice-file", help="Path to voice file (optional)")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    dotenv.load_dotenv()
    
    # Get model and voice paths from arguments or environment variables
    model_path = args.model or os.environ.get("TTS_MODEL_FILE", "tts/onnx/model.onnx")
    voice_path = args.voice_file or os.environ.get("TTS_VOICE_FILE", "tts/voices")
    
    # Create output path if specified
    output_path = None
    if args.output:
        output_path = Path(args.output)
    
    # Initialize KokoroDirect
    try:
        logger.info(f"Initializing TTS with model: {model_path}, voices: {voice_path}")
        kokoro = KokoroDirect(
            model_path=model_path,
            voices_path=voice_path
        )
        
        # Get available voices
        voices = kokoro.get_voice_names()
        if not voices:
            logger.error("No voices available in the voice file")
            sys.exit(1)
            
        logger.info(f"Loaded {len(voices)} voices: {', '.join(voices[:5])}{'...' if len(voices) > 5 else ''}")
        
        # Check if requested voice is available
        if args.voice not in voices:
            logger.warning(f"Requested voice '{args.voice}' not found, using '{voices[0]}' instead")
            args.voice = voices[0]
        
        # Generate audio
        logger.info(f"Generating audio for text: '{args.text}'")
        start_time = os.times().user
        result = kokoro.create(args.text, args.voice, out_path=output_path)
        generation_time = os.times().user - start_time
        
        # Verify the result is a Path object
        if not isinstance(result, Path):
            logger.error(f"❌ TTS pipeline returned {type(result)}, expected Path object")
            sys.exit(1)
            
        # Verify the file exists
        if not result.exists():
            logger.error(f"❌ Output file does not exist: {result}")
            sys.exit(1)
            
        # Verify the file has content
        file_size = result.stat().st_size
        if file_size == 0:
            logger.error(f"❌ Output file is empty: {result}")
            sys.exit(1)
            
        # Verify minimum file size (10kB for short phrases)
        min_size = 10000  # 10kB
        if file_size < min_size:
            logger.error(f"❌ File size too small: {file_size} bytes (minimum: {min_size} bytes)")
            sys.exit(1)
            
        logger.info(f"✅ Generated audio saved to: {result}")
        logger.info(f"File size: {file_size} bytes")
        
        # Try to get audio duration
        try:
            import soundfile as sf
            info = sf.info(result)
            duration = info.duration
            
            # Verify minimum duration (1 second)
            min_duration = 1.0  # 1 second
            if duration < min_duration:
                logger.error(f"❌ Audio duration too short: {duration:.2f}s (minimum: {min_duration:.1f}s)")
                sys.exit(1)
                
            logger.info(f"✅ Audio duration: {duration:.2f}s, Sample rate: {info.samplerate}Hz")
            logger.info(f"Generation speed: {duration/generation_time:.2f}x real-time")
        except Exception as e:
            logger.warning(f"Could not get audio info: {e}")
            
    except TTSWriteError as e:
        logger.error(f"TTS error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
