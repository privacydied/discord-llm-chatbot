#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI script for testing TTS functionality.
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
        logger.info(f"Initializing TTS with model: {model_path}")
        kokoro = KokoroDirect(
            model_path=model_path,
            language="en",
            phonemiser="espeak"
        )
        
        # Load voice(s)
        if os.path.isdir(voice_path):
            # If voice_path is a directory, load all voices in it
            voice_dir = Path(voice_path)
            voice_files = list(voice_dir.glob("*.npz"))
            if not voice_files:
                logger.error(f"No voice files found in {voice_dir}")
                sys.exit(1)
            
            for voice_file in voice_files:
                logger.info(f"Loading voice: {voice_file.stem}")
                kokoro.load_voice(voice_file)
        else:
            # If voice_path is a file, load just that voice
            voice_file = Path(voice_path)
            if not voice_file.exists():
                logger.error(f"Voice file not found: {voice_file}")
                sys.exit(1)
            
            logger.info(f"Loading voice: {voice_file.stem}")
            kokoro.load_voice(voice_file)
        
        # Generate audio
        logger.info(f"Generating audio for text: '{args.text}'")
        result = kokoro.create(args.text, args.voice, out_path=output_path)
        
        # Verify the result
        if result and result.exists() and result.stat().st_size > 0:
            logger.info(f"✅ Generated audio saved to: {result}")
            logger.info(f"File size: {result.stat().st_size} bytes")
            
            # Try to get audio duration
            try:
                import soundfile as sf
                info = sf.info(result)
                logger.info(f"Audio duration: {info.duration:.2f}s, Sample rate: {info.samplerate}Hz")
            except Exception as e:
                logger.warning(f"Could not get audio info: {e}")
        else:
            logger.error(f"Failed to generate audio or file is empty: {result}")
            sys.exit(1)
            
    except TTSWriteError as e:
        logger.error(f"TTS error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
