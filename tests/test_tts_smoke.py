#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple smoke test for TTS functionality to verify audio generation works.
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

# Add the parent directory to the path so we can import bot modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot.kokoro_direct import KokoroDirect
from bot.tts_errors import TTSWriteError


class TTSSmokeTest(unittest.TestCase):
    """Smoke test for TTS functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Default paths for models - these should match what's in the .env file
        self.model_path = Path("tts/onnx/model.onnx")
        self.voice_path = Path("tts/voices/am_michael.npz")
        
        # Skip test if model files don't exist
        if not self.model_path.exists():
            self.skipTest(f"Model file not found: {self.model_path}")
        if not self.voice_path.exists():
            self.skipTest(f"Voice file not found: {self.voice_path}")
            
    def test_generate_audio(self):
        """Test that we can generate audio with the TTS system."""
        # Initialize KokoroDirect
        kokoro = KokoroDirect(
            model_path=self.model_path,
            language="en",
            phonemiser="espeak"
        )
        
        # Load a test voice
        kokoro.load_voice(self.voice_path)
        
        # Create a temporary output path
        output_path = Path("tests/test_output.wav")
        
        try:
            # Generate audio for a simple test phrase
            result = kokoro.create("ping", "am_michael", out_path=output_path)
            
            # Verify that result is a Path object
            self.assertIsInstance(result, Path)
            
            # Verify that the file exists
            self.assertTrue(result.exists())
            
            # Verify that the file has content (should be > 10,000 bytes for "ping")
            self.assertGreater(result.stat().st_size, 10000)
            
            print(f"Generated audio file: {result}, size: {result.stat().st_size} bytes")
            
        finally:
            # Clean up the test file
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    unittest.main()
