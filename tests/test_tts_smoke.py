#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple smoke test for TTS functionality to verify audio generation works.

This test verifies that:
1. The TTS pipeline always returns a valid Path object
2. The audio file exists and is not empty
3. The audio file is large enough to be playable (>10kB for 'ping')
4. The audio file has a duration of at least 1 second
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

# Add the parent directory to the path so we can import bot modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot.kokoro_direct_fixed import KokoroDirect
import soundfile as sf


class TTSSmokeTest(unittest.TestCase):
    """Smoke test for TTS functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Get paths from environment variables or use defaults
        self.model_path = Path(os.environ.get("TTS_MODEL_FILE", "tts/onnx/model.onnx"))
        self.voice_path = Path(os.environ.get("TTS_VOICE_FILE", "tts/voices/voices.npz"))
        
        # Log paths for debugging
        print(f"Using model path: {self.model_path}")
        print(f"Using voice path: {self.voice_path}")
        
        # Skip test if model files don't exist
        if not self.model_path.exists():
            self.skipTest(f"Model file not found: {self.model_path}")
        if not self.voice_path.exists():
            self.skipTest(f"Voice file not found: {self.voice_path}")
            
    def test_generate_audio(self):
        """Test that we can generate audio with the TTS system."""
        # Initialize KokoroDirect
        kokoro = KokoroDirect(
            model_path=str(self.model_path),
            voices_path=str(self.voice_path)
        )
        
        # Create a temporary output path
        output_path = Path("tests/test_output.wav")
        
        try:
            # Get available voices
            voices = kokoro.get_voice_names()
            self.assertTrue(len(voices) > 0, "No voices available")
            print(f"Available voices: {', '.join(voices[:5])}{'...' if len(voices) > 5 else ''}")
            
            # Use the first available voice
            test_voice = voices[0]
            print(f"Using voice: {test_voice}")
            
            # Generate audio for a simple test phrase
            result = kokoro.create("ping", test_voice, out_path=output_path)
            
            # Verify that result is a Path object
            self.assertIsInstance(result, Path, 
                                 f"Expected Path object, got {type(result)} instead")
            
            # Verify that the file exists
            self.assertTrue(result.exists(), 
                           f"Output file {result} does not exist")
            
            # Verify that the file has content (should be > 10,000 bytes for "ping")
            file_size = result.stat().st_size
            self.assertGreater(file_size, 10000, 
                              f"File size ({file_size} bytes) is too small")
            
            # Verify audio duration is at least 1 second
            audio_info = sf.info(result)
            duration = audio_info.duration
            self.assertGreaterEqual(duration, 1.0, 
                                   f"Audio duration ({duration:.2f}s) is too short")
            
            # Verify that the audio is not silent (not all zeros or close to zero)
            audio_data, _ = sf.read(result)
            max_amplitude = np.max(np.abs(audio_data))
            self.assertGreater(max_amplitude, 0.01, 
                              f"Audio appears to be silent (max amplitude: {max_amplitude})")
            
            # Calculate RMS to detect extremely quiet audio
            rms = np.sqrt(np.mean(np.square(audio_data)))
            self.assertGreater(rms, 0.001, 
                              f"Audio is too quiet (RMS: {rms})")
            
            print(f"Generated audio file: {result}, size: {file_size} bytes, "
                  f"duration: {duration:.2f}s, sample rate: {audio_info.samplerate}Hz, "
                  f"max amplitude: {max_amplitude:.4f}, RMS: {rms:.4f}")
            
        finally:
            # Clean up the test file
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    unittest.main()
