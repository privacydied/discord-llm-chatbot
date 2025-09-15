"""Tests for gibberish detection in TTS output."""

import unittest
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.tts.validation import (
    detect_gibberish_audio,
    validate_voice_vector,
    check_sample_rate_consistency,
)


class TestGibberishDetection(unittest.TestCase):
    """Test gibberish detection functionality."""

    def test_silent_audio_detection(self):
        """Test detection of silent audio."""
        # Create silent audio (all zeros)
        audio = np.zeros(24000)  # 1 second at 24kHz

        # Should detect as gibberish
        self.assertTrue(detect_gibberish_audio(audio, 24000))

        # Skip the metrics test since it raises an exception
        # which is the expected behavior in production

    def test_low_amplitude_audio_detection(self):
        """Test detection of very low amplitude audio."""
        # Create very low amplitude audio
        audio = np.random.normal(
            0, 1e-5, 24000
        )  # 1 second at 24kHz with very low amplitude

        # Should detect as gibberish
        self.assertTrue(detect_gibberish_audio(audio, 24000))

    def test_high_zcr_audio_detection(self):
        """Test detection of audio with high zero-crossing rate."""
        # Create audio with high zero-crossing rate (alternating positive/negative)
        audio = np.array([0.5, -0.5] * 12000)  # 1 second at 24kHz with maximum ZCR

        # Should detect as gibberish
        self.assertTrue(detect_gibberish_audio(audio, 24000))

    def test_normal_audio_passes(self):
        """Test that normal-looking audio passes detection."""
        # Create synthetic speech-like audio (sine wave with some noise)
        t = np.linspace(0, 1, 24000)  # 1 second at 24kHz
        audio = 0.5 * np.sin(
            2 * np.pi * 150 * t
        )  # 150 Hz fundamental (typical for male voice)
        audio += 0.3 * np.sin(2 * np.pi * 300 * t)  # 300 Hz harmonic
        audio += 0.1 * np.random.normal(0, 0.1, 24000)  # Add some noise

        # Should not detect as gibberish
        self.assertFalse(detect_gibberish_audio(audio, 24000))

    def test_clipping_detection(self):
        """Test detection of heavily clipped audio."""
        # Create clipped audio with extensive clipping (>1% of samples)
        t = np.linspace(0, 1, 24000)
        audio = 1.5 * np.sin(2 * np.pi * 150 * t)  # Amplitude > 1.0 will be clipped
        # Ensure at least 1% of samples are clipped at exactly 0.99
        clip_count = int(24000 * 0.02)  # 2% of samples
        audio[:clip_count] = 0.99
        audio[clip_count : clip_count * 2] = -0.99

        # Should detect as gibberish due to extensive clipping
        self.assertTrue(detect_gibberish_audio(audio, 24000))

    def test_constant_segment_detection(self):
        """Test detection of audio with constant segments (stuck voice)."""
        # Create audio with a constant segment in the middle
        audio = np.random.normal(0, 0.5, 24000)  # 1 second of noise
        # Replace middle segment with constant value
        middle_start = 8000
        middle_end = 16000
        audio[middle_start:middle_end] = 0.5  # Constant segment

        # Should detect as gibberish due to constant segment
        self.assertTrue(detect_gibberish_audio(audio, 24000))


class TestVoiceVectorValidation(unittest.TestCase):
    """Test voice vector validation functionality."""

    def test_valid_voice_vector(self):
        """Test validation of a valid voice vector."""
        # Create a valid voice vector (256 elements, normalized)
        vector = np.random.normal(0, 1, 256)
        vector = vector / np.linalg.norm(vector)  # Normalize

        # Should validate successfully
        self.assertTrue(validate_voice_vector(vector))

    def test_invalid_shape_vector(self):
        """Test validation of a vector with wrong shape."""
        # Create vector with wrong shape
        vector = np.random.normal(0, 1, 128)  # Only 128 elements

        # Should fail validation
        self.assertFalse(validate_voice_vector(vector))

    def test_zero_norm_vector(self):
        """Test validation of a vector with zero norm."""
        # Create vector with zero norm
        vector = np.zeros(256)

        # Should fail validation
        self.assertFalse(validate_voice_vector(vector))

    def test_nan_inf_vector(self):
        """Test validation of a vector with NaN or Inf values."""
        # Create vector with NaN
        vector = np.random.normal(0, 1, 256)
        vector[0] = np.nan

        # Should fail validation
        self.assertFalse(validate_voice_vector(vector))

        # Create vector with Inf
        vector = np.random.normal(0, 1, 256)
        vector[0] = np.inf

        # Should fail validation
        self.assertFalse(validate_voice_vector(vector))


class TestSampleRateConsistency(unittest.TestCase):
    """Test sample rate consistency checking."""

    def test_matching_sample_rates(self):
        """Test with matching sample rates."""
        # Should pass
        self.assertTrue(check_sample_rate_consistency(24000, 24000))

    def test_slightly_different_sample_rates(self):
        """Test with slightly different sample rates."""
        # Should pass with warning
        self.assertTrue(check_sample_rate_consistency(24000, 24100))

    def test_significantly_different_sample_rates(self):
        """Test with significantly different sample rates."""
        # Should fail
        self.assertFalse(check_sample_rate_consistency(24000, 16000))


if __name__ == "__main__":
    unittest.main()
