import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import importlib.util

# Import the module under test
from bot.tts_validation import (
    TokenizerType,
    detect_available_tokenizers,
    select_tokenizer_for_language,
    is_tokenizer_warning_needed,
    get_tokenizer_warning_message,
    AVAILABLE_TOKENIZERS,
    TOKENISER_MAP
)
from bot.tts_errors import MissingTokeniserError


class TestTokenizerDiscovery(unittest.TestCase):
    """Test suite for tokenizer discovery and selection functionality."""

    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
        
        # Save original AVAILABLE_TOKENIZERS
        self.original_tokenizers = set(AVAILABLE_TOKENIZERS)
        
        # Clear AVAILABLE_TOKENIZERS for each test
        AVAILABLE_TOKENIZERS.clear()
        
    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Restore original AVAILABLE_TOKENIZERS
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.update(self.original_tokenizers)

    def test_tokeniser_map_structure(self):
        """Verify the structure of the TOKENISER_MAP constant."""
        # Check English tokenizer preferences
        self.assertEqual(TOKENISER_MAP["en"][0], "espeak")
        self.assertEqual(TOKENISER_MAP["en"][1], "espeak-ng")
        self.assertEqual(TOKENISER_MAP["en"][2], "phonemizer")
        self.assertEqual(TOKENISER_MAP["en"][3], "g2p_en")
        
        # Check Japanese tokenizer preferences
        self.assertEqual(TOKENISER_MAP["ja"][0], "misaki")
        
        # Check default tokenizer preferences
        self.assertEqual(TOKENISER_MAP["*"][0], "phonemizer")
        self.assertEqual(TOKENISER_MAP["*"][1], "espeak")
        self.assertEqual(TOKENISER_MAP["*"][2], "espeak-ng")

    def test_select_tokenizer_english_with_espeak(self):
        """Test tokenizer selection for English when espeak is available."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("espeak")
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with clean environment
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = select_tokenizer_for_language("en")
            self.assertEqual(tokenizer, "espeak")

    def test_select_tokenizer_english_with_phonemizer(self):
        """Test tokenizer selection for English when phonemizer is available."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("phonemizer")
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with clean environment
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = select_tokenizer_for_language("en")
            self.assertEqual(tokenizer, "phonemizer")

    def test_select_tokenizer_english_with_g2p_en(self):
        """Test tokenizer selection for English when g2p_en is available."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("g2p_en")
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with clean environment
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = select_tokenizer_for_language("en")
            self.assertEqual(tokenizer, "g2p_en")

    def test_select_tokenizer_english_fallback(self):
        """Test tokenizer selection for English when only grapheme is available."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with clean environment
        with patch.dict(os.environ, {}, clear=True):
            # For English, when only grapheme is available, it should raise MissingTokeniserError
            with self.assertRaises(MissingTokeniserError):
                select_tokenizer_for_language("en")

    def test_select_tokenizer_japanese(self):
        """Test tokenizer selection for Japanese."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("misaki")
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with clean environment
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = select_tokenizer_for_language("ja")
            self.assertEqual(tokenizer, "misaki")

    def test_select_tokenizer_japanese_fallback(self):
        """Test tokenizer selection for Japanese when misaki is not available."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("phonemizer")
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with clean environment
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = select_tokenizer_for_language("ja")
            # Actual behavior: falls back to grapheme when misaki is not available
            self.assertEqual(tokenizer, "grapheme")

    def test_select_tokenizer_env_override(self):
        """Test tokenizer selection when TTS_TOKENISER environment variable is set."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("espeak")
        AVAILABLE_TOKENIZERS.add("espeak-ng")
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with environment variable
        with patch.dict(os.environ, {"TTS_TOKENISER": "espeak-ng"}):
            tokenizer = select_tokenizer_for_language("en")
            self.assertEqual(tokenizer, "espeak-ng")

    def test_select_tokenizer_env_override_invalid(self):
        """Test tokenizer selection when TTS_TOKENISER environment variable is invalid."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("espeak")
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with invalid environment variable
        with patch.dict(os.environ, {"TTS_TOKENISER": "nonexistent"}):
            tokenizer = select_tokenizer_for_language("en")
            # Should fall back to first available tokenizer in preference list
            self.assertEqual(tokenizer, "espeak")

    def test_is_tokenizer_warning_needed_grapheme_only(self):
        """Test warning detection when only grapheme tokenizer is available."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with English language
        with patch.dict(os.environ, {"TTS_LANGUAGE": "en"}):
            self.assertTrue(is_tokenizer_warning_needed())

    def test_is_tokenizer_warning_needed_with_phonetic(self):
        """Test warning detection when phonetic tokenizer is available."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("espeak")
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with English language
        with patch.dict(os.environ, {"TTS_LANGUAGE": "en"}):
            self.assertFalse(is_tokenizer_warning_needed())

    def test_is_tokenizer_warning_needed_japanese(self):
        """Test warning detection for Japanese without misaki."""
        # Set up available tokenizers directly
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Test with Japanese language
        with patch.dict(os.environ, {"TTS_LANGUAGE": "ja"}):
            self.assertTrue(is_tokenizer_warning_needed())

    def test_get_tokenizer_warning_message(self):
        """Test warning message generation."""
        # Test English warning message
        message = get_tokenizer_warning_message("en")
        self.assertIn("English phonetic tokeniser", message)
        self.assertIn("espeak", message)
        self.assertIn("phonemizer", message)
        
        # Test Japanese warning message
        message = get_tokenizer_warning_message("ja")
        self.assertIn("Asian language tokenizer", message)
        self.assertIn("misaki", message)


if __name__ == "__main__":
    unittest.main()
