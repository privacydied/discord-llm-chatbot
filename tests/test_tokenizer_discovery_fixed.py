import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import subprocess
import importlib.util

# Import the module under test
import bot.tts_validation
from bot.tts_validation import (
    TokenizerType,
    detect_available_tokenizers,
    select_tokenizer_for_language,
    is_tokenizer_warning_needed,
    get_tokenizer_warning_message,
    AVAILABLE_TOKENIZERS
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

    @patch('bot.tts_validation.dump_environment_diagnostics')
    @patch('subprocess.run')
    @patch('importlib.util.find_spec', return_value=None)
    def test_detect_available_tokenizers_none_available(self, mock_find_spec, mock_run, mock_dump):
        """Test tokenizer detection when none are available."""
        # Mock environment diagnostics to return no available tokenizers
        mock_dump.return_value = {
            'espeak_binary': None,
            'espeak_ng_binary': None,
            'phonemizer_module': False,
            'g2p_en_module': False,
            'misaki_module': False
        }
        
        # Mock subprocess.run to simulate espeak not being found
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'espeak'")
        
        # Clear global state before test
        AVAILABLE_TOKENIZERS.clear()
        
        # Call the function under test
        available = detect_available_tokenizers()
        
        # Verify the returned dictionary
        self.assertFalse(available[TokenizerType.ESPEAK.value])
        self.assertFalse(available[TokenizerType.ESPEAK_NG.value])
        self.assertFalse(available[TokenizerType.PHONEMIZER.value])
        self.assertFalse(available[TokenizerType.G2P_EN.value])
        self.assertFalse(available[TokenizerType.MISAKI.value])
        self.assertTrue(available[TokenizerType.GRAPHEME.value])
        
        # Verify the global set was updated correctly by the function
        self.assertEqual(AVAILABLE_TOKENIZERS, {TokenizerType.GRAPHEME.value})

    @patch('bot.tts_validation.dump_environment_diagnostics')
    @patch('subprocess.run')
    @patch('importlib.util.find_spec', return_value=None)
    def test_detect_available_tokenizers_espeak_available(self, mock_find_spec, mock_run, mock_dump):
        """Test tokenizer detection when espeak is available."""
        # Mock environment diagnostics to return espeak available
        mock_dump.return_value = {
            'espeak_binary': '/usr/bin/espeak',
            'espeak_ng_binary': None,
            'phonemizer_module': False,
            'g2p_en_module': False,
            'misaki_module': False
        }
        
        # Mock subprocess.run to simulate espeak being found
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Clear global state before test
        AVAILABLE_TOKENIZERS.clear()
        
        # Call the function under test
        available = detect_available_tokenizers()
        
        # Verify the returned dictionary
        self.assertTrue(available[TokenizerType.ESPEAK.value])
        self.assertFalse(available[TokenizerType.ESPEAK_NG.value])
        self.assertFalse(available[TokenizerType.PHONEMIZER.value])
        self.assertFalse(available[TokenizerType.G2P_EN.value])
        self.assertFalse(available[TokenizerType.MISAKI.value])
        self.assertTrue(available[TokenizerType.GRAPHEME.value])
        
        # Verify the global set was updated correctly by the function
        self.assertEqual(AVAILABLE_TOKENIZERS, {TokenizerType.ESPEAK.value, TokenizerType.GRAPHEME.value})

    @patch('bot.tts_validation.dump_environment_diagnostics')
    @patch('subprocess.run')
    @patch('importlib.util.find_spec')
    @patch.dict('sys.modules', {'phonemizer': MagicMock()})
    def test_detect_available_tokenizers_phonemizer_available(self, mock_find_spec, mock_run, mock_dump):
        """Test tokenizer detection when phonemizer is available."""
        # Mock environment diagnostics to return phonemizer available
        mock_dump.return_value = {
            'espeak_binary': None,
            'espeak_ng_binary': None,
            'phonemizer_module': True,
            'g2p_en_module': False,
            'misaki_module': False
        }
        
        # Mock find_spec to return a spec for phonemizer only
        def mock_find_spec_side_effect(name):
            if name == 'phonemizer':
                return MagicMock()
            return None
        mock_find_spec.side_effect = mock_find_spec_side_effect
        
        # Clear global state before test
        AVAILABLE_TOKENIZERS.clear()
        
        # Mock the phonemizer import
        with patch.dict('sys.modules', {'phonemizer': MagicMock()}):
            # Call the function under test
            available = detect_available_tokenizers()
        
            # Verify the returned dictionary
            self.assertFalse(available[TokenizerType.ESPEAK.value])
            self.assertFalse(available[TokenizerType.ESPEAK_NG.value])
            self.assertTrue(available[TokenizerType.PHONEMIZER.value])
            self.assertFalse(available[TokenizerType.G2P_EN.value])
            self.assertFalse(available[TokenizerType.MISAKI.value])
            self.assertTrue(available[TokenizerType.GRAPHEME.value])
        
            # Verify the global set was updated correctly by the function
            self.assertEqual(AVAILABLE_TOKENIZERS, {TokenizerType.PHONEMIZER.value, TokenizerType.GRAPHEME.value})

    def test_select_tokenizer_for_language_english_with_espeak(self):
        """Test tokenizer selection for English when espeak is available."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.ESPEAK.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        # Test selection for English
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = select_tokenizer_for_language('en')
            # According to TOKENISER_MAP, espeak is the first choice for English
            self.assertEqual(tokenizer, TokenizerType.ESPEAK.value)

    def test_select_tokenizer_for_language_english_with_phonemizer(self):
        """Test tokenizer selection for English when phonemizer is available."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.PHONEMIZER.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        # Test selection for English
        tokenizer = select_tokenizer_for_language('en')
        self.assertEqual(tokenizer, TokenizerType.PHONEMIZER.value)

    def test_select_tokenizer_for_language_english_with_g2p_en(self):
        """Test tokenizer selection for English when g2p_en is available."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.G2P_EN.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        # Test selection for English
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = select_tokenizer_for_language('en')
            # According to TOKENISER_MAP, g2p_en is the preferred choice when available for English
            self.assertEqual(tokenizer, TokenizerType.G2P_EN.value)

    def test_select_tokenizer_for_language_english_fallback(self):
        """Test tokenizer selection for English when only grapheme is available."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        # Test selection for English
        # Clear any environment variables that might affect the test
        with patch.dict(os.environ, {}, clear=True):
            # For English, when only grapheme is available, it should raise MissingTokeniserError
            with self.assertRaises(MissingTokeniserError):
                select_tokenizer_for_language('en')

    def test_select_tokenizer_for_language_japanese(self):
        """Test tokenizer selection for Japanese."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.MISAKI.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        # Test selection for Japanese
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = select_tokenizer_for_language('ja')
            # For Japanese, misaki is the preferred tokenizer
            self.assertEqual(tokenizer, TokenizerType.MISAKI.value)

    def test_select_tokenizer_for_language_env_override(self):
        """Test tokenizer selection when TTS_TOKENISER environment variable is set."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.ESPEAK.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.ESPEAK_NG.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        # Set environment variable
        with patch.dict(os.environ, {'TTS_TOKENISER': 'espeak-ng'}):
            # Test selection for any language
            tokenizer = select_tokenizer_for_language('en')
            # Environment variable should override the default selection
            self.assertEqual(tokenizer, TokenizerType.ESPEAK_NG.value)

    def test_select_tokenizer_for_language_env_override_invalid(self):
        """Test tokenizer selection when TTS_TOKENISER environment variable is invalid."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.ESPEAK.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        # Set environment variable to a non-existent tokenizer
        with patch.dict(os.environ, {'TTS_TOKENISER': 'nonexistent'}):
            # Test selection for English - should fall back to auto-selection
            tokenizer = select_tokenizer_for_language('en')
            # Should fall back to the first available tokenizer in TOKENISER_MAP for English
            self.assertEqual(tokenizer, TokenizerType.ESPEAK.value)

    def test_is_tokenizer_warning_needed_grapheme_only(self):
        """Test warning detection when only grapheme tokenizer is available."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        # Set TTS_LANGUAGE to English
        with patch.dict(os.environ, {'TTS_LANGUAGE': 'en'}):
            # Should need warning when only grapheme is available for English
            self.assertTrue(is_tokenizer_warning_needed())

    def test_is_tokenizer_warning_needed_with_phonetic(self):
        """Test warning detection when phonetic tokenizer is available."""
        # Set up available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.ESPEAK.value)
        
        # Should not need warning when phonetic tokenizer is available
        self.assertFalse(is_tokenizer_warning_needed())

    def test_get_tokenizer_warning_message(self):
        """Test warning message generation."""
        message = get_tokenizer_warning_message()
        self.assertIn("English phonetic tokeniser", message)
        self.assertIn("espeak", message)
        self.assertIn("phonemizer", message)


if __name__ == '__main__':
    unittest.main()
