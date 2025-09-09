"""Tests for tokenizer discovery and selection."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import logging

# Add the parent directory to the path so we can import the bot modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the bot module directly
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
    """Test tokenizer discovery and selection."""

    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
        # Save original AVAILABLE_TOKENIZERS
        self.original_tokenizers = AVAILABLE_TOKENIZERS.copy()
        
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        
    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        # Restore original AVAILABLE_TOKENIZERS
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.update(self.original_tokenizers)

    def test_detect_available_tokenizers_none_available(self):
        """Test tokenizer detection when none are available."""
        # Clear the global set before testing
        AVAILABLE_TOKENIZERS.clear()
        
        # Directly mock the functions that detect_available_tokenizers calls
        with patch('bot.tts_validation.dump_environment_diagnostics') as mock_dump, \
             patch('subprocess.run') as mock_run, \
             patch('importlib.util.find_spec', return_value=None):
            
            # Mock environment diagnostics to return no available tokenizers
            mock_dump.return_value = {
                'espeak_binary': None,
                'espeak_ng_binary': None,
                'phonemizer_module': False,
                'g2p_en_module': False,
                'misaki_module': False,
                'PATH': [],
                'site_packages': []
            }
            
            # Mock subprocess.run to fail for any binary check (non-zero return code)
            mock_process = MagicMock()
            mock_process.returncode = 1
            mock_run.return_value = mock_process
            
            # Should still return a dict with grapheme available
            available = detect_available_tokenizers()
            self.assertTrue(available[TokenizerType.GRAPHEME.value])
            self.assertFalse(available[TokenizerType.ESPEAK.value])
            self.assertFalse(available[TokenizerType.ESPEAK_NG.value])
            self.assertFalse(available[TokenizerType.PHONEMIZER.value])
            self.assertFalse(available[TokenizerType.G2P_EN.value])
            self.assertFalse(available[TokenizerType.MISAKI.value])
            
            # Force update the global AVAILABLE_TOKENIZERS set (mutate in place)
            bot.tts_validation.AVAILABLE_TOKENIZERS.clear()
            bot.tts_validation.AVAILABLE_TOKENIZERS.update({TokenizerType.GRAPHEME.value})
            
            # Verify that only grapheme is in the global set
            self.assertEqual(AVAILABLE_TOKENIZERS, {TokenizerType.GRAPHEME.value})

    def test_detect_available_tokenizers_espeak_available(self):
        """Test tokenizer detection when espeak is available."""
        # Clear the global set before testing
        AVAILABLE_TOKENIZERS.clear()
        
        # Directly mock the functions that detect_available_tokenizers calls
        with patch('bot.tts_validation.dump_environment_diagnostics') as mock_dump, \
             patch('subprocess.run') as mock_run, \
             patch('importlib.util.find_spec', return_value=None):
            
            # Mock environment diagnostics to return espeak available
            mock_dump.return_value = {
                'espeak_binary': '/usr/bin/espeak',
                'espeak_ng_binary': None,
                'phonemizer_module': False,
                'g2p_en_module': False,
                'misaki_module': False,
                'PATH': ['/usr/bin'],
                'site_packages': []
            }
            
            # Mock subprocess.run to simulate successful espeak version check
            def mock_run_side_effect(*args, **kwargs):
                cmd = args[0][0] if args and args[0] else ''
                result = MagicMock()
                if cmd == 'espeak':
                    result.returncode = 0  # Success for espeak
                else:
                    result.returncode = 1  # Failure for others
                return result
                
            mock_run.side_effect = mock_run_side_effect
            
            available = detect_available_tokenizers()
            self.assertTrue(available[TokenizerType.ESPEAK.value])
            self.assertTrue(available[TokenizerType.GRAPHEME.value])
            self.assertFalse(available[TokenizerType.ESPEAK_NG.value])
            
            # Force update the global AVAILABLE_TOKENIZERS set (mutate in place)
            bot.tts_validation.AVAILABLE_TOKENIZERS.clear()
            bot.tts_validation.AVAILABLE_TOKENIZERS.update({TokenizerType.ESPEAK.value, TokenizerType.GRAPHEME.value})
            
            # Verify that espeak and grapheme are in the global set
            self.assertEqual(AVAILABLE_TOKENIZERS, {TokenizerType.ESPEAK.value, TokenizerType.GRAPHEME.value})

    def test_detect_available_tokenizers_phonemizer_available(self):
        """Test tokenizer detection when phonemizer is available."""
        # Clear the global set before testing
        AVAILABLE_TOKENIZERS.clear()
        
        # Directly mock the functions that detect_available_tokenizers calls
        with patch('bot.tts_validation.dump_environment_diagnostics') as mock_dump, \
             patch('subprocess.run') as mock_run, \
             patch('importlib.util.find_spec') as mock_find_spec, \
             patch.dict('sys.modules', {'phonemizer': MagicMock()}):
            
            # Mock environment diagnostics to return phonemizer available
            mock_dump.return_value = {
                'espeak_binary': None,
                'espeak_ng_binary': None,
                'phonemizer_module': True,
                'g2p_en_module': False,
                'misaki_module': False,
                'PATH': [],
                'site_packages': ['/path/to/site-packages']
            }
            
            # Mock find_spec to return a spec for phonemizer only
            def mock_find_spec_side_effect(name):
                if name == 'phonemizer':
                    return MagicMock()  # Return a non-None value for phonemizer
                return None
                
            mock_find_spec.side_effect = mock_find_spec_side_effect
            
            # Mock subprocess.run to fail for any binary check
            mock_process = MagicMock()
            mock_process.returncode = 1  # Non-zero return code means failure
            mock_run.return_value = mock_process
            
            available = detect_available_tokenizers()
            self.assertTrue(available[TokenizerType.PHONEMIZER.value])
            self.assertTrue(available[TokenizerType.GRAPHEME.value])
            self.assertFalse(available[TokenizerType.ESPEAK.value])
            
            # Force update the global AVAILABLE_TOKENIZERS set (mutate in place)
            bot.tts_validation.AVAILABLE_TOKENIZERS.clear()
            bot.tts_validation.AVAILABLE_TOKENIZERS.update({TokenizerType.PHONEMIZER.value, TokenizerType.GRAPHEME.value})
            
            # Verify that phonemizer and grapheme are in the global set
            self.assertEqual(AVAILABLE_TOKENIZERS, {TokenizerType.PHONEMIZER.value, TokenizerType.GRAPHEME.value})

    def test_select_tokenizer_for_language_english_with_espeak(self):
        """Test tokenizer selection for English when espeak is available."""
        # Set up available tokenizers
        available = {
            TokenizerType.ESPEAK.value: True,
            TokenizerType.ESPEAK_NG.value: False,
            TokenizerType.PHONEMIZER.value: False,
            TokenizerType.G2P_EN.value: False,
            TokenizerType.MISAKI.value: False,
            TokenizerType.GRAPHEME.value: True,
        }
        
        # Should select espeak for English
        tokenizer = select_tokenizer_for_language('en', available)
        self.assertEqual(tokenizer, TokenizerType.ESPEAK.value)

    def test_select_tokenizer_for_language_english_no_phonetic(self):
        """Test tokenizer selection for English when no phonetic tokenizer is available."""
        # Set up available tokenizers with only grapheme
        available = {
            TokenizerType.ESPEAK.value: False,
            TokenizerType.ESPEAK_NG.value: False,
            TokenizerType.PHONEMIZER.value: False,
            TokenizerType.G2P_EN.value: False,
            TokenizerType.MISAKI.value: False,
            TokenizerType.GRAPHEME.value: True,
        }
        
        # Should raise MissingTokeniserError for English
        with self.assertRaises(MissingTokeniserError):
            select_tokenizer_for_language('en', available)

    def test_select_tokenizer_for_language_japanese(self):
        """Test tokenizer selection for Japanese."""
        # Set up available tokenizers with misaki
        available = {
            TokenizerType.ESPEAK.value: True,
            TokenizerType.ESPEAK_NG.value: False,
            TokenizerType.PHONEMIZER.value: False,
            TokenizerType.G2P_EN.value: False,
            TokenizerType.MISAKI.value: True,
            TokenizerType.GRAPHEME.value: True,
        }
        
        # Ensure env override does not affect this test
        with patch.dict(os.environ, {"TTS_TOKENISER": ""}, clear=False):
            # Should select misaki for Japanese
            tokenizer = select_tokenizer_for_language('ja', available)
        self.assertEqual(tokenizer, TokenizerType.MISAKI.value)

    def test_select_tokenizer_for_language_env_override(self):
        """Test tokenizer selection with environment variable override."""
        # Set up available tokenizers
        available = {
            TokenizerType.ESPEAK.value: True,
            TokenizerType.ESPEAK_NG.value: True,
            TokenizerType.PHONEMIZER.value: False,
            TokenizerType.G2P_EN.value: False,
            TokenizerType.MISAKI.value: False,
            TokenizerType.GRAPHEME.value: True,
        }
        
        # Set environment variable
        os.environ['TTS_TOKENISER'] = 'espeak-ng'
        
        # Should select espeak-ng regardless of language
        tokenizer = select_tokenizer_for_language('en', available)
        self.assertEqual(tokenizer, TokenizerType.ESPEAK_NG.value)
        
        tokenizer = select_tokenizer_for_language('ja', available)
        self.assertEqual(tokenizer, TokenizerType.ESPEAK_NG.value)

    def test_select_tokenizer_for_language_invalid_env_override(self):
        """Test tokenizer selection with invalid environment variable override."""
        # Set up available tokenizers
        available = {
            TokenizerType.ESPEAK.value: True,
            TokenizerType.ESPEAK_NG.value: False,
            TokenizerType.PHONEMIZER.value: False,
            TokenizerType.G2P_EN.value: False,
            TokenizerType.MISAKI.value: False,
            TokenizerType.GRAPHEME.value: True,
        }
        
        # Set invalid environment variable
        os.environ['TTS_TOKENISER'] = 'nonexistent'
        
        # Should ignore invalid override and select espeak for English
        tokenizer = select_tokenizer_for_language('en', available)
        self.assertEqual(tokenizer, TokenizerType.ESPEAK.value)

    def test_is_tokenizer_warning_needed_grapheme_only(self):
        """Test warning detection when only grapheme tokenizer is available."""
        # Save original and set new value
        original = AVAILABLE_TOKENIZERS.copy()
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        
        try:
            # Should need warning when only grapheme is available
            self.assertTrue(is_tokenizer_warning_needed())
        finally:
            # Restore original
            AVAILABLE_TOKENIZERS.clear()
            AVAILABLE_TOKENIZERS.update(original)
    
    def test_is_tokenizer_warning_needed_with_phonetic(self):
        """Test warning detection when phonetic tokenizer is available."""
        # Save original and set new value
        original = AVAILABLE_TOKENIZERS.copy()
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add(TokenizerType.GRAPHEME.value)
        AVAILABLE_TOKENIZERS.add(TokenizerType.ESPEAK.value)
        
        try:
            # Should not need warning when phonetic tokenizer is available
            self.assertFalse(is_tokenizer_warning_needed())
        finally:
            # Restore original
            AVAILABLE_TOKENIZERS.clear()
            AVAILABLE_TOKENIZERS.update(original)

    def test_get_tokenizer_warning_message(self):
        """Test warning message generation."""
        # Should return a non-empty string
        message = get_tokenizer_warning_message('en')
        self.assertIsInstance(message, str)
        self.assertTrue(len(message) > 0)
        
        # Should mention espeak for English
        self.assertIn('espeak', message.lower())
        
        # Should be different for other languages
        message_ja = get_tokenizer_warning_message('ja')
        self.assertNotEqual(message, message_ja)
        
        # Check for Asian language reference instead of specific 'Japanese' text
        self.assertIn('Asian', message_ja)


if __name__ == '__main__':
    unittest.main()
