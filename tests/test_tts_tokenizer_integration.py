"""
Integration tests for TTS tokenizer registry integration.

These tests verify that the TTS system correctly uses the tokenizer registry
for selecting appropriate tokenizers based on language.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import os
import tempfile
from pathlib import Path
import numpy as np

# Import the modules under test
from bot.tokenizer_registry import TokenizerRegistry, discover_tokenizers
from bot.kokoro_direct_fixed import KokoroDirect
from bot.tts_errors import MissingTokeniserError


class TestTTSTokenizerIntegration(unittest.TestCase):
    """Test suite for TTS tokenizer integration."""

    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
        
        # Create a fresh registry for each test
        self.registry = TokenizerRegistry()
        TokenizerRegistry._instance = self.registry
        
        # Create temporary files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "model.onnx"
        self.voices_path = Path(self.temp_dir.name) / "voices.npz"
        
        # Create dummy files
        self.model_path.write_bytes(b"dummy model data")
        self.voices_path.write_bytes(b"dummy voices data")
        
    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Reset the singleton instance
        TokenizerRegistry._instance = None
        
        # Clean up temporary directory
        self.temp_dir.cleanup()

    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_model')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_voices')
    @patch('bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics')
    @patch('subprocess.run')
    def test_kokoro_uses_registry_for_english(self, mock_run, mock_dump, mock_load_voices, mock_load_model):
        """Test that KokoroDirect uses the tokenizer registry for English."""
        # Mock environment diagnostics to return espeak available
        mock_dump.return_value = {
            'espeak_binary': '/usr/bin/espeak',
            'espeak_ng_binary': None,
            'phonemizer_module': True,
            'g2p_en_module': False,
            'misaki_module': False
        }
        
        # Mock subprocess.run to simulate espeak being found
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Initialize registry
        registry = TokenizerRegistry.get_instance()
        available = registry.discover_tokenizers()
        
        # Verify espeak is available
        self.assertTrue('espeak' in registry._available_tokenizers)
        
        # Set language to English
        os.environ['TTS_LANGUAGE'] = 'en'
        
        # Create KokoroDirect instance
        with patch('bot.kokoro_direct_fixed.select_tokenizer_for_language', wraps=registry.select_tokenizer_for_language) as mock_select:
            kokoro = KokoroDirect(str(self.model_path), str(self.voices_path))
            
            # Call _select_phonemiser
            phonemiser = kokoro._select_phonemiser('en')
            
            # Verify select_tokenizer_for_language was called with 'en'
            # Note: It may be called multiple times during initialization
            mock_select.assert_any_call('en')
            
            # Verify espeak was selected
            self.assertEqual(phonemiser, 'espeak')

    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_model')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_voices')
    @patch('bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics')
    @patch('subprocess.run')
    def test_kokoro_uses_registry_for_japanese(self, mock_run, mock_dump, mock_load_voices, mock_load_model):
        """Test that KokoroDirect uses the tokenizer registry for Japanese."""
        # Mock environment diagnostics to return misaki available
        mock_dump.return_value = {
            'espeak_binary': None,
            'espeak_ng_binary': None,
            'phonemizer_module': False,
            'g2p_en_module': False,
            'misaki_module': True
        }
        
        # Initialize registry
        registry = TokenizerRegistry.get_instance()
        available = registry.discover_tokenizers()
        
        # Add misaki to available tokenizers
        registry._available_tokenizers.add('misaki')
        
        # Set language to Japanese
        os.environ['TTS_LANGUAGE'] = 'ja'
        
        # Create KokoroDirect instance
        with patch('bot.kokoro_direct_fixed.select_tokenizer_for_language', wraps=registry.select_tokenizer_for_language) as mock_select:
            kokoro = KokoroDirect(str(self.model_path), str(self.voices_path))
            
            # Call _select_phonemiser
            phonemiser = kokoro._select_phonemiser('ja')
            
            # Verify select_tokenizer_for_language was called with 'ja'
            # Note: It may be called multiple times during initialization
            mock_select.assert_any_call('ja')
            
            # Verify misaki was selected
            self.assertEqual(phonemiser, 'misaki')

    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_model')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_voices')
    @patch('bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics')
    def test_kokoro_uses_env_override(self, mock_dump, mock_load_voices, mock_load_model):
        """Test that KokoroDirect respects TTS_PHONEMISER environment variable."""
        # Mock environment diagnostics
        mock_dump.return_value = {
            'espeak_binary': '/usr/bin/espeak',
            'espeak_ng_binary': None,
            'phonemizer_module': True,
            'g2p_en_module': False,
            'misaki_module': False
        }
        
        # Initialize registry
        registry = TokenizerRegistry.get_instance()
        available = registry.discover_tokenizers()
        
        # Set environment variable override
        os.environ['TTS_PHONEMISER'] = 'custom_tokenizer'
        
        # Create KokoroDirect instance
        kokoro = KokoroDirect(str(self.model_path), str(self.voices_path))
        
        # Call _select_phonemiser
        phonemiser = kokoro._select_phonemiser('en')
        
        # Verify environment variable was respected
        self.assertEqual(phonemiser, 'custom_tokenizer')

    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_model')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_voices')
    @patch('bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics')
    def test_kokoro_fallback_on_registry_error(self, mock_dump, mock_load_voices, mock_load_model):
        """Test that KokoroDirect falls back to legacy logic if registry fails."""
        # Mock environment diagnostics to return no tokenizers
        mock_dump.return_value = {
            'espeak_binary': None,
            'espeak_ng_binary': None,
            'phonemizer_module': False,
            'g2p_en_module': False,
            'misaki_module': False
        }
        
        # Initialize registry
        registry = TokenizerRegistry.get_instance()
        available = registry.discover_tokenizers()
        
        # Only grapheme should be available
        self.assertEqual(registry._available_tokenizers, {'grapheme'})
        
        # Create KokoroDirect instance
        with patch('bot.kokoro_direct_fixed.select_tokenizer_for_language', side_effect=MissingTokeniserError("No tokenizer found")) as mock_select:
            kokoro = KokoroDirect(str(self.model_path), str(self.voices_path))
            
            # Call _select_phonemiser for English
            phonemiser = kokoro._select_phonemiser('en')
            
            # Verify fallback to espeak for English
            self.assertEqual(phonemiser, 'espeak')
            
            # Reset mock and set up for next test
            mock_select.reset_mock()
            mock_select.side_effect = MissingTokeniserError("No tokenizer found")
            
            # Call _select_phonemiser for Japanese
            phonemiser = kokoro._select_phonemiser('ja')
            
            # Verify select_tokenizer_for_language was called with 'ja'
            mock_select.assert_called_with('ja')
            
            # Verify fallback to misaki for Japanese
            self.assertEqual(phonemiser, 'misaki')


if __name__ == "__main__":
    unittest.main()
