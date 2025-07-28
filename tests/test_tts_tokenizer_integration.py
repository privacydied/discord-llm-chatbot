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


class TestTTSTokenizerIntegrationEspeakWrapper(unittest.TestCase):
    """Test TTS tokenizer integration with KokoroDirect and EspeakWrapper."""

    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
        
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
        
        # Clean up temporary directory
        self.temp_dir.cleanup()

    @patch('bot.tts.kokoro_bootstrap.register_espeak_wrapper')
    def test_espeak_wrapper_registration(self, mock_register):
        """Test that EspeakWrapper is registered under all required aliases."""
        # Import bootstrap module to register EspeakWrapper
        from bot.tts import TOKENIZER_ALIASES
        
        # Import kokoro_onnx modules to check registration
        with patch('kokoro_onnx.tokenizer_registry.AVAILABLE_TOKENIZERS', {}) as mock_tokenizers:
            # Call the registration function
            from bot.tts import register_espeak_wrapper
            register_espeak_wrapper()
            
            # Check that EspeakWrapper is registered under all required aliases
            for alias in TOKENIZER_ALIASES:
                self.assertIn(alias, mock_tokenizers, f"Alias '{alias}' not registered")
    
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_model')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_voices')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._write_audio')
    def test_kokoro_direct_tokenizer_initialization(self, mock_write, mock_load_voices, mock_load_model):
        """Test that KokoroDirect initializes with the correct tokenizer."""
        # Mock audio writing to return a valid path
        mock_write.return_value = Path("/tmp/test.wav")
        
        # Mock the tokenizer class
        mock_tokenizer = MagicMock()
        mock_tokenizer.__class__.__name__ = "EspeakWrapper"
        
        # Import KokoroDirect
        from bot.kokoro_direct_fixed import KokoroDirect
        
        # Initialize KokoroDirect with a mocked tokenizer
        with patch('bot.kokoro_direct_fixed.Tokenizer', return_value=mock_tokenizer):
            kokoro = KokoroDirect(
                model_path=str(self.model_path),
                voice_path=str(self.voices_path),
                cache_dir="tts/cache",
            )
            
            # Check that tokenizer is initialized
            self.assertIsNotNone(kokoro.tokenizer, "Kokoro tokenizer is None after initialization")
            self.assertEqual(kokoro.tokenizer.__class__.__name__, "EspeakWrapper", 
                f"Kokoro tokenizer is {kokoro.tokenizer.__class__.__name__}, expected EspeakWrapper")
    
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_model')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_voices')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._write_audio')
    def test_english_tokenizer_in_language_cache(self, mock_write, mock_load_voices, mock_load_model):
        """Test that English tokenizer is correctly set in language cache."""
        # Mock audio writing to return a valid path
        mock_write.return_value = Path("/tmp/test.wav")
        
        # Mock the tokenizer class
        mock_tokenizer = MagicMock()
        mock_tokenizer.__class__.__name__ = "EspeakWrapper"
        
        # Import KokoroDirect
        from bot.kokoro_direct_fixed import KokoroDirect
        
        # Initialize KokoroDirect with a mocked tokenizer
        with patch('bot.kokoro_direct_fixed.Tokenizer', return_value=mock_tokenizer):
            kokoro = KokoroDirect(
                model_path=str(self.model_path),
                voice_path=str(self.voices_path),
                cache_dir="tts/cache",
            )
            
            # Mock the tokenize method to return a valid phoneme sequence
            mock_tokenizer.tokenize.return_value = ["t", "ɛ", "s", "t"]
            
            # Generate text to ensure language cache is populated
            test_text = "This is a test of the English tokenizer."
            with patch('numpy.ndarray', MagicMock()):
                with patch('bot.kokoro_direct_fixed.KokoroDirect._generate_audio', return_value=np.zeros(1000)):
                    kokoro.create(test_text, "default", out_path=Path("/tmp/test.wav"))
            
            # Check that English tokenizer is in language cache
            self.assertTrue(hasattr(kokoro, '_lang_tokenizer_cache'), "KokoroDirect has no _lang_tokenizer_cache attribute")
            self.assertIn('en', kokoro._lang_tokenizer_cache, "English tokenizer not in language cache")
            self.assertIsNotNone(kokoro._lang_tokenizer_cache['en'], "English tokenizer in cache is None")
            self.assertEqual(kokoro._lang_tokenizer_cache['en'].__class__.__name__, "EspeakWrapper", 
                f"English tokenizer in cache is {kokoro._lang_tokenizer_cache['en'].__class__.__name__}, expected EspeakWrapper")
    
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_model')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._load_voices')
    @patch('bot.kokoro_direct_fixed.KokoroDirect._write_audio')
    def test_no_grapheme_fallback_warning(self, mock_write, mock_load_voices, mock_load_model):
        """Test that no grapheme fallback warnings appear during TTS generation."""
        # Mock audio writing to return a valid path
        mock_write.return_value = Path("/tmp/test.wav")
        
        # Mock the tokenizer class
        mock_tokenizer = MagicMock()
        mock_tokenizer.__class__.__name__ = "EspeakWrapper"
        
        # Import KokoroDirect
        from bot.kokoro_direct_fixed import KokoroDirect
        
        # Initialize KokoroDirect with a mocked tokenizer
        with patch('bot.kokoro_direct_fixed.Tokenizer', return_value=mock_tokenizer):
            kokoro = KokoroDirect(
                model_path=str(self.model_path),
                voice_path=str(self.voices_path),
                cache_dir="tts/cache",
            )
            
            # Mock the tokenize method to return a valid phoneme sequence
            mock_tokenizer.tokenize.return_value = ["t", "ɛ", "s", "t"]
            
            # Set up logging capture
            with self.assertLogs(level='WARNING') as log_capture:
                # Generate text
                test_text = "This is a test of the English tokenizer."
                with patch('numpy.ndarray', MagicMock()):
                    with patch('bot.kokoro_direct_fixed.KokoroDirect._generate_audio', return_value=np.zeros(1000)):
                        result = kokoro.create(test_text, "default", out_path=Path("/tmp/test.wav"))
                
                # Check that no grapheme fallback warnings were logged
                for record in log_capture.records:
                    self.assertNotIn("grapheme", record.getMessage().lower(), f"Found grapheme fallback warning: {record.getMessage()}")
                    self.assertNotIn("fallback", record.getMessage().lower(), f"Found fallback warning: {record.getMessage()}")
            
            # Check that the result is a Path
            self.assertIsInstance(result, Path, f"Result is not a Path, got {type(result)}")
    
    @patch('bot.tts_manager_fixed.TTSManager._init_tokenizer_registry')
    @patch('bot.tts_manager_fixed.TTSManager._load_kokoro')
    def test_tts_manager_integration(self, mock_load_kokoro, mock_init_registry):
        """Test TTSManager integration with kokoro_bootstrap and KokoroDirect."""
        # Import TTSManager
        from bot.tts_manager_fixed import TTSManager
        
        # Create a minimal config
        config = {
            'TTS_BACKEND': 'kokoro-onnx',
            'TTS_VOICE': 'default',
        }
        
        # Mock the KokoroDirect instance
        mock_kokoro = MagicMock()
        mock_kokoro.tokenizer.__class__.__name__ = "EspeakWrapper"
        mock_load_kokoro.return_value = mock_kokoro
        
        # Initialize TTSManager
        tts_manager = TTSManager(config)
        
        # Set available to True
        tts_manager.available = True
        tts_manager.kokoro = mock_kokoro
        
        # Mock generate_speech to return a valid path
        mock_kokoro.create.return_value = Path("/tmp/test.wav")
        
        # Generate speech
        test_text = "This is a test of the TTSManager integration."
        with patch('asyncio.run'):
            result_path = tts_manager.generate_speech(test_text)
            
            # Check that speech was generated
            self.assertIsNotNone(result_path, "TTSManager.generate_speech returned None")
            self.assertEqual(result_path, Path("/tmp/test.wav"), "Unexpected result path")
            
            # Verify that create was called with the right parameters
            mock_kokoro.create.assert_called_once()
            args, kwargs = mock_kokoro.create.call_args
            self.assertEqual(args[0], test_text, f"Expected text '{test_text}', got '{args[0]}'")


if __name__ == "__main__":
    unittest.main()
