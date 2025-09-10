"""
Test TTS environment variable compatibility and tokenizer selection.
"""

import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules we want to test
from bot.tts.manager_fixed import TTSManager
from bot.tts.kokoro_direct_fixed import KokoroDirect, TokenizationMethod


class TestTTSEnvVarCompat(unittest.TestCase):
    """Test TTS environment variable compatibility."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create mock config
        self.config = {
            "tts": {
                "model_path": str(self.temp_path / "model.onnx"),
                "voices_path": str(self.temp_path / "voices.bin"),
            }
        }

        # Save original environment
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Clean up temporary directory
        self.temp_dir.cleanup()

    @patch("bot.tts.manager_fixed.KokoroDirect")
    def test_new_env_vars(self, mock_kokoro):
        """Test that new environment variables are used correctly."""
        # Set new environment variables
        os.environ["TTS_MODEL_PATH"] = str(self.temp_path / "new_model.onnx")
        os.environ["TTS_VOICES_PATH"] = str(self.temp_path / "new_voices.bin")

        # Create TTSManager
        manager = TTSManager(self.config)

        # Load model (this will initialize KokoroDirect)
        manager.load_model()

        # Check that KokoroDirect was initialized with correct paths
        mock_kokoro.assert_called_once()
        args, kwargs = mock_kokoro.call_args
        self.assertEqual(kwargs["model_path"], str(self.temp_path / "new_model.onnx"))
        self.assertEqual(kwargs["voices_path"], str(self.temp_path / "new_voices.bin"))

    @patch("bot.tts.manager_fixed.KokoroDirect")
    def test_old_env_vars(self, mock_kokoro):
        """Test that old environment variables are used as fallback."""
        # Set old environment variables
        os.environ["TTS_MODEL_FILE"] = str(self.temp_path / "old_model.onnx")
        os.environ["TTS_VOICE_FILE"] = str(self.temp_path / "old_voices.bin")

        # Create TTSManager
        manager = TTSManager(self.config)

        # Load model (this will initialize KokoroDirect)
        manager.load_model()

        # Check that KokoroDirect was initialized with correct paths
        mock_kokoro.assert_called_once()
        args, kwargs = mock_kokoro.call_args
        self.assertEqual(kwargs["model_path"], str(self.temp_path / "old_model.onnx"))
        self.assertEqual(kwargs["voices_path"], str(self.temp_path / "old_voices.bin"))

    @patch("bot.tts.manager_fixed.KokoroDirect")
    def test_new_vars_override_old(self, mock_kokoro):
        """Test that new environment variables override old ones."""
        # Set both old and new environment variables
        os.environ["TTS_MODEL_FILE"] = str(self.temp_path / "old_model.onnx")
        os.environ["TTS_VOICE_FILE"] = str(self.temp_path / "old_voices.bin")
        os.environ["TTS_MODEL_PATH"] = str(self.temp_path / "new_model.onnx")
        os.environ["TTS_VOICES_PATH"] = str(self.temp_path / "new_voices.bin")

        # Create TTSManager
        manager = TTSManager(self.config)

        # Load model (this will initialize KokoroDirect)
        manager.load_model()

        # Check that KokoroDirect was initialized with new paths
        mock_kokoro.assert_called_once()
        args, kwargs = mock_kokoro.call_args
        self.assertEqual(kwargs["model_path"], str(self.temp_path / "new_model.onnx"))
        self.assertEqual(kwargs["voices_path"], str(self.temp_path / "new_voices.bin"))


class TestTokenizerSelection(unittest.TestCase):
    """Test tokenizer selection and discovery."""

    @patch("bot.tts.kokoro_direct_fixed.importlib.util.find_spec")
    @patch("bot.tts.kokoro_direct_fixed.shutil.which")
    def test_tokenizer_discovery(self, mock_which, mock_find_spec):
        """Test that tokenizer methods are correctly discovered."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock()
        mock_tokenizer.phoneme_to_id = MagicMock()

        # Mock external dependencies
        mock_which.return_value = "/usr/bin/espeak"  # espeak is available
        mock_find_spec.side_effect = (
            lambda name: MagicMock() if name == "phonemizer" else None
        )

        # Create KokoroDirect with mocked tokenizer
        with patch.object(KokoroDirect, "_load_model"):
            kokoro = KokoroDirect(
                model_path="dummy_model.onnx", voices_path="dummy_voices.bin"
            )
            kokoro.tokenizer = mock_tokenizer
            kokoro._detect_tokenization_methods()

        # Check that the correct methods were detected
        self.assertIn(
            TokenizationMethod.PHONEME_ENCODE, kokoro.available_tokenization_methods
        )
        self.assertIn(
            TokenizationMethod.PHONEME_TO_ID, kokoro.available_tokenization_methods
        )
        self.assertIn(TokenizationMethod.ESPEAK, kokoro.available_tokenization_methods)
        self.assertIn(
            TokenizationMethod.PHONEMIZER, kokoro.available_tokenization_methods
        )
        self.assertNotIn(
            TokenizationMethod.MISAKI, kokoro.available_tokenization_methods
        )

    def test_phonemizer_selection_english(self):
        """Test that the correct phonemizer is selected for English."""
        # Create KokoroDirect with mocked methods
        with (
            patch.object(KokoroDirect, "_load_model"),
            patch.object(KokoroDirect, "_detect_tokenization_methods"),
        ):
            kokoro = KokoroDirect(
                model_path="dummy_model.onnx", voices_path="dummy_voices.bin"
            )

            # Test English language
            phonemizer = kokoro._select_phonemiser("en")
            self.assertEqual(phonemizer, "espeak")

    def test_phonemizer_selection_japanese(self):
        """Test that the correct phonemizer is selected for Japanese."""
        # Create KokoroDirect with mocked methods
        with (
            patch.object(KokoroDirect, "_load_model"),
            patch.object(KokoroDirect, "_detect_tokenization_methods"),
        ):
            kokoro = KokoroDirect(
                model_path="dummy_model.onnx", voices_path="dummy_voices.bin"
            )

            # Test Japanese language
            phonemizer = kokoro._select_phonemiser("ja")
            self.assertEqual(phonemizer, "misaki")

    def test_phonemizer_env_override(self):
        """Test that environment variable can override phonemizer selection."""
        # Set environment variable
        os.environ["TTS_PHONEMISER"] = "custom_phonemizer"

        try:
            # Create KokoroDirect with mocked methods
            with (
                patch.object(KokoroDirect, "_load_model"),
                patch.object(KokoroDirect, "_detect_tokenization_methods"),
            ):
                kokoro = KokoroDirect(
                    model_path="dummy_model.onnx", voices_path="dummy_voices.bin"
                )

                # Test that environment variable overrides language-based selection
                phonemizer = kokoro._select_phonemiser("en")
                self.assertEqual(phonemizer, "custom_phonemizer")
        finally:
            # Clean up environment
            if "TTS_PHONEMISER" in os.environ:
                del os.environ["TTS_PHONEMISER"]


if __name__ == "__main__":
    unittest.main()
