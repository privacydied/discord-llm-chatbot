"""Integration tests for TTS tokenizer registry integration.

Tests for TTS tokenizer integration with espeak-ng and other phoneme systems.
"""

import pytest

# Skip all tests — require system-level tokenizer binaries (espeak-ng)
pytestmark = pytest.mark.skip(reason="Requires system-level tokenizer binaries (espeak-ng)")

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Import the modules under test
from bot.tokenizer_registry import TokenizerRegistry
from bot.tts.errors import MissingTokeniserError
from bot.tts.kokoro_direct import KokoroDirect


class TestTTSTokenizerIntegration(unittest.TestCase):
    """Test suite for TTS tokenizer integration."""

    def setUp(self) -> None:
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

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Reset the singleton instance
        TokenizerRegistry._instance = None

        # Clean up temporary directory
        self.temp_dir.cleanup()

    @patch("bot.tts.kokoro_direct.KokoroDirect._load_model")
    @patch("bot.tts.kokoro_direct.KokoroDirect._load_voices")
    def test_kokoro_english_avoids_registry(self, mock_load_voices, mock_load_model) -> None:
        """English path should not call the tokenizer registry and should select 'espeak' placeholder."""
        # Set language to English
        os.environ["TTS_LANGUAGE"] = "en"

        # Create KokoroDirect instance and ensure registry is NOT consulted
        with patch("bot.tokenizer_registry.select_tokenizer_for_language") as mock_select:
            kokoro = KokoroDirect(str(self.model_path), str(self.voices_path))
            # Call _select_phonemiser
            phonemiser = kokoro._select_phonemiser("en")

            mock_select.assert_not_called()
            assert phonemiser == "espeak"

    @patch("bot.tts.kokoro_direct.KokoroDirect._load_model")
    @patch("bot.tts.kokoro_direct.KokoroDirect._load_voices")
    def test_kokoro_japanese_avoids_registry(self, mock_load_voices, mock_load_model) -> None:
        """Non-English KokoroDirect selection should not require registry; returns 'misaki' for ja."""
        os.environ["TTS_LANGUAGE"] = "ja"

        with patch("bot.tokenizer_registry.select_tokenizer_for_language") as mock_select:
            kokoro = KokoroDirect(str(self.model_path), str(self.voices_path))
            phonemiser = kokoro._select_phonemiser("ja")

            mock_select.assert_not_called()
            assert phonemiser == "misaki"

    @patch("bot.tts.kokoro_direct.KokoroDirect._load_model")
    @patch("bot.tts.kokoro_direct.KokoroDirect._load_voices")
    def test_kokoro_uses_env_override(self, mock_load_voices, mock_load_model) -> None:
        """Test that KokoroDirect respects TTS_PHONEMISER environment variable."""
        # Set environment variable override
        os.environ["TTS_PHONEMISER"] = "custom_tokenizer"

        # Create KokoroDirect instance
        kokoro = KokoroDirect(str(self.model_path), str(self.voices_path))

        # Call _select_phonemiser
        phonemiser = kokoro._select_phonemiser("en")

        # Verify environment variable was respected
        assert phonemiser == "custom_tokenizer"

    @patch("bot.tts.kokoro_direct.KokoroDirect._load_model")
    @patch("bot.tts.kokoro_direct.KokoroDirect._load_voices")
    def test_kokoro_independent_of_registry_errors(self, mock_load_voices, mock_load_model) -> None:
        """KokoroDirect should not rely on registry; even if registry fails, selection proceeds."""
        with patch(
            "bot.tokenizer_registry.select_tokenizer_for_language",
            side_effect=MissingTokeniserError("No tokenizer found"),
        ) as mock_select:
            kokoro = KokoroDirect(str(self.model_path), str(self.voices_path))

            # English selection remains 'espeak' without consulting registry
            phonemiser_en = kokoro._select_phonemiser("en")
            assert phonemiser_en == "espeak"
            mock_select.assert_not_called()

            # Japanese selection remains 'misaki' without consulting registry
            phonemiser_ja = kokoro._select_phonemiser("ja")
            assert phonemiser_ja == "misaki"
            mock_select.assert_not_called()


class TestTTSTokenizerIntegrationEspeakWrapper(unittest.TestCase):
    """Test TTS tokenizer integration with KokoroDirect and EspeakWrapper."""

    def setUp(self) -> None:
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

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Clean up temporary directory
        self.temp_dir.cleanup()

    @patch("bot.tts.kokoro_direct.KokoroDirect._load_model")
    @patch("bot.tts.kokoro_direct.KokoroDirect._load_voices")
    @patch("bot.tts.kokoro_direct.KokoroDirect._save_audio_to_wav")
    def test_kokoro_direct_tokenizer_initialization(self, mock_save, mock_load_voices, mock_load_model) -> None:
        """Test that KokoroDirect initializes with the correct tokenizer."""
        # Mock audio writing to return a valid path
        mock_save.return_value = None

        # Mock the tokenizer class
        mock_tokenizer = MagicMock()
        mock_tokenizer.__class__.__name__ = "EspeakWrapper"

        # Import KokoroDirect
        from bot.tts.kokoro_direct import KokoroDirect

        # Initialize KokoroDirect with a mocked tokenizer
        with patch("kokoro_onnx.tokenizer.Tokenizer", return_value=mock_tokenizer):
            kokoro = KokoroDirect(
                str(self.model_path),
                str(self.voices_path),
            )

            # Check that tokenizer is initialized
            assert kokoro.tokenizer is not None, "Kokoro tokenizer is None after initialization"
            assert kokoro.tokenizer.__class__.__name__ == "EspeakWrapper", f"Kokoro tokenizer is {kokoro.tokenizer.__class__.__name__}, expected EspeakWrapper"

    @patch("bot.tts.kokoro_direct.KokoroDirect._load_model")
    @patch("bot.tts.kokoro_direct.KokoroDirect._load_voices")
    @patch("bot.tts.kokoro_direct.KokoroDirect._save_audio_to_wav")
    def test_english_tokenizer_initialized(self, mock_save, mock_load_voices, mock_load_model) -> None:
        """Test that English tokenizer is initialized and usable (no cache assumptions)."""
        mock_save.return_value = None

        # Mock the tokenizer class
        mock_tokenizer = MagicMock()
        mock_tokenizer.__class__.__name__ = "EspeakWrapper"
        mock_tokenizer.tokenize.return_value = ["t", "ɛ", "s", "t"]

        from bot.tts.kokoro_direct import KokoroDirect

        with patch("kokoro_onnx.tokenizer.Tokenizer", return_value=mock_tokenizer):
            kokoro = KokoroDirect(
                str(self.model_path),
                str(self.voices_path),
            )

            # Patch audio creation to avoid onnx calls
            with patch(
                "bot.tts.kokoro_direct.KokoroDirect._create_audio",
                return_value=(np.ones(1000, dtype=np.float32), 24000),
            ):
                result = kokoro.create(
                    "This is a test of the English tokenizer.",
                    "default",
                    out_path=Path("/tmp/test.wav"),
                )

            assert kokoro.tokenizer is not None, "KokoroDirect tokenizer not initialized"
            assert kokoro.tokenizer.__class__.__name__ == "EspeakWrapper"
            assert isinstance(result, Path)

    @patch("bot.tts.kokoro_direct.KokoroDirect._load_model")
    @patch("bot.tts.kokoro_direct.KokoroDirect._load_voices")
    @patch("bot.tts.kokoro_direct.KokoroDirect._save_audio_to_wav")
    def test_no_grapheme_fallback_warning(self, mock_save, mock_load_voices, mock_load_model) -> None:
        """Test that no grapheme fallback warnings appear during TTS generation."""
        mock_save.return_value = None

        # Mock the tokenizer class
        mock_tokenizer = MagicMock()
        mock_tokenizer.__class__.__name__ = "EspeakWrapper"

        # Import KokoroDirect
        from bot.tts.kokoro_direct import KokoroDirect

        # Initialize KokoroDirect with a mocked tokenizer
        with patch("kokoro_onnx.tokenizer.Tokenizer", return_value=mock_tokenizer):
            kokoro = KokoroDirect(
                str(self.model_path),
                str(self.voices_path),
            )

            # Mock the tokenize method to return a valid phoneme sequence
            mock_tokenizer.tokenize.return_value = ["t", "ɛ", "s", "t"]

            # Set up logging capture without asserting a warning must exist
            records = []

            class _ListHandler(logging.Handler):
                def emit(self, record) -> None:
                    records.append(record)

            handler = _ListHandler(level=logging.WARNING)
            root_logger = logging.getLogger()
            old_level = root_logger.level
            root_logger.setLevel(logging.WARNING)
            root_logger.addHandler(handler)
            try:
                # Generate text
                test_text = "This is a test of the English tokenizer."
                with patch(
                    "bot.tts.kokoro_direct.KokoroDirect._create_audio",
                    return_value=(np.ones(1000, dtype=np.float32), 24000),
                ):
                    result = kokoro.create(test_text, "default", out_path=Path("/tmp/test.wav"))
            finally:
                root_logger.removeHandler(handler)
                root_logger.setLevel(old_level)

            # Check that no grapheme fallback warnings were logged
            for record in records:
                msg = record.getMessage().lower()
                assert "grapheme" not in msg, f"Found grapheme fallback warning: {record.getMessage()}"
                assert "fallback" not in msg, f"Found fallback warning: {record.getMessage()}"

            # Check that the result is a Path
            assert isinstance(result, Path), f"Result is not a Path, got {type(result)}"

    @patch("bot.tts.manager.TTSManager._init_tokenizer_registry")
    @patch("bot.tts.manager.TTSManager._load_kokoro")
    def test_tts_manager_integration(self, mock_load_kokoro, mock_init_registry) -> None:
        """Test TTSManager integration with KokoroDirect."""
        # Import TTSManager
        from bot.tts.manager import TTSManager

        # Create a minimal config
        config = {
            "TTS_BACKEND": "kokoro-onnx",
            "TTS_VOICE": "default",
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
        with patch("asyncio.run"):
            result_path = tts_manager.generate_speech(test_text)

            # Check that speech was generated
            assert result_path is not None, "TTSManager.generate_speech returned None"
            assert result_path == Path("/tmp/test.wav"), "Unexpected result path"

            # Verify that create was called with the right parameters
            mock_kokoro.create.assert_called_once()
            args, _kwargs = mock_kokoro.create.call_args
            assert args[0] == test_text, f"Expected text '{test_text}', got '{args[0]}'"


if __name__ == "__main__":
    unittest.main()
