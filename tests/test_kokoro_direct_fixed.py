"""
Tests for the fixed KokoroDirect implementation.
"""

import os
import logging
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the fixed implementation
from bot.tts.kokoro_direct_fixed import KokoroDirect
from bot.tts.errors import TTSWriteError


class TestKokoroDirect:
    """Test the fixed KokoroDirect implementation."""

    def setup_method(self):
        # Load paths from environment variables
        self.model_path = os.getenv("TTS_MODEL_PATH", "tts/onnx/kokoro-v1.0.onnx")
        self.voices_path = os.getenv(
            "TTS_VOICES_PATH", "tts/onnx/voices/voices-v1.0.bin"
        )

        # Create test voice file
        self.test_voice_id = "test_voice"
        self.test_voice_file = self.voices_path

        # Add required tokenizers to the registry
        from bot.tokenizer_registry import TokenizerRegistry

        registry = TokenizerRegistry.get_instance()
        registry._available_tokenizers.add("espeak")
        registry._available_tokenizers.add("misaki")

        # Initialize engine with correct paths
        self.engine = KokoroDirect(
            model_path=self.model_path, voices_path=self.voices_path
        )
        # Prevent the engine from reloading voices (which would overwrite our change)
        self.engine._load_voices = lambda: None
        # Clear any existing voices data to force lazy loading in tests
        self.engine._voices_data = {}
        self.engine.voices = []
        # Add the test voice data
        test_voice = np.random.rand(512, 256).astype(np.float32)
        self.engine._voices_data[self.test_voice_id] = test_voice
        self.engine.voices.append(self.test_voice_id)

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock the kokoro_onnx tokenizer."""
        with patch("kokoro_onnx.tokenizer.Tokenizer") as mock:
            tokenizer_instance = MagicMock()
            tokenizer_instance.tokenize.return_value = np.array(
                [1, 2, 3, 4, 5], dtype=np.int64
            )
            mock.return_value = tokenizer_instance
            yield mock

    @pytest.fixture
    def mock_ort(self):
        """Mock onnxruntime."""
        with (
            patch("onnxruntime.InferenceSession") as mock_session,
            patch("onnxruntime.get_available_providers") as mock_providers,
        ):
            # Mock session
            session_instance = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input_ids"
            session_instance.get_inputs.return_value = [mock_input]

            # Mock inference output
            test_audio = np.random.rand(24000).astype(
                np.float32
            )  # 1 second of random audio
            session_instance.run.return_value = [test_audio]

            mock_session.return_value = session_instance
            mock_providers.return_value = ["CPUExecutionProvider"]

            yield mock_session

    @pytest.fixture
    def mock_np_load(self):
        """Mock numpy.load for voice loading."""
        with patch("numpy.load") as mock:
            # Create mock NPZ data
            mock_npz = MagicMock()
            mock_npz.files = [self.test_voice_id]

            # Create test voice embedding
            test_voice = np.random.rand(512, 256).astype(np.float32)
            mock_npz.__getitem__.return_value = test_voice

            mock.return_value = mock_npz
            yield mock

    @pytest.fixture
    def mock_soundfile(self):
        """Mock soundfile for WAV writing."""
        with patch("soundfile.write") as mock:
            yield mock

    def test_initialization(self, mock_tokenizer, mock_ort, mock_np_load):
        """Test KokoroDirect initialization."""
        assert self.engine.tokenizer is not None
        assert self.engine.sess is not None
        assert self.test_voice_id in self.engine.voices
        assert self.test_voice_id in self.engine.voices_data

    def test_get_voice_names(self):
        """Test get_voice_names method."""
        voices = self.engine.get_voice_names()
        assert self.test_voice_id in voices

    def test_create_audio(self):
        """Test _create_audio method."""
        voice_embedding = np.random.rand(512, 256).astype(np.float32)
        audio, sample_rate = self.engine._create_audio(
            "This is a test.", voice_embedding
        )

        assert audio is not None
        assert len(audio) > 0
        assert sample_rate == 24000

    def test_create_with_voice_id(self, mock_soundfile):
        """Test create method with voice ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_output.wav"
            result_path = self.engine.create(
                "This is a test.", self.test_voice_id, out_path=out_path
            )

            assert result_path == out_path
            mock_soundfile.assert_called_once()

    def test_create_with_embedding(self, mock_soundfile):
        """Test create method with voice embedding."""
        voice_embedding = np.random.rand(512, 256).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_output.wav"
            result_path = self.engine.create(
                "This is a test.", voice_embedding, out_path=out_path
            )

            assert result_path == out_path
            mock_soundfile.assert_called_once()

    def test_create_auto_path(self, mock_soundfile):
        """Test create method with auto-generated output path."""
        result_path = self.engine.create("This is a test.", self.test_voice_id)

        assert result_path is not None
        assert result_path.suffix == ".wav"
        mock_soundfile.assert_called_once()

    def test_create_soundfile_error_raises(self, mock_soundfile):
        """soundfile failure should raise TTSWriteError (no scipy fallback)."""
        # Make soundfile.write raise an exception
        mock_soundfile.side_effect = Exception("Soundfile error")

        with pytest.raises(TTSWriteError):
            self.engine.create("This is a test.", self.test_voice_id)

    def test_create_soundfile_error_raises_ttswriteerror(self, mock_soundfile):
        """Ensure no fallback and raise on writer error."""
        # Make soundfile.write raise an exception
        mock_soundfile.side_effect = Exception("Soundfile error")

        with pytest.raises(TTSWriteError):
            self.engine.create("This is a test.", self.test_voice_id)

    def test_create_empty_audio(self, mock_ort):
        """Test handling of empty audio."""
        # Make the model return empty audio
        mock_ort.return_value.run.return_value = [np.array([])]

        with pytest.raises(TTSWriteError):
            self.engine.create("This is a test.", self.test_voice_id)

    def test_phonemiser_selection(self):
        """Test phonemiser selection based on language."""
        # Test English
        with patch.dict(os.environ, {"TTS_LANGUAGE": "en"}):
            engine = KokoroDirect(self.model_path, self.voices_path)
            assert engine.phonemiser == "espeak"

        # Test Japanese
        with patch.dict(os.environ, {"TTS_LANGUAGE": "ja"}):
            engine = KokoroDirect(self.model_path, self.voices_path)
            assert engine.phonemiser == "misaki"

        # Test Chinese
        with patch.dict(os.environ, {"TTS_LANGUAGE": "zh"}):
            engine = KokoroDirect(self.model_path, self.voices_path)
            assert engine.phonemiser == "misaki"

    def test_kd_quiet_path_no_tokenizer_noise(self, caplog):
        """Test that the quiet path (disable_autodiscovery=True) produces no tokenizer-related noise."""
        caplog.set_level("DEBUG")

        # Create KD instance - this should not log tokenizer discovery during init
        kd = KokoroDirect(
            model_path="tts/kokoro-v1.0.onnx", voices_path="tts/voices-v1.0.bin"
        )

        # Call create with disable_autodiscovery=True
        kd.create(
            text="hello",
            voice="af_heart",
            disable_autodiscovery=True,
            logger=logging.getLogger("test"),
        )

        logs = "\n".join(r.message for r in caplog.records)

        # Assert no tokenizer-related noise in logs
        assert "TTS_TOKENISER" not in logs
        assert "No known tokenization methods found" not in logs
        assert "Found phonemizer package" not in logs
        assert "Found misaki package" not in logs
        assert "Found espeak" not in logs
        assert "tokenizer.method" not in logs
        assert "tokenizer.external" not in logs

        # But should still have the expected logs
        assert "Using pre-tokenized tokens" in logs
        assert "Created" in logs and "audio" in logs
        assert "Saved audio" in logs

    def test_tts_audio_quality_fixes(self, caplog):
        """Test TTS audio quality fixes: IPA routing, WAV normalization, quiet path."""
        caplog.set_level("DEBUG")

        # Test plain text (should use quiet grapheme path)
        kd = KokoroDirect(
            model_path="tts/kokoro-v1.0.onnx", voices_path="tts/voices-v1.0.bin"
        )

        # Clear any logs from initialization
        caplog.clear()

        out_path = kd.create(
            text="hello world",
            voice="af_heart",
            disable_autodiscovery=True,
            logger=logging.getLogger("test"),
        )

        logs = "\n".join(r.message for r in caplog.records)

        # Should use quiet grapheme path without tokenizer noise
        assert "Using pre-tokenized tokens" in logs
        assert (
            "Detected IPA phonemes" not in logs
        )  # Should not detect IPA in plain text
        assert "No known tokenization methods found" not in logs
        assert "Found phonemizer package" not in logs
        assert "Found misaki package" not in logs
        assert "Found espeak" not in logs

        # Verify output is a valid path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Test IPA input (should route to phoneme path)
        caplog.clear()

        out_path_ipa = kd.create(
            phonemes="həˈloʊ wɝːld",
            voice="af_heart",
            disable_autodiscovery=True,
            logger=logging.getLogger("test"),
        )

        logs_ipa = "\n".join(r.message for r in caplog.records)

        # Should use phoneme path without autodiscovery
        assert "Using pre-tokenized tokens" in logs_ipa
        assert out_path_ipa.exists()
        assert out_path_ipa.stat().st_size > 0

        # Test WAV format verification
        with open(out_path, "rb") as f:
            wav_data = f.read()

        # Should be valid WAV data (starts with RIFF)
        assert wav_data.startswith(b"RIFF")
        assert b"WAVE" in wav_data

        # Clean up
        out_path.unlink(missing_ok=True)
        out_path_ipa.unlink(missing_ok=True)
