"""
Tests for the fixed KokoroDirect implementation.
"""

import os
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the fixed implementation
from bot.kokoro_direct_fixed import KokoroDirect
from bot.tts_errors import TTSWriteError

# Test constants
SAMPLE_RATE = 24000
TEST_TEXT = "This is a test."
TEST_VOICE_ID = "test_voice"

class TestKokoroDirect:
    """Test the fixed KokoroDirect implementation."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock the kokoro_onnx tokenizer."""
        with patch('kokoro_onnx.tokenizer.Tokenizer') as mock:
            tokenizer_instance = MagicMock()
            tokenizer_instance.tokenize.return_value = np.array([1, 2, 3, 4, 5], dtype=np.int64)
            mock.return_value = tokenizer_instance
            yield mock
    
    @pytest.fixture
    def mock_ort(self):
        """Mock onnxruntime."""
        with patch('onnxruntime.InferenceSession') as mock_session, \
             patch('onnxruntime.get_available_providers') as mock_providers:
            
            # Mock session
            session_instance = MagicMock()
            mock_input = MagicMock()
            mock_input.name = 'input_ids'
            session_instance.get_inputs.return_value = [mock_input]
            
            # Mock inference output
            test_audio = np.random.rand(SAMPLE_RATE).astype(np.float32)  # 1 second of random audio
            session_instance.run.return_value = [test_audio]
            
            mock_session.return_value = session_instance
            mock_providers.return_value = ['CPUExecutionProvider']
            
            yield mock_session
    
    @pytest.fixture
    def mock_np_load(self):
        """Mock numpy.load for voice loading."""
        with patch('numpy.load') as mock:
            # Create mock NPZ data
            mock_npz = MagicMock()
            mock_npz.files = [TEST_VOICE_ID]
            
            # Create test voice embedding
            test_voice = np.random.rand(512, 256).astype(np.float32)
            mock_npz.__getitem__.return_value = test_voice
            
            mock.return_value = mock_npz
            yield mock
    
    @pytest.fixture
    def mock_soundfile(self):
        """Mock soundfile for WAV writing."""
        with patch('soundfile.write') as mock:
            yield mock
    
    @pytest.fixture
    def kokoro_direct(self, mock_tokenizer, mock_ort, mock_np_load):
        """Create a KokoroDirect instance with mocked dependencies."""
        model_path = "models/test_model.onnx"
        voices_path = "models/test_voices.npz"
        
        kokoro = KokoroDirect(model_path, voices_path)
        return kokoro
    
    def test_initialization(self, kokoro_direct, mock_tokenizer, mock_ort, mock_np_load):
        """Test KokoroDirect initialization."""
        assert kokoro_direct.tokenizer is not None
        assert kokoro_direct.sess is not None
        assert TEST_VOICE_ID in kokoro_direct.voices
        assert TEST_VOICE_ID in kokoro_direct.voices_data
    
    def test_get_voice_names(self, kokoro_direct):
        """Test get_voice_names method."""
        voices = kokoro_direct.get_voice_names()
        assert TEST_VOICE_ID in voices
    
    def test_create_audio(self, kokoro_direct):
        """Test _create_audio method."""
        voice_embedding = np.random.rand(512, 256).astype(np.float32)
        audio, sample_rate = kokoro_direct._create_audio(TEST_TEXT, voice_embedding)
        
        assert audio is not None
        assert len(audio) > 0
        assert sample_rate == SAMPLE_RATE
    
    def test_create_with_voice_id(self, kokoro_direct, mock_soundfile):
        """Test create method with voice ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_output.wav"
            result_path = kokoro_direct.create(TEST_TEXT, TEST_VOICE_ID, out_path=out_path)
            
            assert result_path == out_path
            mock_soundfile.assert_called_once()
    
    def test_create_with_embedding(self, kokoro_direct, mock_soundfile):
        """Test create method with voice embedding."""
        voice_embedding = np.random.rand(512, 256).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_output.wav"
            result_path = kokoro_direct.create(TEST_TEXT, voice_embedding, out_path=out_path)
            
            assert result_path == out_path
            mock_soundfile.assert_called_once()
    
    def test_create_auto_path(self, kokoro_direct, mock_soundfile):
        """Test create method with auto-generated output path."""
        result_path = kokoro_direct.create(TEST_TEXT, TEST_VOICE_ID)
        
        assert result_path is not None
        assert result_path.suffix == ".wav"
        mock_soundfile.assert_called_once()
    
    def test_create_soundfile_error_scipy_fallback(self, kokoro_direct, mock_soundfile):
        """Test fallback to scipy when soundfile fails."""
        # Make soundfile.write raise an exception
        mock_soundfile.side_effect = Exception("Soundfile error")
        
        # Mock scipy.io.wavfile
        with patch('scipy.io.wavfile.write') as mock_scipy:
            result_path = kokoro_direct.create(TEST_TEXT, TEST_VOICE_ID)
            
            assert result_path is not None
            assert result_path.suffix == ".wav"
            mock_soundfile.assert_called_once()
            mock_scipy.assert_called_once()
    
    def test_create_both_writers_fail(self, kokoro_direct, mock_soundfile):
        """Test TTSWriteError when both soundfile and scipy fail."""
        # Make soundfile.write raise an exception
        mock_soundfile.side_effect = Exception("Soundfile error")
        
        # Mock scipy.io.wavfile to also fail
        with patch('scipy.io.wavfile.write', side_effect=Exception("Scipy error")):
            with pytest.raises(TTSWriteError):
                kokoro_direct.create(TEST_TEXT, TEST_VOICE_ID)
    
    def test_create_empty_audio(self, kokoro_direct, mock_ort):
        """Test handling of empty audio."""
        # Make the model return empty audio
        mock_ort.return_value.run.return_value = [np.array([])]
        
        with pytest.raises(TTSWriteError):
            kokoro_direct.create(TEST_TEXT, TEST_VOICE_ID)
    
    def test_phonemiser_selection(self):
        """Test phonemiser selection based on language."""
        # Test English
        with patch.dict(os.environ, {"TTS_LANGUAGE": "en"}):
            kokoro = KokoroDirect("model.onnx", "voices.npz")
            assert kokoro.phonemiser == "espeak"
        
        # Test Japanese
        with patch.dict(os.environ, {"TTS_LANGUAGE": "ja"}):
            kokoro = KokoroDirect("model.onnx", "voices.npz")
            assert kokoro.phonemiser == "misaki"
        
        # Test Chinese
        with patch.dict(os.environ, {"TTS_LANGUAGE": "zh"}):
            kokoro = KokoroDirect("model.onnx", "voices.npz")
            assert kokoro.phonemiser == "misaki"
        
        # Test override with environment variable
        with patch.dict(os.environ, {"TTS_LANGUAGE": "en", "TTS_PHONEMISER": "custom"}):
            kokoro = KokoroDirect("model.onnx", "voices.npz")
            assert kokoro.phonemiser == "custom"
