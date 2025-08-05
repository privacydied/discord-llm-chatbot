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
from bot.tts.errors import TTSWriteError

class TestKokoroDirect:
    """Test the fixed KokoroDirect implementation."""
    
    def setup_method(self):
        # Load paths from environment variables
        self.model_path = os.getenv('TTS_MODEL_PATH', 'tts/onnx/kokoro-v1.0.onnx')
        self.voices_path = os.getenv('TTS_VOICES_PATH', 'tts/onnx/voices/voices-v1.0.bin')
        
        # Create test voice file
        self.test_voice_id = 'test_voice'
        self.test_voice_file = self.voices_path
        
        # Add required tokenizers to the registry
        from bot.tokenizer_registry import TokenizerRegistry
        registry = TokenizerRegistry.get_instance()
        registry._available_tokenizers.add('espeak')
        registry._available_tokenizers.add('misaki')
        
        # Initialize engine with correct paths
        self.engine = KokoroDirect(model_path=self.model_path, voices_path=self.voices_path)
        # Prevent the engine from reloading voices (which would overwrite our change)
        self.engine._load_voices = lambda: None
        # Add the test voice
        self.engine.voices.append(self.test_voice_id)
    
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
            test_audio = np.random.rand(24000).astype(np.float32)  # 1 second of random audio
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
            mock_npz.files = [self.test_voice_id]
            
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
        audio, sample_rate = self.engine._create_audio("This is a test.", voice_embedding)
        
        assert audio is not None
        assert len(audio) > 0
        assert sample_rate == 24000
    
    def test_create_with_voice_id(self, mock_soundfile):
        """Test create method with voice ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_output.wav"
            result_path = self.engine.create("This is a test.", self.test_voice_id, out_path=out_path)
            
            assert result_path == out_path
            mock_soundfile.assert_called_once()
    
    def test_create_with_embedding(self, mock_soundfile):
        """Test create method with voice embedding."""
        voice_embedding = np.random.rand(512, 256).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_output.wav"
            result_path = self.engine.create("This is a test.", voice_embedding, out_path=out_path)
            
            assert result_path == out_path
            mock_soundfile.assert_called_once()
    
    def test_create_auto_path(self, mock_soundfile):
        """Test create method with auto-generated output path."""
        result_path = self.engine.create("This is a test.", self.test_voice_id)
        
        assert result_path is not None
        assert result_path.suffix == ".wav"
        mock_soundfile.assert_called_once()
    
    def test_create_soundfile_error_scipy_fallback(self, mock_soundfile):
        """Test fallback to scipy when soundfile fails."""
        # Make soundfile.write raise an exception
        mock_soundfile.side_effect = Exception("Soundfile error")
        
        # Mock scipy.io.wavfile
        with patch('scipy.io.wavfile.write') as mock_scipy:
            result_path = self.engine.create("This is a test.", self.test_voice_id)
            
            assert result_path is not None
            assert result_path.suffix == ".wav"
            mock_soundfile.assert_called_once()
            mock_scipy.assert_called_once()
    
    def test_create_both_writers_fail(self, mock_soundfile):
        """Test TTSWriteError when both soundfile and scipy fail."""
        # Make soundfile.write raise an exception
        mock_soundfile.side_effect = Exception("Soundfile error")
        
        # Mock scipy.io.wavfile to also fail
        with patch('scipy.io.wavfile.write', side_effect=Exception("Scipy error")):
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
        
        # Test override with environment variable
        with patch.dict(os.environ, {"TTS_LANGUAGE": "en", "TTS_PHONEMISER": "custom"}):
            engine = KokoroDirect(self.model_path, self.voices_path)
            assert engine.phonemiser == "custom"
