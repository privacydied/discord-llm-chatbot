"""
Unit tests for the KokoroDirect voice loader.

Verifies that the patched `_load_voice` function correctly loads all 26 voices
from a zipped NPZ archive and that each embedding has the correct shape.
"""
import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

# Add the project root to the path to allow importing the bot module
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot.kokoro_direct import KokoroDirect

# Constants for the test
NUM_VOICES = 26
VOICE_EMBEDDING_SHAPE = (512, 256)
VOICE_IDS = [f"voice_{i}" for i in range(NUM_VOICES)]

@pytest.fixture(scope="module")
def voice_pack_fixture():
    """Creates a temporary but valid NPZ voice pack and a minimal ONNX model for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        onnx_dir = tmp_path / "onnx"
        voices_dir = tmp_path / "voices"
        onnx_dir.mkdir()
        voices_dir.mkdir()

        # Create a minimal, valid ONNX model file (model.onnx)
        model_path = onnx_dir / "model.onnx"
        # This is a base64 encoded minimal ONNX model that defines an identity operation.
        # It's not functional for TTS, but it is a valid model that can be loaded.
        onnx_model_b64 = "CAASgQIKaXJfdmVyc2lvbhAIIg4KBWlucHV0EgVpbnB1dCIOAAAAYiIKBm91dHB1dBIHb3V0cHV0Ig4AAABgaiYKBWlucHV0GgUKAQQoAWomCgZvdXRwdXQaBQoBBCgBcg4KBWlucHV0EgVpbnB1dDgBqgEKDElkZW50aXR5Tm9kZRIFSW5wdXQYASIGT3V0cHV0GAIgAkgCGghpZGVudGl0eRoFaW5wdXQiB291dHB1dCoGaW5wdXQqB291dHB1dFIAChBwcm9kdWNlcl92ZXJzaW9uEgYxLjE1LjFSAAoOZG9tYWluX3ZlcnNpb24SAjE4"
        with open(model_path, "wb") as f:
            import base64
            f.write(base64.b64decode(onnx_model_b64))

        # Create a dummy voice pack file
        voice_pack_file = voices_dir / "voices-v1.0.bin"
        voice_data = {}
        for voice_id in VOICE_IDS:
            voice_data[voice_id] = np.random.rand(*VOICE_EMBEDDING_SHAPE).astype(np.float32)
        np.savez_compressed(voice_pack_file, **voice_data)
        
        yield tmp_path


def test_kokoro_initialization(voice_pack_fixture, mocker):
    """Test that KokoroDirect initializes correctly and finds all available voices."""
    config = {
        "KOKORO_MODEL_PATH": str(voice_pack_fixture / "onnx"),
        "KOKORO_VOICE_PACK_PATH": str(voice_pack_fixture / "voices")
    }
    mocker.patch("bot.kokoro_direct.KokoroDirect._init_session", return_value=None)
    kokoro = KokoroDirect(config=config)
    assert kokoro is not None
    assert kokoro.session is None
    assert len(kokoro.available_voices) == NUM_VOICES

def test_load_all_voices_correctly(voice_pack_fixture, mocker):
    """Test that every voice in the pack can be loaded and has the correct shape."""
    config = {
        "KOKORO_MODEL_PATH": str(voice_pack_fixture / "onnx"),
        "KOKORO_VOICE_PACK_PATH": str(voice_pack_fixture / "voices")
    }
    mocker.patch("bot.kokoro_direct.KokoroDirect._init_session", return_value=None)
    kokoro = KokoroDirect(config=config)
    
    for voice_id in VOICE_IDS:
        embedding = kokoro._load_voice(voice_id)
        assert isinstance(embedding, np.ndarray), f"Voice {voice_id} did not load as a numpy array."
        assert embedding.shape[1:] == (1, 256), f"Voice {voice_id} has incorrect shape. Expected (*, 1, 256), got {embedding.shape}."

def test_load_nonexistent_voice_raises_error(voice_pack_fixture, mocker):
    """Test that attempting to load a voice that does not exist raises a ValueError."""
    config = {
        "KOKORO_MODEL_PATH": str(voice_pack_fixture / "onnx"),
        "KOKORO_VOICE_PACK_PATH": str(voice_pack_fixture / "voices")
    }
    mocker.patch("bot.kokoro_direct.KokoroDirect._init_session", return_value=None)
    kokoro = KokoroDirect(config=config)
    non_existent_voice_id = "voice_999"
    
    with pytest.raises(ValueError) as excinfo:
        kokoro._load_voice(non_existent_voice_id)
    
    assert non_existent_voice_id in str(excinfo.value)
