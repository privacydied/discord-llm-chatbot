"""
Tests for the KokoroDirect TTS engine interface.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from onnxruntime import InferenceSession

from bot.tts.kokoro_direct import KokoroDirect, SAMPLE_RATE


@pytest.fixture
def mock_kokoro_direct():
    """Fixture to create a KokoroDirect instance with a mocked ONNX session."""
    # Patch the methods that perform I/O during initialization
    with (
        patch("bot.tts.kokoro_direct.en.G2P"),
        patch("bot.tts.kokoro_direct.KokoroDirect._init_session"),
        patch("bot.tts.kokoro_direct.KokoroDirect._load_available_voices"),
    ):
        kokoro = KokoroDirect(onnx_dir="dummy/onnx", voices_dir="dummy/voices")

        # Manually set the session and g2p attributes that the constructor would have created
        kokoro.session = MagicMock(spec=InferenceSession)
        kokoro.g2p = MagicMock()

        # Mock the session's run method to return a dummy audio array
        mock_onnx_output = np.random.rand(1, 1024).astype(np.float32)
        kokoro.session.run.return_value = [mock_onnx_output]

        yield kokoro


def test_create_audio_with_g2p(mock_kokoro_direct):
    """Test that audio can be created from text using the misaki G2P phonemizer."""
    # Mock the internal voice loading to avoid file I/O
    # The style vector shape is (N, 256), so we mock a plausible embedding
    mock_voice_embedding = np.random.rand(512, 256).astype(np.float32)
    mock_kokoro_direct._load_voice = MagicMock(return_value=mock_voice_embedding)

    text_to_synthesize = "Hello, world."
    voice_id = "test_voice"

    # Call the create method, which should now use misaki
    audio, sample_rate = mock_kokoro_direct.create(
        text=text_to_synthesize, voice_id=voice_id
    )

    # Assertions
    assert isinstance(audio, np.ndarray), "Audio output should be a numpy array"
    assert audio.ndim == 1, "Audio should be a 1D array of samples"
    assert sample_rate == SAMPLE_RATE, f"Sample rate should be {SAMPLE_RATE}"
    mock_kokoro_direct._load_voice.assert_called_once_with(voice_id)

    # Check that the ONNX session was called
    mock_kokoro_direct.session.run.assert_called_once()
