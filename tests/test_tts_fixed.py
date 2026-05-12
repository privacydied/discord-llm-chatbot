"""
Test script to verify the fixed TTS pipeline.
Marked as integration — requires TTS binaries (espeak-ng, Kokoro ONNX).
"""

import os
import sys
import importlib
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("TTS_MODEL_PATH"),
    reason="TTS_MODEL_PATH not set; skipping TTS integration test",
)
def test_tts_pipeline():
    """Test the TTS pipeline with the fixed KokoroDirect class."""
    KokoroDirect = getattr(
        importlib.import_module("bot.tts.kokoro_direct_fixed"),
        "KokoroDirect",
    )

    model_path = os.environ.get("TTS_MODEL_PATH", "tts/onnx/kokoro-v1.0.onnx")
    voices_path = os.environ.get("TTS_VOICES_PATH", "tts/voices/voices-v1.0.bin")

    kokoro = KokoroDirect(model_path, voices_path)
    voices = kokoro.get_voice_names()
    assert voices, "Expected at least one voice to be available"

    voice = voices[0]
    text = "This is a test of the fixed TTS pipeline."

    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "test_output.wav"
    result_path = kokoro.create(text, voice, out_path=output_path)

    assert isinstance(result_path, Path)
    assert result_path.exists()
    assert result_path.stat().st_size > 0
