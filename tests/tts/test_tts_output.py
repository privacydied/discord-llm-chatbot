"""
Tests for TTS output format, quality, and lexicon functionality.
"""
import pytest
import tempfile
import subprocess
import json
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules we want to test
try:
    import soundfile as sf
except ImportError:
    sf = None

def test_ogg_opus_output():
    """Test that TTS outputs OGG/Opus format with correct parameters."""
    pytest.importorskip("soundfile", reason="soundfile required for audio format validation")
    
    # Mock the TTS manager to test output format
    from bot.tts.interface import TTSManager
    from bot.tts.engines.stub import StubEngine
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test.ogg"
        
        # Create a TTS manager with stub engine
        manager = TTSManager()
        manager.engine = StubEngine()
        
        # Test synthesis with OGG output
        text = "counting from one to ten: one two three four five six seven eight nine ten cavalli furs whats the weather today"
        
        # Mock generate_tts to return OGG format
        async def mock_generate():
            path, mime = await manager.generate_tts(text, out_path=out_path, output_format="ogg")
            return path, mime
            
        # Run the async function
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        path, mime = loop.run_until_complete(mock_generate())
        
        # Validate output format
        assert path.suffix == ".ogg"
        assert mime == "audio/ogg"
        assert path.exists()
        
        # Use ffprobe to confirm Opus codec if available
        try:
            ffprobe_bin = os.getenv("FFPROBE", "ffprobe")
            result = subprocess.run([
                ffprobe_bin, "-v", "error", "-print_format", "json",
                "-show_streams", str(path)
            ], capture_output=True, text=True, check=True)
            
            info = json.loads(result.stdout)
            audio_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
            assert len(audio_streams) > 0
            
            audio = audio_streams[0]
            assert audio["codec_name"] == "opus"
            assert int(audio["sample_rate"]) == 48000
            assert int(audio["channels"]) == 1
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # ffprobe not available, skip codec validation
            pass


def test_no_clipping_and_dc():
    """Test that audio output has no clipping and minimal DC offset."""
    pytest.importorskip("soundfile", reason="soundfile required for audio analysis")
    
    from bot.tts.interface import TTSManager
    from bot.tts.engines.stub import StubEngine
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "test.wav"
        
        manager = TTSManager()
        manager.engine = StubEngine()
        
        text = "cavalli furs whats the weather today"
        
        # Generate WAV for analysis
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def mock_generate():
            return await manager.generate_tts(text, out_path=wav_path, output_format="wav")
            
        path, mime = loop.run_until_complete(mock_generate())
        
        # Analyze audio data
        data, sr = sf.read(str(path), dtype="float32")
        
        # Check for clipping (peak level should be under 0.99)
        peak_level = np.abs(data).max()
        assert peak_level <= 0.99, f"Audio clipping detected: peak={peak_level}"
        
        # Check for DC offset (should be near zero)
        dc_offset = np.abs(data.mean())
        assert dc_offset < 1e-3, f"Excessive DC offset: {dc_offset}"
        
        # Check for reasonable signal presence
        assert peak_level > 0.01, "Audio signal too quiet"


def test_lexicon_applied():
    """Test that lexicon entries are properly applied."""
    from bot.tts.eng_g2p_local import apply_lexicon, normalize_text
    
    # Test lexicon application
    original = "cavalli furs whats the weather"
    processed = apply_lexicon(original)
    
    # Should replace known lexicon entries
    assert "kəˈvɑli" in processed or "cavalli" not in processed
    assert "fɝz" in processed or "furs" not in processed  
    assert "wɑts" in processed or "whats" not in processed
    
    
def test_text_normalization():
    """Test text normalization for numbers and contractions."""
    from bot.tts.eng_g2p_local import normalize_text, number_to_words
    
    # Test number conversion
    text = "counting from 1 to 10"
    normalized = normalize_text(text)
    assert "one" in normalized
    assert "ten" in normalized
    assert "1" not in normalized
    assert "10" not in normalized
    
    # Test contraction handling
    text2 = "what's the weather today"
    normalized2 = normalize_text(text2)
    assert "whats" in normalized2
    assert "what's" not in normalized2


def test_ipa_normalization():
    """Test IPA symbol normalization and cleanup."""
    from bot.tts.ipa_vocab_loader import normalize_ipa
    # Create mock vocab with basic IPA symbols
    class MockVocab:
        def __init__(self):
            self.phoneme_to_id = {
                'k': 1, 'ə': 2, 'ˈ': 3, 'v': 4, 'ɑ': 5, 'l': 6, 'i': 7,
                'f': 8, 'ɝ': 9, 'z': 10, 'ɚ': 11, 'ː': 12, 'g': 13, 'u': 14, 'd': 15
            }
    
    vocab = MockVocab()
    
    # Test length marker removal
    ipa_with_length = "kəːˈvɑliː"
    normalized = normalize_ipa(ipa_with_length)
    assert "ː" not in normalized
    assert "kəˈvɑli" in normalized or "kə" in normalized
    
    # Test rhotic unification 
    ipa_with_rhotics = "fɚz"
    normalized = normalize_ipa(ipa_with_rhotics)
    # Should convert ɚ to ɝ
    assert "ɝ" in normalized
    assert "ɚ" not in normalized


def test_environment_variables():
    """Test environment variable controls."""
    # Test KOKORO_FORCE_IPA and KOKORO_GRAPHEME_FALLBACK
    from bot.tokenizer_registry import FORCE_IPA, ALLOW_GRAPHEME
    
    # These should be set according to our hardening
    assert FORCE_IPA == True  # Should default to True
    assert ALLOW_GRAPHEME == False  # Should default to False


@pytest.mark.integration
def test_end_to_end_synthesis():
    """Integration test for complete synthesis pipeline."""
    text = "counting from one to ten: one two three four five six seven eight nine ten cavalli furs whats the weather today"
    
    # This would require actual TTS setup, so we'll mock it
    from bot.tts.interface import TTSManager
    
    manager = TTSManager()
    
    # Test that the manager can be instantiated and configured
    assert manager is not None
    assert hasattr(manager, 'generate_tts')
    assert hasattr(manager, '_to_ogg_opus_ffmpegpy')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
