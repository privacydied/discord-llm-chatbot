"""
Test script to verify the fixed TTS pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the fixed KokoroDirect class
from bot.kokoro_direct_fixed import KokoroDirect

def test_tts_pipeline():
    """Test the TTS pipeline with the fixed KokoroDirect class."""
    try:
        # Get model and voices paths from environment or use defaults
        model_path = os.environ.get('TTS_MODEL_PATH', 'tts/onnx/kokoro-v1.0.onnx')
        voices_path = os.environ.get('TTS_VOICES_PATH', 'tts/voices/voices-v1.0.bin')
        
        logger.info(f"Using model path: {model_path}")
        logger.info(f"Using voices path: {voices_path}")
        
        # Initialize KokoroDirect
        logger.info("Initializing KokoroDirect...")
        kokoro = KokoroDirect(model_path, voices_path)
        
        # Get available voices
        voices = kokoro.get_voice_names()
        logger.info(f"Available voices: {', '.join(voices[:5])}{'...' if len(voices) > 5 else ''}")
        
        # Select a voice
        voice = voices[0] if voices else "default"
        logger.info(f"Using voice: {voice}")
        
        # Generate speech
        text = "This is a test of the fixed TTS pipeline."
        logger.info(f"Generating speech for text: '{text}'")
        
        # Create output directory if it doesn't exist
        output_dir = Path("tests/output")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate speech and get path to audio file
        output_path = output_dir / "test_output.wav"
        result_path = kokoro.create(text, voice, out_path=output_path)
        
        # Verify the result
        if isinstance(result_path, Path) and result_path.exists() and result_path.stat().st_size > 0:
            logger.info(f"Success! Audio generated at: {result_path} ({result_path.stat().st_size} bytes)")
            return True
        else:
            logger.error(f"Failed: Result is not a valid Path or file is empty. Result: {result_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing TTS pipeline: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_tts_pipeline()
    sys.exit(0 if success else 1)
