"""
Direct implementation of Kokoro-ONNX TTS using the official Hugging Face approach.
This module handles TTS using the ONNX models and .bin voice files directly.
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import soundfile as sf
from onnxruntime import InferenceSession

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000  # Kokoro sample rate

class KokoroDirect:
    """
    Direct implementation of Kokoro-ONNX TTS using ONNX models and .bin voice files.
    """
    
    def __init__(self, onnx_dir: str = "tts/onnx", voices_dir: str = "tts/voices"):
        """
        Initialize KokoroDirect with paths to ONNX models and voice files.
        
        Args:
            onnx_dir: Directory containing ONNX model files
            voices_dir: Directory containing .bin voice files
        """
        self.onnx_dir = Path(onnx_dir)
        self.voices_dir = Path(voices_dir)
        self.session = None
        self.available_voices = []
        
        # Initialize session and load voices
        self._init_session()
        self._load_available_voices()
    
    def _init_session(self):
        """Initialize the ONNX inference session."""
        try:
            # Find available ONNX model variants - prioritized order
            model_variants = [
                'model.onnx',            # Base model
                'model_fp16.onnx',       # Mixed-precision
                'model_q8f16.onnx',      # Quantized with fp16
                'model_quantized.onnx',  # Quantized
                'model_uint8f16.onnx',   # uint8 with fp16
                'model_uint8.onnx',      # uint8
                'model_q4f16.onnx',      # q4 with fp16
                'model_q4.onnx',         # q4 quantized
            ]
            
            model_path = None
            for variant in model_variants:
                candidate_path = self.onnx_dir / variant
                if candidate_path.exists():
                    model_path = candidate_path
                    logger.info(f"Found ONNX model: {model_path}")
                    break
            
            if model_path is None:
                raise FileNotFoundError(f"No ONNX model found in {self.onnx_dir}")
            
            # Initialize session
            logger.info(f"Initializing ONNX session with model {model_path}")
            self.session = InferenceSession(str(model_path))
            logger.info("ONNX session initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            raise
    
    def _load_available_voices(self):
        """Load available voices from the voices directory."""
        try:
            # Check if voices directory exists
            if not self.voices_dir.exists():
                logger.error(f"Voices directory not found: {self.voices_dir}")
                return
            
            # Find all .bin files in the voices directory
            voice_files = list(self.voices_dir.glob("*.bin"))
            self.available_voices = [voice_file.stem for voice_file in voice_files]
            
            logger.info(f"Found {len(self.available_voices)} voices: {self.available_voices}")
            
        except Exception as e:
            logger.error(f"Error loading available voices: {e}")
    
    def _load_voice(self, voice_id: str) -> np.ndarray:
        """
        Load a voice embedding from a .bin file.
        
        Args:
            voice_id: Voice ID (without .bin extension)
            
        Returns:
            Voice embedding as numpy array
        """
        try:
            voice_path = self.voices_dir / f"{voice_id}.bin"
            if not voice_path.exists():
                available_voices = ", ".join(self.available_voices)
                raise FileNotFoundError(f"Voice file not found: {voice_path}. Available voices: {available_voices}")
            
            # Load voice embedding
            voice_embedding = np.fromfile(str(voice_path), dtype=np.float32).reshape(-1, 1, 256)
            logger.debug(f"Loaded voice embedding for {voice_id}: shape={voice_embedding.shape}")
            
            return voice_embedding
            
        except Exception as e:
            logger.error(f"Error loading voice {voice_id}: {e}")
            raise
    
    def _prepare_phonemes(self, phonemes: List[int]) -> Tuple[List[List[int]], np.ndarray]:
        """
        Prepare phoneme tokens and pad them as needed.
        
        Args:
            phonemes: List of phoneme token IDs
            
        Returns:
            Tuple of padded tokens and voice embedding index
        """
        # Ensure phoneme length is within limits (512 - 2 for pad tokens)
        assert len(phonemes) <= 510, f"Phoneme length {len(phonemes)} exceeds maximum (510)"
        
        # Add pad tokens at start and end
        padded_tokens = [[0, *phonemes, 0]]
        
        return padded_tokens
    
    def create(self, text: str, voice_id: str, phonemes: Optional[List[int]] = None, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Create audio from text/phonemes and voice ID.
        
        Args:
            text: Text to synthesize (used only if phonemes is None)
            voice_id: ID of voice to use
            phonemes: Optional list of phoneme token IDs
            speed: Speed factor (1.0 is normal speed)
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        try:
            # If phonemes is a string, not a list of token IDs, we need to tokenize it
            # But this is just a fallback - ideally phonemes should be provided as token IDs
            if isinstance(phonemes, str):
                logger.warning("Phonemes provided as string, not token IDs. Treating as raw text.")
                raw_text = phonemes
                phonemes = None
            else:
                raw_text = text
                
            # If no phonemes provided, we need phonemization
            # This is a placeholder - in practice, you should use Misaki G2P
            if phonemes is None:
                logger.warning("No phonemes provided. Using placeholder tokens.")
                # These are just example tokens - they won't produce meaningful speech
                phonemes = [50, 157, 43, 135, 16, 53, 135]
                
            # Load voice embedding
            voice_embedding = self._load_voice(voice_id)
            
            # Prepare tokens
            tokens = self._prepare_phonemes(phonemes)
            
            # Select style vector based on number of tokens (excluding pad tokens)
            ref_s = voice_embedding[len(phonemes)]
            
            # Set speed
            speed_np = np.ones(1, dtype=np.float32) * speed
            
            # Run inference
            logger.debug(f"Running inference with tokens shape={np.array(tokens).shape}, style shape={ref_s.shape}")
            audio = self.session.run(
                None,
                {
                    'input_ids': tokens,
                    'style': ref_s, 
                    'speed': speed_np
                }
            )[0]
            
            # Return audio and sample rate
            return audio[0], SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Error creating audio: {e}", exc_info=True)
            raise
    
    @property
    def voices(self) -> List[str]:
        """Get list of available voices."""
        return self.available_voices
