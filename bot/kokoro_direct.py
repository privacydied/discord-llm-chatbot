"""
Direct implementation of Kokoro-ONNX TTS using the official Hugging Face approach.
This module handles TTS using the ONNX models and .bin voice files directly.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from misaki import en
import numpy as np
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
    
    def __init__(self, config: dict = None, onnx_dir: str = "tts/onnx", voices_dir: str = "tts/voices"):
        """
        Initialize KokoroDirect with paths to ONNX models and voice files.
        
        Args:
            config: Optional configuration dictionary.
            onnx_dir: Directory containing ONNX model files (fallback).
            voices_dir: Directory containing .bin voice files (fallback).
        """
        from bot.config import load_config
        self.config = config if config is not None else load_config()
        logger.debug(f"KokoroDirect received config: {config}")
        logger.debug(f"KokoroDirect loaded config: {self.config}")
        
        onnx_dir = self.config.get("KOKORO_MODEL_PATH", onnx_dir)
        voices_dir = self.config.get("KOKORO_VOICE_PACK_PATH", voices_dir)
        
        logger.debug(f"Using ONNX dir: {onnx_dir}")
        logger.debug(f"Using voices dir: {voices_dir}")
        self.onnx_dir = Path(onnx_dir)
        self.voices_dir = Path(voices_dir)
        self.session = None
        self._voices_cache = {}
        self.available_voices = []

        # Initialize the G2P converter
        try:
            self.g2p = en.G2P()
            logger.info("Misaki G2P engine initialized for English.")
        except Exception as e:
            logger.error("Failed to initialize Misaki G2P engine.", exc_info=True)
            raise RuntimeError("Could not initialize G2P engine") from e
        
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
        logger.debug(f"Attempting to load voices from {self.voices_dir}")
        try:
            voice_pack_path = self.voices_dir / "voices-v1.0.bin"
            if not voice_pack_path.exists():
                logger.error(f"CRITICAL: Voice pack not found at {voice_pack_path}. No voices loaded.")
                self.available_voices = []
                return

            logger.debug(f"Found voice pack at {voice_pack_path}, loading...")
            with np.load(voice_pack_path, allow_pickle=True) as data:
                self._voices_cache = {name: data[name] for name in data.files}
                self.available_voices = list(self._voices_cache.keys())
                logger.debug(f"Loaded {len(self.available_voices)} voices into cache.")
            
            logger.info(f"Found {len(self.available_voices)} voices: {self.available_voices}")

        except Exception as e:
            logger.error(f"Error loading available voices: {e}", exc_info=True)
            self.available_voices = []
    
    def _load_voice(self, voice_id: str) -> np.ndarray:
        """
        Load a voice embedding from the voices-v1.0.bin NPZ archive.

        Args:
            voice_id: The ID of the desired voice.

        Returns:
            Voice embedding as a numpy array of shape (N, 1, 256).
        """
        try:
            if voice_id not in self._voices_cache:
                raise ValueError(f"Voice '{voice_id}' not found or no voices loaded.")

            voice_embedding = self._voices_cache[voice_id]

            # Handle both 2D (N, 256) and 3D (N, 1, 256) shapes.
            if voice_embedding.ndim == 3 and voice_embedding.shape[1:] == (1, 256):
                logger.debug(f"Loaded pre-shaped 3D voice '{voice_id}': shape={voice_embedding.shape}")
                return voice_embedding
            elif voice_embedding.ndim == 2 and voice_embedding.shape[1] == 256:
                reshaped_embedding = voice_embedding.reshape(-1, 1, 256)
                logger.debug(f"Reshaped and loaded 2D voice '{voice_id}': shape={reshaped_embedding.shape}")
                return reshaped_embedding
            else:
                raise ValueError(f"Unexpected voice embedding shape for '{voice_id}': {voice_embedding.shape}")

        except Exception as e:
            logger.error(f"Error loading voice '{voice_id}': {e}", exc_info=True)
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
                phonemes = None
            else:
                pass
                
            # If no phonemes provided, perform basic character-to-phoneme mapping.
            # This is a temporary fix. A proper G2P model should be used for production.
            if phonemes is None:
                logger.debug(f"No phonemes provided. Using Misaki G2P for: '{text}'")
                try:
                    # Use the initialized G2P object to convert text to phoneme IDs.
                    phonemes = self.g2p(text)
                    # Ensure we don't exceed the model's limit (512 tokens, with 2 for start/end).
                    phonemes = phonemes[:510]
                    logger.debug(f"Generated phoneme IDs: {phonemes}")
                except Exception as e:
                    logger.error(f"Misaki G2P failed for text: '{text}'. Error: {e}", exc_info=True)
                    raise ValueError(f"Failed to phonemize text: {text}") from e
                
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
    
    def is_available(self) -> bool:
        """Check if the TTS engine is initialized and ready."""
        return self.session is not None and len(self.available_voices) > 0

    @property
    def voices(self) -> List[str]:
        """Get list of available voices."""
        return self.available_voices
