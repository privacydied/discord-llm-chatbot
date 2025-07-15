"""
Wrapper for Kokoro-ONNX TTS to handle numpy.float32 compatibility issues.
"""

import logging
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional

class KokoroWrapper:
    """Wrapper for Kokoro-ONNX to handle numpy.float32 compatibility issues."""
    
    def __init__(self, model_path: str, voices_path: str):
        """
        Initialize KokoroWrapper.
        
        Args:
            model_path: Path to the Kokoro-ONNX model file
            voices_path: Path to the voices.json file
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self.kokoro = None
        self._voices_data = {}
        self._load_voices()
        self._init_kokoro()
    
    def _load_voices(self) -> None:
        """Load and preprocess voices from the voices.json file."""
        try:
            # Load voices file
            voices_path = Path(self.voices_path)
            if not voices_path.exists():
                logging.error(f"Voices file not found: {voices_path}")
                return
                
            with open(voices_path, 'r') as f:
                try:
                    voices_data = json.load(f)
                    if not isinstance(voices_data, dict):
                        logging.error(f"Voices data is not a dictionary: {type(voices_data)}")
                        return
                        
                    # Convert any voice embeddings to numpy arrays if they aren't already
                    for voice_id, embedding in voices_data.items():
                        if isinstance(embedding, list):
                            voices_data[voice_id] = np.array(embedding, dtype=np.float32)
                    
                    self._voices_data = voices_data
                    self.voices = list(voices_data.keys())
                    logging.info(f"Loaded {len(self.voices)} voices from {voices_path}")
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to decode voices file: {e}")
        except Exception as e:
            logging.error(f"Error loading voices: {e}")
    
    def _init_kokoro(self) -> None:
        """Initialize the Kokoro-ONNX engine."""
        try:
            from kokoro_onnx import Kokoro
            
            # Create a temporary voices.json file with the correct format
            temp_voices_path = Path("tts/temp_voices.json")
            
            # Prepare voices data for serialization
            serializable_voices = {}
            for voice_id, embedding in self._voices_data.items():
                if isinstance(embedding, np.ndarray):
                    serializable_voices[voice_id] = embedding.tolist()
                else:
                    serializable_voices[voice_id] = embedding
            
            # Save temporary voices file
            with open(temp_voices_path, 'w') as f:
                json.dump(serializable_voices, f)
            
            # Monkey-patch numpy.load to allow pickled files, as the library doesn't support it directly.
            original_load = np.load
            try:
                # Temporarily set allow_pickle to True for the Kokoro model loading
                np.load = lambda *a, **k: original_load(*a, **k, allow_pickle=True)
                
                # Initialize Kokoro with the temporary voices file
                self.kokoro = Kokoro(self.model_path, str(temp_voices_path))
                logging.info("Successfully initialized Kokoro-ONNX engine")
            finally:
                # Restore the original numpy.load to maintain security
                np.load = original_load
            # Add voices attribute to match Kokoro interface
            if self.kokoro and not hasattr(self.kokoro, 'voices'):
                setattr(self.kokoro, 'voices', self.voices)
                
        except ImportError as e:
            logging.error(f"Failed to import Kokoro-ONNX: {e}")
        except Exception as e:
            logging.error(f"Failed to initialize Kokoro-ONNX: {e}", exc_info=True)
    
    def create(self, text: str, voice_id: str, phonemes: Optional[str] = None, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Create audio from text and voice ID.
        
        Args:
            text: Text to synthesize
            voice_id: ID of voice to use
            phonemes: Optional phonemes to use (if None, text will be used)
            speed: Speed factor (1.0 is normal speed)
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        if not self.kokoro:
            raise RuntimeError("Kokoro-ONNX engine not initialized")
            
        if voice_id not in self._voices_data:
            available_voices = list(self._voices_data.keys())
            if len(available_voices) == 0:
                raise ValueError("No voices available")
            
            # Fall back to the first available voice
            voice_id = available_voices[0]
            logging.warning(f"Voice '{voice_id}' not found, falling back to '{voice_id}'")
        
        # Get voice embedding
        voice_embedding = self._voices_data[voice_id]
        
        # Pass voice embedding directly to create
        try:
            return self.kokoro.create(text, voice_embedding, phonemes=phonemes, speed=speed)
        except Exception as e:
            logging.error(f"Error in Kokoro create: {e}", exc_info=True)
            
            # Try an alternative approach if the standard one fails
            try:
                if phonemes is None:
                    # If no phonemes provided, let's try with text directly
                    return self.kokoro._create_audio(text, voice_embedding, speed)
                else:
                    return self.kokoro._create_audio(phonemes, voice_embedding, speed)
            except Exception as e2:
                logging.error(f"Alternative approach also failed: {e2}", exc_info=True)
                raise RuntimeError(f"Failed to generate speech: {e2}")
