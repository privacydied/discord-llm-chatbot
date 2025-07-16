"""
Wrapper for Kokoro-ONNX TTS to handle numpy.float32 compatibility issues.
Provides direct NPZ voice data handling and fixes ONNX input signature mismatches.
"""

import logging
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union, Any
from numpy.typing import NDArray

# Import our direct NPZ handler
from bot.kokoro_direct import KokoroDirect

class KokoroWrapper:
    """Wrapper for Kokoro-ONNX to handle numpy.float32 compatibility issues."""
    
    def __init__(self, model_path: str, voices_path: str):
        """
        Initialize KokoroWrapper.
        
        Args:
            model_path: Path to the Kokoro-ONNX model file
            voices_path: Path to the voices file (NPZ format)
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self.kokoro = None
        self._voices_data = {}
        self._voices_list = []
        self._load_voices()
        self._init_kokoro()
    
    def _load_voices(self) -> None:
        """Load and preprocess voices from the binary voices file (.bin or .npz)."""
        try:
            # Load voices file
            voices_path = Path(self.voices_path)
            if not voices_path.exists():
                logging.error(f"Voices file not found: {voices_path}", extra={'subsys': 'tts', 'event': 'load_voices.error.not_found'})
                return
                
            # Initialize voices_data as an empty dict to avoid reference errors
            voices_data = {}
            loaded_successfully = False
            
            # First try to load as NPZ regardless of extension
            # Many voice files might have .bin extension but actually be NPZ format
            try:
                logging.debug(f"Trying to load as NPZ file: {voices_path}", extra={'subsys': 'tts', 'event': 'load_voices.try_npz'})
                npz_data = np.load(voices_path, allow_pickle=True)
                
                # Check if it's an NpzFile object
                if hasattr(npz_data, 'files'):
                    logging.debug(f"File is NPZ format with keys: {npz_data.files}", extra={'subsys': 'tts', 'event': 'load_voices.npz.keys'})
                    
                    # Extract each voice embedding from the NPZ file
                    for voice_id in npz_data.files:
                        # Extract the actual voice embedding data
                        voice_data = npz_data[voice_id]
                        if isinstance(voice_data, np.ndarray):
                            voices_data[voice_id] = voice_data
                            logging.debug(f"Loaded voice {voice_id} with shape {voice_data.shape}", 
                                        extra={'subsys': 'tts', 'event': 'load_voices.npz.voice'})
                        else:
                            logging.warning(f"Voice data for {voice_id} is not a numpy array: {type(voice_data)}", 
                                           extra={'subsys': 'tts', 'event': 'load_voices.npz.warning'})
                    
                    if voices_data:
                        logging.debug(f"Successfully loaded NPZ file with {len(voices_data)} voices", 
                                    extra={'subsys': 'tts', 'event': 'load_voices.npz.success'})
                        loaded_successfully = True
                    else:
                        logging.warning("NPZ file contained no valid voice data", 
                                      extra={'subsys': 'tts', 'event': 'load_voices.npz.empty'})
                else:
                    logging.debug("File is not an NPZ file (no 'files' attribute)", 
                                extra={'subsys': 'tts', 'event': 'load_voices.not_npz'})
            except Exception as e:
                logging.debug(f"Not an NPZ file: {e}", extra={'subsys': 'tts', 'event': 'load_voices.not_npz_error'})
            
            # If NPZ loading failed, try as numpy array
            if not loaded_successfully:
                try:
                    logging.debug(f"Trying to load as numpy array: {voices_path}", extra={'subsys': 'tts', 'event': 'load_voices.try_numpy'})
                    loaded_data = np.load(voices_path, allow_pickle=True)
                    
                    # Check if it's a numpy array that can be converted to dict
                    if isinstance(loaded_data, np.ndarray) and hasattr(loaded_data, 'item') and callable(loaded_data.item):
                        try:
                            voices_dict = loaded_data.item()
                            if isinstance(voices_dict, dict):
                                voices_data = voices_dict
                                logging.debug(f"Successfully loaded numpy array as dictionary with {len(voices_data)} items", 
                                            extra={'subsys': 'tts', 'event': 'load_voices.numpy.success'})
                                loaded_successfully = True
                            else:
                                logging.warning(f"Numpy array item() is not a dictionary: {type(voices_dict)}", 
                                              extra={'subsys': 'tts', 'event': 'load_voices.numpy.not_dict'})
                        except Exception as e:
                            logging.warning(f"Could not convert numpy array to dict: {e}", 
                                          extra={'subsys': 'tts', 'event': 'load_voices.numpy.convert_error'})
                    else:
                        logging.debug(f"Not a convertible numpy array: {type(loaded_data)}", 
                                    extra={'subsys': 'tts', 'event': 'load_voices.not_numpy_array'})
                except Exception as e:
                    logging.debug(f"Not a numpy array: {e}", extra={'subsys': 'tts', 'event': 'load_voices.not_numpy_error'})
            
            # If all binary formats failed, try JSON as last resort
            if not loaded_successfully:
                try:
                    logging.debug(f"Trying to load as JSON: {voices_path}", extra={'subsys': 'tts', 'event': 'load_voices.try_json'})
                    with open(voices_path, 'r') as f:
                        loaded_data = json.load(f)
                        if isinstance(loaded_data, dict):
                            voices_data = loaded_data
                            logging.debug(f"Successfully loaded JSON with {len(voices_data)} items", 
                                        extra={'subsys': 'tts', 'event': 'load_voices.json.success'})
                            loaded_successfully = True
                        else:
                            logging.warning(f"JSON data is not a dictionary: {type(loaded_data)}", 
                                          extra={'subsys': 'tts', 'event': 'load_voices.json.not_dict'})
                except json.JSONDecodeError as e:
                    logging.debug(f"Not a JSON file: {e}", extra={'subsys': 'tts', 'event': 'load_voices.not_json_error'})
                except Exception as e:
                    logging.debug(f"Error loading as JSON: {e}", extra={'subsys': 'tts', 'event': 'load_voices.json_error'})
            
            # If we still haven't loaded anything successfully, give up
            if not loaded_successfully:
                logging.error(f"Could not load voices file in any supported format: {voices_path}", 
                             extra={'subsys': 'tts', 'event': 'load_voices.error.unsupported_format'})
                return
            
            # Convert any voice embeddings to numpy arrays if they aren't already
            for voice_id, embedding in voices_data.items():
                if isinstance(embedding, list):
                    voices_data[voice_id] = np.array(embedding, dtype=np.float32)
            
            # Set the voices data and list of available voices
            self._voices_data = voices_data
            self._voices_list = list(voices_data.keys())
            logging.info(f"Loaded {len(self._voices_list)} voices from {voices_path}", 
                        extra={'subsys': 'tts', 'event': 'load_voices.success'})
        except Exception as e:
            logging.error(f"Error loading voices: {e}", 
                         extra={'subsys': 'tts', 'event': 'load_voices.error'}, exc_info=True)
    
    @property
    def voices(self) -> List[str]:
        """Get list of available voice names."""
        if self.kokoro and hasattr(self.kokoro, 'get_voice_names'):
            return self.kokoro.get_voice_names()
        return self._voices_list

    def _init_kokoro(self) -> None:
        """Initialize the KokoroDirect engine for direct NPZ handling."""
        try:
            # Initialize KokoroDirect with the model and voices paths
            # This will load the NPZ file directly without JSON conversion
            self.kokoro = KokoroDirect(self.model_path, self.voices_path)
            
            # Log success with detailed information
            logging.info(f"Successfully initialized KokoroDirect with {len(self.voices)} voices", 
                        extra={'subsys': 'tts', 'event': 'init_kokoro.success'})
            
            # Log ONNX providers for debugging
            if hasattr(self.kokoro, 'sess') and hasattr(self.kokoro.sess, 'get_providers'):
                providers = self.kokoro.sess.get_providers()
                logging.debug(f"Using ONNX providers: {providers}", 
                             extra={'subsys': 'tts', 'event': 'init_kokoro.providers'})
                
        except ImportError as e:
            logging.error(f"Failed to import required modules: {e}", 
                         extra={'subsys': 'tts', 'event': 'init_kokoro.import_error'})
        except Exception as e:
            logging.error(f"Failed to initialize KokoroDirect: {e}", 
                         extra={'subsys': 'tts', 'event': 'init_kokoro.error'}, exc_info=True)
    
    def create(self, text: str, voice_id_or_embedding, phonemes: Optional[str] = None, speed: float = 1.0, *, out_path: Optional[Path] = None) -> Tuple[np.ndarray, int]:
        """
        Create audio from text and voice ID or embedding.
        
        Args:
            text: Text to synthesize
            voice_id_or_embedding: ID of voice to use or a voice embedding numpy array
            phonemes: Optional phonemes to use (if None, text will be used)
            speed: Speed factor (1.0 is normal speed)
            out_path: Optional output path for the WAV file
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        if not self.kokoro:
            raise RuntimeError("KokoroDirect engine not initialized")
        
        # Log the TTS request
        logging.debug(f"Creating TTS for text: '{text[:50]}...' with {'voice_id' if isinstance(voice_id_or_embedding, str) else 'embedding'}", 
                     extra={'subsys': 'tts', 'event': 'create.start'})
        
        # Our KokoroDirect implementation already handles both voice IDs and embeddings
        # It also already uses 'input_ids' instead of 'tokens' for ONNX input
        try:
            # Directly pass to KokoroDirect's create method which now returns a Path
            wav_path = self.kokoro.create(text, voice_id_or_embedding, phonemes=phonemes, speed=speed, out_path=out_path)
            
            # Log success
            logging.debug(f"Successfully created TTS audio file: {wav_path}", 
                         extra={'subsys': 'tts', 'event': 'create.success'})
            
            # Read the WAV file to get audio samples and sample rate
            try:
                import soundfile as sf
                audio, sample_rate = sf.read(wav_path)
                logging.debug(f"Read audio from {wav_path}: {len(audio)/sample_rate:.2f}s", 
                             extra={'subsys': 'tts', 'event': 'create.read_wav'})
            except ImportError:
                # Fall back to scipy if soundfile is not available
                from scipy.io import wavfile
                sample_rate, audio = wavfile.read(wav_path)
                # Convert int16 to float32 if needed
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32767.0
                logging.debug(f"Read audio from {wav_path} using scipy: {len(audio)/sample_rate:.2f}s", 
                             extra={'subsys': 'tts', 'event': 'create.read_wav_scipy'})
            
            return audio, sample_rate
            
        except Exception as e:
            logging.error(f"Error in KokoroDirect create: {e}", 
                         extra={'subsys': 'tts', 'event': 'create.error'}, exc_info=True)
            from .tts_errors import TTSWriteError
            if isinstance(e, TTSWriteError):
                raise
            raise RuntimeError(f"Failed to generate speech: {e}")
