"""
Direct NPZ integration for Kokoro-ONNX TTS engine.
Handles NPZ voice data directly without JSON conversion and fixes ONNX input signature mismatches.
"""

import os
import logging
import numpy as np
import time
import tempfile
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from numpy.typing import NDArray

# Import custom exceptions
from .tts_errors import TTSWriteError

# Setup logging
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
MAX_PHONEME_LENGTH = 512
CACHE_DIR = os.environ.get('XDG_CACHE_HOME', Path('tts/cache'))

class KokoroDirect:
    """
    Direct NPZ integration for Kokoro-ONNX TTS engine.
    Handles NPZ voice data directly without JSON conversion.
    Fixes ONNX input signature mismatches ('tokens' -> 'input_ids').
    """
    
    def __init__(self, model_path: str, voices_path: str):
        """
        Initialize KokoroDirect with model and voices paths.
        
        Args:
            model_path: Path to the ONNX model file
            voices_path: Path to the NPZ voices file
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self.voices_data = {}
        self.voices = []
        self.tokenizer = None
        self.sess = None
        self.language = os.environ.get("TTS_LANGUAGE", "en")
        self.phonemiser = self._select_phonemiser(self.language)
        
        # Load the model and voices
        self._load_model()
        self._load_voices()
        
        logger.info(f"Initialized KokoroDirect with {len(self.voices)} voices")
        
    def _select_phonemiser(self, language: str) -> str:
        """
        Select the appropriate phonemiser based on language.
        
        Args:
            language: Language code (e.g., 'en', 'ja')
            
        Returns:
            Phonemiser name to use
        """
        # Allow override from environment
        env_phonemiser = os.environ.get("TTS_PHONEMISER")
        if env_phonemiser:
            return env_phonemiser
            
        # Select based on language
        if language.startswith("en"):
            phonemiser = "espeak"
        elif language.startswith(("ja", "zh")):
            phonemiser = "misaki"
        else:
            phonemiser = "espeak"
            
        # Log warning if using non-espeak for English
        if language.startswith("en") and phonemiser != "espeak":
            logger.warning(
                f"Using non-recommended phonemiser '{phonemiser}' for English language", 
                extra={'subsys': 'tts', 'event': 'phonemiser.warning'}
            )
            
        logger.debug(
            f"Selected phonemiser '{phonemiser}' for language '{language}'", 
            extra={'subsys': 'tts', 'event': 'phonemiser.select'}
        )
        return phonemiser
        
    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        try:
            # Import here to avoid dependency issues
            from kokoro_onnx.tokenizer import Tokenizer
            import onnxruntime as ort
            
            # Create tokenizer
            self.tokenizer = Tokenizer()
            
            # Log ONNX providers
            providers = ort.get_available_providers()
            logger.debug(f"ONNX providers: {providers}", extra={'subsys': 'tts', 'event': 'onnx.providers'})
            
            # Create ONNX session
            self.sess = ort.InferenceSession(self.model_path, providers=providers)
            
            # Log input names for debugging
            input_names = [input.name for input in self.sess.get_inputs()]
            logger.debug(f"ONNX model input names: {input_names}", extra={'subsys': 'tts', 'event': 'onnx.inputs'})
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}", extra={'subsys': 'tts', 'event': 'load_model.error'})
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}", extra={'subsys': 'tts', 'event': 'load_model.error'}, exc_info=True)
            raise
    
    def _load_voices(self) -> None:
        """Load voice embeddings directly from NPZ file."""
        try:
            voices_path = Path(self.voices_path)
            if not voices_path.exists():
                logger.error(f"Voices file not found: {voices_path}", extra={'subsys': 'tts', 'event': 'load_voices.error'})
                return
            
            # Load NPZ file
            logger.debug(f"Loading NPZ voices from: {voices_path}", extra={'subsys': 'tts', 'event': 'load_voices.npz'})
            npz_data = np.load(voices_path, allow_pickle=True)
            
            if hasattr(npz_data, 'files'):
                logger.debug(f"NPZ file contains {len(npz_data.files)} voices", extra={'subsys': 'tts', 'event': 'load_voices.npz.count'})
                
                # Extract each voice embedding
                for voice_id in npz_data.files:
                    voice_data = npz_data[voice_id]
                    if isinstance(voice_data, np.ndarray):
                        # Handle different voice embedding shapes
                        if voice_data.shape[0] == MAX_PHONEME_LENGTH and (len(voice_data.shape) == 2 and voice_data.shape[1] == 256):
                            # Perfect shape (512, 256)
                            self.voices_data[voice_id] = voice_data
                            self.voices.append(voice_id)
                            logger.debug(f"Loaded voice {voice_id} with shape {voice_data.shape}", 
                                        extra={'subsys': 'tts', 'event': 'load_voices.voice'})
                        elif voice_data.shape[0] == MAX_PHONEME_LENGTH and len(voice_data.shape) == 3 and voice_data.shape[1] == 1 and voice_data.shape[2] == 256:
                            # Shape (512, 1, 256) - already correct format
                            self.voices_data[voice_id] = voice_data
                            self.voices.append(voice_id)
                            logger.debug(f"Loaded voice {voice_id} with shape {voice_data.shape}", 
                                        extra={'subsys': 'tts', 'event': 'load_voices.voice'})
                        elif (voice_data.shape[0] == 510 or voice_data.shape[0] == 511) and (
                              (len(voice_data.shape) == 2 and voice_data.shape[1] == 256) or
                              (len(voice_data.shape) == 3 and voice_data.shape[1] == 1 and voice_data.shape[2] == 256)):
                            # Shape (510, 256) or (510, 1, 256) - need padding to 512
                            pad_size = MAX_PHONEME_LENGTH - voice_data.shape[0]
                            if len(voice_data.shape) == 2:
                                # Pad (510, 256) to (512, 256)
                                padded_data = np.pad(voice_data, ((0, pad_size), (0, 0)), 'constant')
                                self.voices_data[voice_id] = padded_data
                            else:
                                # Pad (510, 1, 256) to (512, 1, 256)
                                padded_data = np.pad(voice_data, ((0, pad_size), (0, 0), (0, 0)), 'constant')
                                self.voices_data[voice_id] = padded_data
                                
                            self.voices.append(voice_id)
                            
                            # Log min/max values to check for 0-padding dominance
                            non_zero_data = voice_data[voice_data != 0]
                            if len(non_zero_data) > 0:
                                min_val = np.min(non_zero_data)
                                max_val = np.max(non_zero_data)
                                logger.debug(
                                    f"Padded voice {voice_id} from {voice_data.shape} to {self.voices_data[voice_id].shape}, "
                                    f"non-zero min/max: {min_val:.4f}/{max_val:.4f}", 
                                    extra={'subsys': 'tts', 'event': 'load_voices.padded'}
                                )
                            else:
                                logger.warning(
                                    f"Voice {voice_id} contains all zeros", 
                                    extra={'subsys': 'tts', 'event': 'load_voices.warning.zeros'}
                                )
                        else:
                            logger.warning(
                                f"Skipping voice {voice_id} with incompatible shape {voice_data.shape}", 
                                extra={'subsys': 'tts', 'event': 'load_voices.warning.shape'}
                            )
                    else:
                        logger.warning(
                            f"Skipping voice {voice_id} with non-array type {type(voice_data)}", 
                            extra={'subsys': 'tts', 'event': 'load_voices.warning.type'}
                        )
                
                logger.info(f"Loaded {len(self.voices)} voices from NPZ file", 
                          extra={'subsys': 'tts', 'event': 'load_voices.success'})
            else:
                logger.error("Invalid NPZ file format (no 'files' attribute)", 
                           extra={'subsys': 'tts', 'event': 'load_voices.error.format'})
        except Exception as e:
            logger.error(f"Failed to load voices: {e}", 
                       extra={'subsys': 'tts', 'event': 'load_voices.error'}, exc_info=True)
    
    def get_voice_names(self) -> List[str]:
        """Get list of available voice names."""
        return self.voices
    
    def _create_audio(self, phonemes: str, voice: NDArray[np.float32], speed: float = 1.0) -> Tuple[NDArray[np.float32], int]:
        """
        Create audio from phonemes and voice embedding.
        Fixes input signature mismatch by using 'input_ids' instead of 'tokens'.
        
        Args:
            phonemes: Phoneme string
            voice: Voice embedding array
            speed: Speed factor (1.0 is normal)
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        if self.sess is None:
            raise RuntimeError("ONNX session not initialized")
        
        try:
            # Start timing
            start_t = time.time()
            
            # Tokenize phonemes
            tokens = self.tokenizer.tokenize(phonemes)
            
            # Convert tokens list to numpy array if it's a list
            if isinstance(tokens, list):
                tokens = np.array(tokens, dtype=np.int64)
                logger.debug(
                    f"Converted tokens list to numpy array with shape {tokens.shape}",
                    extra={'subsys': 'tts', 'event': 'create_audio.tokens_conversion'}
                )
            
            # Prepare inputs for the model
            # Note: Kokoro-ONNX 0.4.9+ expects 'style' parameter as float32 with shape (1,1)
            inputs = {
                'input_ids': tokens,
                'speaker_embedding': voice,
                'speed': np.array([speed], dtype=np.float32),
                'style': np.array([[0.0]], dtype=np.float32)  # Default neutral style (0) as float32 with shape (1,1)
            }
            
            # Log the input shapes for debugging
            logger.debug(
                f"ONNX inputs: input_ids={tokens.shape}, speaker_embedding={voice.shape}, speed={speed}, style=[[0.0]] (neutral float32 shape (1,1))", 
                extra={'subsys': 'tts', 'event': 'create_audio.inputs'}
            )
            
            # Run inference
            outputs = self.sess.run(None, inputs)
            
            # Extract audio from outputs
            if outputs and len(outputs) > 0 and outputs[0] is not None:
                audio = outputs[0].squeeze()
                
                # Log audio shape and stats
                logger.debug(
                    f"Generated audio: shape={audio.shape}, dtype={audio.dtype}", 
                    extra={'subsys': 'tts', 'event': 'create_audio.output'}
                )
            else:
                logger.warning(
                    "ONNX model returned empty output", 
                    extra={'subsys': 'tts', 'event': 'create_audio.empty'}
                )
                audio = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)  # 0.5 second of silence
        
            # Log performance metrics
            audio_duration = len(audio) / SAMPLE_RATE
            create_duration = time.time() - start_t
            speedup_factor = audio_duration / create_duration
            
            logger.debug(
                f"Created {audio_duration:.2f}s audio for {len(phonemes)} phonemes in {create_duration:.2f}s ({speedup_factor:.2f}x real-time)",
                extra={'subsys': 'tts', 'event': 'create_audio.complete'}
            )
            
            return audio, SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Error in ONNX inference: {e}", 
                       extra={'subsys': 'tts', 'event': 'create_audio.error'}, exc_info=True)
            raise
    
    def create(self, text: str, voice_id_or_embedding: Union[str, NDArray[np.float32]], 
              phonemes: Optional[str] = None, speed: float = 1.0, *, out_path: Optional[Path] = None) -> Path:
        """
        Create audio from text or phonemes using specified voice and write to WAV file.
        
        Args:
            text: Text to synthesize (used if phonemes not provided)
            voice_id_or_embedding: Voice ID string or embedding array
            phonemes: Optional phoneme string (if None, text is used)
            speed: Speed factor (1.0 is normal)
            out_path: Optional output path for the WAV file
            
        Returns:
            Path to the generated audio file
            
        Raises:
            TTSWriteError: If audio generation or file writing fails
        """
        # Handle voice ID or direct embedding
        voice_embedding = None
        voice_id = None
        
        if isinstance(voice_id_or_embedding, str):
            voice_id = voice_id_or_embedding
            if voice_id not in self.voices_data:
                available_voices = self.voices
                if not available_voices:
                    raise ValueError("No voices available")
                
                # Fall back to first available voice
                voice_id = available_voices[0]
                logger.warning(f"Voice '{voice_id_or_embedding}' not found, falling back to '{voice_id}'", 
                             extra={'subsys': 'tts', 'event': 'create.fallback_voice'})
            
            voice_embedding = self.voices_data[voice_id]
            logger.debug(f"Using voice: {voice_id}", extra={'subsys': 'tts', 'event': 'create.voice_id'})
        else:
            # Assume it's already a voice embedding
            voice_embedding = voice_id_or_embedding
            logger.debug(f"Using direct voice embedding with shape {voice_embedding.shape}", 
                       extra={'subsys': 'tts', 'event': 'create.direct_embedding'})
        
        # Use text directly if no phonemes provided
        if phonemes is None:
            # In a real implementation, we would convert text to phonemes here
            # For now, just use the text directly
            phonemes = text
            logger.debug(f"No phonemes provided, using text directly: {text[:50]}...", 
                       extra={'subsys': 'tts', 'event': 'create.text_as_phonemes'})
        
        # Log phonemiser and language
        logger.debug(
            f"Kokoro: voice={voice_id or 'custom'} lang={self.language} phonemiser={self.phonemiser}", 
            extra={'subsys': 'tts', 'event': 'create.params'}
        )
        
        # Create audio
        try:
            audio, sample_rate = self._create_audio(phonemes, voice_embedding, speed)
            
            # Normalize audio to be within float32 range for WAV files (-1.0 to 1.0)
            # Most audio libraries prefer float32 in [-1, 1] range for WAV files
            if audio.size > 0:
                # Check if we have any audio data
                if np.max(np.abs(audio)) > 0:
                    # Normalize to range [-1, 1]
                    audio = audio / np.max(np.abs(audio))
                    # Keep as float32 for better compatibility with audio libraries
                    audio = audio.astype(np.float32)
                    
                    # Log audio stats for debugging
                    logger.debug(
                        f"Audio samples: shape={audio.shape}, dtype={audio.dtype}, min={np.min(audio):.2f}, max={np.max(audio):.2f}, sr={sample_rate}", 
                        extra={'subsys': 'tts', 'event': 'create.audio_stats'}
                    )
                else:
                    # If audio is all zeros, just create zeros with float32 dtype
                    audio = np.zeros(len(audio), dtype=np.float32)
                    logger.error(
                        "Generated audio contains all zeros", 
                        extra={'subsys': 'tts', 'event': 'create.error.zeros'}
                    )
            else:
                # Empty audio, create a small silent segment
                logger.error(
                    "Generated empty audio (zero length)", 
                    extra={'subsys': 'tts', 'event': 'create.error.empty'}
                )
                audio = np.zeros(sample_rate // 2, dtype=np.float32)  # 0.5 second of silence
            
            # Ensure audio is a 1D array (WAV files expect 1D arrays)
            if len(audio.shape) > 1:
                # Reshape to 1D if it's a 2D array with shape like (1, N)
                audio = audio.reshape(-1)
                logger.debug(
                    f"Reshaped audio to 1D: {audio.shape}", 
                    extra={'subsys': 'tts', 'event': 'create.reshape'}
                )
            
            # Create a temporary file if no output path provided
            if out_path is None:
                try:
                    # Create temp directory if it doesn't exist
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    # Create temporary file with .wav extension
                    fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=CACHE_DIR)
                    os.close(fd)  # Close the file descriptor
                    out_path = Path(temp_path)
                except Exception as e:
                    logger.error(
                        f"Failed to create temporary file: {e}", 
                        extra={'subsys': 'tts', 'event': 'create.error.temp_file'}, 
                        exc_info=True
                    )
                    raise TTSWriteError(f"Failed to create temporary file: {e}")
            
            # Save as WAV using soundfile (more reliable for float32 audio)
            try:
                # Write as WAV file with subtype 'FLOAT' for float32 data
                sf.write(out_path, audio, sample_rate, subtype='FLOAT')
                logger.debug(
                    f"Saved audio to {out_path} using soundfile (FLOAT subtype)", 
                    extra={'subsys': 'tts', 'event': 'create.save', 'path': str(out_path), 'library': 'soundfile'}
                )
            except Exception as e:
                logger.warning(
                    f"Failed to save with soundfile: {e}, trying scipy", 
                    extra={'subsys': 'tts', 'event': 'create.fallback_scipy'}
                )
                # Fallback to scipy if soundfile fails
                try:
                    from scipy.io import wavfile
                    # Convert to int16 for scipy wavfile
                    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                    wavfile.write(out_path, sample_rate, audio_int16)
                    logger.debug(
                        f"Saved audio to {out_path} using scipy (int16)", 
                        extra={'subsys': 'tts', 'event': 'create.save', 'path': str(out_path), 'library': 'scipy'}
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save audio with both libraries: {e}", 
                        extra={'subsys': 'tts', 'event': 'create.error.save_failed'}, 
                        exc_info=True
                    )
                    raise TTSWriteError(f"Failed to save audio file: {e}")
            
            # Verify the file was created and has content
            if not out_path.exists() or out_path.stat().st_size == 0:
                logger.error(
                    f"Failed to create audio file or file is empty: {out_path}", 
                    extra={'subsys': 'tts', 'event': 'create.error.file_creation'}
                )
                raise TTSWriteError(f"Audio file is empty or does not exist: {out_path}")
                
            return out_path
            
        except TTSWriteError:
            # Re-raise TTSWriteError without wrapping
            raise
        except Exception as e:
            logger.error(
                f"Failed to generate speech: {e}", 
                extra={'subsys': 'tts', 'event': 'create.error'}, 
                exc_info=True
            )
            raise TTSWriteError(f"Failed to generate speech: {e}")
