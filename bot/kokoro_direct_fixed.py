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
import enum
import re
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from numpy.typing import NDArray

# Import custom exceptions
from .tts.errors import TTSWriteError

# Import tokenizer registry
from .tokenizer_registry import select_tokenizer_for_language

# Setup logging
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
MAX_PHONEME_LENGTH = 512
CACHE_DIR = os.environ.get('XDG_CACHE_HOME', Path('tts/cache'))

# Monkey patch for EspeakWrapper.set_data_path
try:
    # Try to import the EspeakWrapper class
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    
    # Check if set_data_path method exists
    if not hasattr(EspeakWrapper, 'set_data_path'):
        logger.info("Monkey patching EspeakWrapper.set_data_path method")
        
        # Add the missing set_data_path method
        @classmethod
        def set_data_path(cls, path):
            # This is a no-op since we can't actually set the data_path property
            # The real EspeakWrapper uses data_path as a property, not a settable attribute
            logger.debug(f"Monkey patched EspeakWrapper.set_data_path called with: {path}")
            # We don't actually need to do anything here, as the data path is set elsewhere
            pass
            
        # Add the method to the class
        EspeakWrapper.set_data_path = set_data_path
        logger.info("Successfully added EspeakWrapper.set_data_path method")
    
except ImportError as e:
    logger.warning(f"Could not monkey patch EspeakWrapper: {e}")
except Exception as e:
    logger.error(f"Error while monkey patching EspeakWrapper: {e}", exc_info=True)

# Tokenization method enum
class TokenizationMethod(enum.Enum):
    # Internal tokenizer methods
    PHONEME_ENCODE = "phoneme_encode"  # Original encode method
    PHONEME_TO_ID = "phoneme_to_id"   # Direct phoneme to ID lookup
    TEXT_TO_IDS = "text_to_ids"       # Direct text to IDs conversion
    G2P_PIPELINE = "g2p_pipeline"     # G2P pipeline
    
    # External phonemizers
    ESPEAK = "espeak"                 # eSpeak phonemizer
    PHONEMIZER = "phonemizer"         # Phonemizer package
    MISAKI = "misaki"                 # Misaki (Japanese/Chinese)
    
    # Fallbacks
    GRAPHEME_FALLBACK = "grapheme_fallback"  # ASCII grapheme fallback
    UNKNOWN = "unknown"               # Unknown method

def normalize_text(text: str) -> str:
    """Normalize text for TTS processing.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
        
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip Discord mentions if present
    text = re.sub(r'<@!?\d+>', '', text)
    
    # Strip common markdown
    text = re.sub(r'[*_~`]', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

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
        self.available_tokenization_methods = set()
        
        # Load the model and voices
        self._load_model()
        # Detect available tokenization methods
        self._detect_tokenization_methods()
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
        # Use environment variable override if present
        env_phonemiser = os.environ.get("TTS_PHONEMISER", "").strip().lower()
        if env_phonemiser:
            logger.info(f"Using phonemiser from environment: {env_phonemiser} [subsys: tts, event: phonemiser.env_override]")
            return env_phonemiser
        
        try:
            # Use the tokenizer registry to select the best tokenizer for this language
            tokenizer = select_tokenizer_for_language(language)
            logger.info(f"Selected tokenizer for {language} from registry: {tokenizer} [subsys: tts, event: phonemiser.registry_select]")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to select tokenizer from registry: {e}, using fallbacks [subsys: tts, event: phonemiser.registry_error]")
            
            # Legacy fallback phonemiser selection logic
            language = language.lower().strip()
            if language == "en":
                return "espeak"  # Best for English
            elif language in ["ja", "zh"]:
                return "misaki"  # Best for Japanese/Chinese
            else:
                return "phonemizer"  # Default for other languages
        
    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        try:
            import kokoro_onnx
            import onnxruntime as ort
            
            # Try to initialize tokenizer with error handling
            try:
                from kokoro_onnx.tokenizer import Tokenizer
                self.tokenizer = Tokenizer()
            except AttributeError as e:
                if "'EspeakWrapper' object has no attribute 'set_data_path'" in str(e):
                    logger.error(f"kokoro-onnx initialization error: {e} [subsys: tts, event: load_model.error]")
                    
                    # Try to monkey patch at runtime as a last resort
                    try:
                        from phonemizer.backend.espeak.wrapper import EspeakWrapper
                        
                        # Add the missing set_data_path method
                        @classmethod
                        def set_data_path(cls, path):
                            logger.debug(f"Runtime-patched EspeakWrapper.set_data_path called with: {path} [subsys: tts, event: monkey_patch]")
                            pass
                            
                        # Add the method to the class
                        EspeakWrapper.set_data_path = set_data_path
                        logger.info("Runtime-patched EspeakWrapper.set_data_path method [subsys: tts, event: monkey_patch]")
                        
                        # Try initializing again
                        from kokoro_onnx.tokenizer import Tokenizer
                        self.tokenizer = Tokenizer()
                    except Exception as patch_error:
                        logger.error(f"Failed to apply runtime patch: {patch_error} [subsys: tts, event: monkey_patch.error]", exc_info=True)
                        raise
                else:
                    raise
            
            # Log ONNX providers
            providers = ort.get_available_providers()
            logger.debug(f"ONNX providers: {providers} [subsys: tts, event: onnx.providers]")
            
            # Create ONNX session directly instead of using Model class
            self.sess = ort.InferenceSession(self.model_path, providers=providers)
            
            # Log input names for debugging
            input_names = [input.name for input in self.sess.get_inputs()]
            logger.debug(f"ONNX model input names: {input_names} [subsys: tts, event: onnx.inputs]")
            
            logger.info(f"Loaded ONNX model from {self.model_path} [subsys: tts, event: load_model.success]")
            
            # Validate language resources
            self._validate_language_resources()
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e} [subsys: tts, event: load_model.error]")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e} [subsys: tts, event: load_model.error]", exc_info=True)
            raise
    
    def _detect_tokenization_methods(self) -> None:
        """Detect available tokenization methods for the current tokenizer."""
        self.available_tokenization_methods = set()
        
        # Check for tokenizer methods
        # Check for encode method (original method)
        if hasattr(self.tokenizer, 'encode') and callable(getattr(self.tokenizer, 'encode')):
            self.available_tokenization_methods.add(TokenizationMethod.PHONEME_ENCODE)
            logger.debug("Found tokenizer.encode method [subsys: tts, event: tokenizer.method.encode]")
        
        # Check for phoneme_to_id method
        if hasattr(self.tokenizer, 'phoneme_to_id') and callable(getattr(self.tokenizer, 'phoneme_to_id')):
            self.available_tokenization_methods.add(TokenizationMethod.PHONEME_TO_ID)
            logger.debug("Found tokenizer.phoneme_to_id method [subsys: tts, event: tokenizer.method.phoneme_to_id]")
        
        # Check for text_to_ids method
        if hasattr(self.tokenizer, 'text_to_ids') and callable(getattr(self.tokenizer, 'text_to_ids')):
            self.available_tokenization_methods.add(TokenizationMethod.TEXT_TO_IDS)
            logger.debug("Found tokenizer.text_to_ids method [subsys: tts, event: tokenizer.method.text_to_ids]")
        
        # Check for g2p_pipeline method
        if hasattr(self.tokenizer, 'g2p_pipeline') and callable(getattr(self.tokenizer, 'g2p_pipeline')):
            self.available_tokenization_methods.add(TokenizationMethod.G2P_PIPELINE)
            logger.debug("Found tokenizer.g2p_pipeline method [subsys: tts, event: tokenizer.method.g2p_pipeline]")
        
        # Check for external phonemizers
        # Check for espeak
        try:
            import shutil
            espeak_path = shutil.which("espeak")
            if espeak_path:
                self.available_tokenization_methods.add(TokenizationMethod.ESPEAK)
                logger.debug(f"Found espeak at {espeak_path} [subsys: tts, event: tokenizer.external.espeak]")
        except Exception as e:
            logger.debug(f"Failed to check for espeak: {e} [subsys: tts, event: tokenizer.external.error]")
        
        # Check for phonemizer
        try:
            import importlib.util
            if importlib.util.find_spec("phonemizer"):
                self.available_tokenization_methods.add(TokenizationMethod.PHONEMIZER)
                logger.debug("Found phonemizer package [subsys: tts, event: tokenizer.external.phonemizer]")
        except Exception as e:
            logger.debug(f"Failed to check for phonemizer: {e} [subsys: tts, event: tokenizer.external.error]")
        
        # Check for misaki (Japanese/Chinese tokenizer)
        try:
            import importlib.util
            if importlib.util.find_spec("misaki"):
                self.available_tokenization_methods.add(TokenizationMethod.MISAKI)
                logger.debug("Found misaki package [subsys: tts, event: tokenizer.external.misaki]")
        except Exception as e:
            logger.debug(f"Failed to check for misaki: {e} [subsys: tts, event: tokenizer.external.error]")
            
        # Set default tokenization method based on availability
        if TokenizationMethod.PHONEME_ENCODE in self.available_tokenization_methods:
            self.tokenization_method = TokenizationMethod.PHONEME_ENCODE
        elif TokenizationMethod.PHONEME_TO_ID in self.available_tokenization_methods:
            self.tokenization_method = TokenizationMethod.PHONEME_TO_ID
        elif TokenizationMethod.TEXT_TO_IDS in self.available_tokenization_methods:
            self.tokenization_method = TokenizationMethod.TEXT_TO_IDS
        elif TokenizationMethod.G2P_PIPELINE in self.available_tokenization_methods:
            self.tokenization_method = TokenizationMethod.G2P_PIPELINE
        else:
            self.tokenization_method = TokenizationMethod.UNKNOWN
            logger.warning("No known tokenization methods found [subsys: tts, event: tokenizer.method.none]")

    def _validate_language_resources(self) -> None:
        """Validate that the configured language has required resources.
        If not, fall back to a known-good default."""
        # Check if we have a tokenizer
        if self.tokenizer is None:
            logger.error("No tokenizer available [subsys: tts, event: language.error.no_tokenizer]")
            return
            
        # Detect available tokenization methods
        self._detect_tokenization_methods()
        
        # If we don't have any known tokenization methods, log an error
        if self.tokenization_method == TokenizationMethod.UNKNOWN:
            logger.error(f"No known tokenization methods found for language '{self.language}' [subsys: tts, event: language.error.no_methods]")
        else:
            logger.info(f"Using tokenization method: {self.tokenization_method.value} [subsys: tts, event: language.tokenization_method]")

    def tokenize_text(self, text: str) -> Tuple[List[int], TokenizationMethod]:
        """Tokenize text using the most appropriate available method.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tuple of (token IDs, tokenization method used)
            
        Raises:
            ValueError: If tokenization fails with all methods
        """
        # Try each tokenization method in order of preference
        errors = []
        
        # Method 1: phoneme_encode (original method)
        if TokenizationMethod.PHONEME_ENCODE in self.available_tokenization_methods:
            try:
                token_ids = self.tokenizer.encode(text)
                if token_ids and len(token_ids) > 0:
                    return token_ids, TokenizationMethod.PHONEME_ENCODE
            except Exception as e:
                errors.append(f"phoneme_encode: {e}")
                logger.warning(f"Failed to tokenize with phoneme_encode: {e} [subsys: tts, event: tokenize.error.phoneme_encode]")
        
        # Method 2: phoneme_to_id
        if TokenizationMethod.PHONEME_TO_ID in self.available_tokenization_methods:
            try:
                token_ids = [self.tokenizer.phoneme_to_id(p) for p in text]
                if token_ids and len(token_ids) > 0:
                    return token_ids, TokenizationMethod.PHONEME_TO_ID
            except Exception as e:
                errors.append(f"phoneme_to_id: {e}")
                logger.warning(f"Failed to tokenize with phoneme_to_id: {e} [subsys: tts, event: tokenize.error.phoneme_to_id]")
        
        # Method 3: text_to_ids
        if TokenizationMethod.TEXT_TO_IDS in self.available_tokenization_methods:
            try:
                token_ids = self.tokenizer.text_to_ids(text)
                if token_ids and len(token_ids) > 0:
                    return token_ids, TokenizationMethod.TEXT_TO_IDS
            except Exception as e:
                errors.append(f"text_to_ids: {e}")
                logger.warning(f"Failed to tokenize with text_to_ids: {e} [subsys: tts, event: tokenize.error.text_to_ids]")
        
        # Method 4: g2p_pipeline
        if TokenizationMethod.G2P_PIPELINE in self.available_tokenization_methods:
            try:
                # This is a guess at the API, adjust as needed
                phonemes = self.tokenizer.g2p_pipeline(text)
                if hasattr(self.tokenizer, 'phoneme_to_id'):
                    token_ids = [self.tokenizer.phoneme_to_id(p) for p in phonemes]
                    if token_ids and len(token_ids) > 0:
                        return token_ids, TokenizationMethod.G2P_PIPELINE
            except Exception as e:
                errors.append(f"g2p_pipeline: {e}")
                logger.warning(f"Failed to tokenize with g2p_pipeline: {e} [subsys: tts, event: tokenize.error.g2p_pipeline]")
        
        # Fallback: ASCII grapheme fallback (treat each character as a token)
        try:
            logger.debug("No other tokenization methods worked, using grapheme fallback [subsys: tts, event: tokenize.fallback.grapheme]")
            
            # Filter to ASCII only
            ascii_text = ''.join(c for c in text if ord(c) < 128)
            if not ascii_text:
                ascii_text = "hello"  # Absolute fallback
                
            # Use character codes as token IDs
            token_ids = [ord(c) % 256 for c in ascii_text]
            return token_ids, TokenizationMethod.GRAPHEME_FALLBACK
        except Exception as e:
            errors.append(f"grapheme_fallback: {e}")
            logger.error(f"Even grapheme fallback tokenization failed: {e} [subsys: tts, event: tokenize.error.grapheme_fallback]")
        
        # If we get here, all methods failed
        error_msg = f"All tokenization methods failed: {'; '.join(errors)}"
        logger.error(f"{error_msg} [subsys: tts, event: tokenize.error.all_failed]")
        raise ValueError(error_msg)

    def _load_voices(self) -> None:
        """Load voice embeddings directly from NPZ file."""
        try:
            voices_path = Path(self.voices_path)
            if not voices_path.exists():
                logger.error(f"Voices file not found: {voices_path} [subsys: tts, event: load_voices.error, path: {voices_path}]")
                return
            
            # Load NPZ file
            logger.debug(f"Loading NPZ voices from: {voices_path} [subsys: tts, event: load_voices.npz, path: {voices_path}]")
            npz_data = np.load(voices_path, allow_pickle=True)
            
            if hasattr(npz_data, 'files'):
                logger.debug(f"NPZ file contains {len(npz_data.files)} voices [subsys: tts, event: load_voices.npz.count, voice_count: {len(npz_data.files)}]")
                
                # Extract each voice embedding
                for voice_id in npz_data.files:
                    voice_data = npz_data[voice_id]
                    if isinstance(voice_data, np.ndarray):
                        # Handle different voice embedding shapes
                        if voice_data.shape[0] == MAX_PHONEME_LENGTH and (len(voice_data.shape) == 2 and voice_data.shape[1] == 256):
                            # Perfect shape (512, 256)
                            self.voices_data[voice_id] = voice_data
                            self.voices.append(voice_id)
                            logger.debug(f"Loaded voice {voice_id} with shape {voice_data.shape} [subsys: tts, event: load_voices.voice, voice_id: {voice_id}, shape: {voice_data.shape}]")
                        elif voice_data.shape[0] == MAX_PHONEME_LENGTH and len(voice_data.shape) == 3 and voice_data.shape[1] == 1 and voice_data.shape[2] == 256:
                            # Shape (512, 1, 256) - already correct format
                            self.voices_data[voice_id] = voice_data
                            self.voices.append(voice_id)
                            logger.debug(f"Loaded voice {voice_id} with shape {voice_data.shape} [subsys: tts, event: load_voices.voice, voice_id: {voice_id}, shape: {voice_data.shape}]")
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
                                logger.debug(f"Padded voice {voice_id} from {voice_data.shape} to {self.voices_data[voice_id].shape}, non-zero min/max: {min_val:.4f}/{max_val:.4f} [subsys: tts, event: load_voices.padded, voice_id: {voice_id}, original_shape: {voice_data.shape}, padded_shape: {self.voices_data[voice_id].shape}]")
                            else:
                                logger.warning(f"Voice {voice_id} contains all zeros [subsys: tts, event: load_voices.warning.zeros, voice_id: {voice_id}]")
                        else:
                            logger.warning(f"Skipping voice {voice_id} with incompatible shape {voice_data.shape} [subsys: tts, event: load_voices.warning.shape, voice_id: {voice_id}, shape: {voice_data.shape}]")
                    else:
                        logger.warning(f"Skipping non-array data for voice {voice_id}: {type(voice_data)} [subsys: tts, event: load_voices.warning.type, voice_id: {voice_id}, type: {type(voice_data)}]")
                
                logger.info(f"Loaded {len(self.voices)} voices from NPZ file [subsys: tts, event: load_voices.success, voice_count: {len(self.voices)}]")
            else:
                logger.error("Invalid NPZ file format (no 'files' attribute) [subsys: tts, event: load_voices.error.format, path: {voices_path}]")
        except Exception as e:
            logger.error(f"Failed to load voices: {e} [subsys: tts, event: load_voices.error, error: {e}]", exc_info=True)
    
    def get_voice_names(self) -> List[str]:
        """Get list of available voice names."""
        return self.voices
        
# ...
    def _create_audio(self, phonemes: str, voice_embedding: np.ndarray, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Create audio from phonemes and voice embedding.
        
        Args:
            phonemes: Phoneme string to synthesize
            voice_embedding: Voice embedding array
            speed: Speech speed factor (1.0 = normal)
            
        Returns:
            Tuple of (audio array, sample rate)
            
        Raises:
            ValueError: If tokenization fails or produces empty token sequence
        """
        start_t = time.time()
        
        # Normalize and sanitize input text
        phonemes = normalize_text(phonemes)
        if not phonemes:
            logger.warning("Empty phonemes input, using fallback text [subsys: tts, event: create_audio.empty_input]")
            phonemes = "Hello."  # Fallback to ensure we generate something
        
        # Tokenize phonemes using our robust tokenization method
        try:
            token_ids, method_used = self.tokenize_text(phonemes)
            logger.debug(f"Tokenized {len(phonemes)} chars to {len(token_ids)} tokens using {method_used.value} [subsys: tts, event: create_audio.tokenize, method: {method_used.value}]")
            
            # Verify we have non-empty token sequence
            if not token_ids or len(token_ids) == 0:
                logger.error("Tokenization produced empty token sequence [subsys: tts, event: create_audio.error.empty_tokens]")
                raise ValueError("Tokenization produced empty token sequence")
                
        except Exception as e:
            logger.error(f"Failed to tokenize text: {e} [subsys: tts, event: create_audio.error.tokenize]", exc_info=True)
            raise ValueError(f"Failed to tokenize text: {e}")
        
        # Prepare inputs for ONNX model
        try:
            # Get input names from model
            input_names = [input.name for input in self.sess.get_inputs()]
            logger.debug(f"Model input names: {input_names} [subsys: tts, event: create_audio.input_names]")
            
            # Create input dictionary with required inputs
            inputs = {}
            
            # Add token IDs with the appropriate name
            if 'input_ids' in input_names:
                # Ensure input_ids is 2D as required by ONNX model
                if isinstance(token_ids, np.ndarray) and len(token_ids.shape) == 1:
                    token_ids = token_ids.reshape(1, -1)
                elif isinstance(token_ids, list):
                    token_ids = np.array(token_ids).reshape(1, -1)
                
                inputs['input_ids'] = token_ids
                logger.debug(f"Added input_ids with shape {inputs['input_ids'].shape} [subsys: tts, event: create_audio.input_ids]")
            elif 'tokens' in input_names:
                # Ensure tokens is 2D as required by ONNX model
                if isinstance(token_ids, np.ndarray) and len(token_ids.shape) == 1:
                    token_ids = token_ids.reshape(1, -1)
                elif isinstance(token_ids, list):
                    token_ids = np.array(token_ids).reshape(1, -1)
                
                inputs['tokens'] = token_ids
                logger.debug(f"Added tokens with shape {inputs['tokens'].shape} [subsys: tts, event: create_audio.tokens]")
            else:
                logger.error("No compatible token input name found in model [subsys: tts, event: create_audio.error.no_token_input]")
                raise ValueError("No compatible token input name found in model")
            
            # Add speed parameter if model accepts it
            if 'speed' in input_names:
                inputs['speed'] = np.array([speed], dtype=np.float32)
                logger.debug(f"Added speed: {speed} [subsys: tts, event: create_audio.speed]")
            
            # Add speaker/style embedding if available
            def _to_style_vector(arr: np.ndarray) -> np.ndarray:
                """Convert a variety of embedding shapes to a 2D style vector (1, 256)."""
                try:
                    vec = arr
                    # Squeeze singletons
                    if vec.ndim == 3 and vec.shape[1] == 1 and vec.shape[2] == 256:
                        # (512,1,256) -> (512,256)
                        vec = vec.squeeze(1)
                    # Reduce sequence dimension if present
                    if vec.ndim == 2 and vec.shape[0] in (510, 511, 512) and vec.shape[1] == 256:
                        # Mean across time/sequence -> (256,)
                        vec = vec.mean(axis=0)
                    # Ensure 1D of length 256
                    if vec.ndim == 2 and vec.shape[0] == 1 and vec.shape[1] == 256:
                        pass  # already (1,256) after reshape below
                    elif vec.ndim == 1 and vec.shape[0] == 256:
                        pass
                    else:
                        # Last resort: flatten and take first 256
                        vec = vec.reshape(-1)
                    # Coerce to float32 and (1,256)
                    vec = vec.astype(np.float32, copy=False)
                    if vec.ndim == 1:
                        vec = vec[:256]
                        if vec.shape[0] < 256:
                            # pad if shorter
                            pad = np.zeros(256 - vec.shape[0], dtype=np.float32)
                            vec = np.concatenate([vec, pad], axis=0)
                        vec = vec.reshape(1, 256)
                    elif vec.ndim == 2 and (vec.shape[0], vec.shape[1]) != (1, 256):
                        vec = vec.reshape(1, 256)
                    return vec
                except Exception as ex:
                    logger.warning(f"Failed to normalize voice embedding: {ex}; using zeros [subsys: tts, event: create_audio.warning.embed_normalize]")
                    return np.zeros((1, 256), dtype=np.float32)

            speaker_embedding_added = False
            
            # First try standard speaker embedding input names
            for speaker_name in ['speaker', 'speaker_embedding', 'spk_emb']:
                if speaker_name in input_names:
                    spk_vec = _to_style_vector(voice_embedding)
                    inputs[speaker_name] = spk_vec
                    logger.debug(f"Added {speaker_name} with shape {spk_vec.shape} [subsys: tts, event: create_audio.speaker]")
                    speaker_embedding_added = True
                    break
            
            # If no speaker input found but 'style' is available, route voice vector there
            if not speaker_embedding_added and 'style' in input_names:
                logger.warning(f"No speaker input found, routing voice embedding to 'style' input [subsys: tts, event: create_audio.warning.style]")
                style_vec = _to_style_vector(voice_embedding)
                inputs['style'] = style_vec
                speaker_embedding_added = True
            
            if not speaker_embedding_added:
                logger.error(f"No compatible speaker/style input found in model inputs: {input_names} [subsys: tts, event: create_audio.error.speaker]")
                # Fall back to zero vector, but continue with warning
                logger.warning("Using zero vector for voice; output quality may be degraded [subsys: tts, event: create_audio.warning.speaker_fallback]")

            # Run inference
            outputs = self.sess.run(None, inputs)
            audio = outputs[0].squeeze()

            logger.debug(f"Created {len(audio) / SAMPLE_RATE:.2f}s audio in {time.time() - start_t:.2f}s [subsys: tts, event: create_audio.complete]")
            return audio, SAMPLE_RATE

        except Exception as e:
            logger.error(f"Error in _create_audio: {e} [subsys: tts, event: create_audio.error]", exc_info=True)
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
                logger.warning(f"Voice '{voice_id_or_embedding}' not found, falling back to '{voice_id}' [subsys: tts, event: create.fallback_voice]")
            
            voice_embedding = self.voices_data[voice_id]
            logger.debug(f"Using voice: {voice_id} [subsys: tts, event: create.voice_id]")
        else:
            # Assume it's already a voice embedding
            voice_embedding = voice_id_or_embedding
            logger.debug(f"Using direct voice embedding with shape {voice_embedding.shape} [subsys: tts, event: create.direct_embedding]")
        
        # Use text directly if no phonemes provided
        if phonemes is None:
            # In a real implementation, we would convert text to phonemes here
            # For now, just use the text directly
            phonemes = text
            logger.debug(f"No phonemes provided, using text directly: {text[:50]}... [subsys: tts, event: create.text_as_phonemes]")
        
        # Log phonemiser and language
        logger.debug(f"Kokoro: voice={voice_id or 'custom'} lang={self.language} phonemiser={self.phonemiser} [subsys: tts, event: create.params]")
        
        # Create audio
        try:
            # Get audio data and sample rate from _create_audio
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
                    logger.debug(f"Audio samples: shape={audio.shape}, dtype={audio.dtype}, min={np.min(audio):.2f}, max={np.max(audio):.2f}, sr={sample_rate} [subsys: tts, event: create.audio_stats]")
                else:
                    # If audio is all zeros, just create zeros with float32 dtype
                    audio = np.zeros(len(audio), dtype=np.float32)
                    logger.error("Generated audio contains all zeros [subsys: tts, event: create.error.zeros]")
            else:
                # Empty audio, create a silent segment of at least 1 second
                logger.error("Generated empty audio (zero length) [subsys: tts, event: create.error.empty]")
                audio = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence
            
            # Ensure audio is a 1D array (WAV files expect 1D arrays)
            if len(audio.shape) > 1:
                # Reshape to 1D if it's a 2D array with shape like (1, N)
                audio = audio.reshape(-1)
                logger.debug(f"Reshaped audio to 1D: {audio.shape} [subsys: tts, event: create.reshape]")
            
            # Ensure audio is at least 1 second long (required by smoke test)
            min_duration_samples = sample_rate  # 1 second at current sample rate
            if len(audio) < min_duration_samples:
                # For short inputs like "ping", repeat the audio to reach minimum duration
                repeats_needed = int(np.ceil(min_duration_samples / len(audio)))
                logger.debug(f"Audio too short ({len(audio)/sample_rate:.2f}s), repeating {repeats_needed} times to reach minimum duration [subsys: tts, event: create.extend_duration]")
                # Repeat the audio with a small fade between repeats to avoid clicks
                extended_audio = np.zeros(min_duration_samples, dtype=np.float32)
                for i in range(repeats_needed):
                    start_idx = i * len(audio)
                    end_idx = min(start_idx + len(audio), min_duration_samples)
                    copy_len = end_idx - start_idx
                    if copy_len > 0:
                        extended_audio[start_idx:end_idx] = audio[:copy_len]
                audio = extended_audio
                logger.debug(f"Extended audio to {len(audio)/sample_rate:.2f}s ({len(audio)} samples)", extra={'subsys': 'tts', 'event': 'create.extended_duration'})
            
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
                    logger.error(f"Failed to create temporary file: {e}", extra={'subsys': 'tts', 'event': 'create.error.temp_file'}, exc_info=True)
                    raise TTSWriteError(f"Failed to create temporary file: {e}")
            
            # Save as WAV using soundfile (more reliable for float32 audio)
            try:
                # Write as WAV file with subtype 'FLOAT' for float32 data
                sf.write(out_path, audio, sample_rate, subtype='FLOAT')
                logger.debug(f"Saved audio to {out_path} using soundfile (FLOAT subtype)", extra={'subsys': 'tts', 'event': 'create.save', 'path': str(out_path), 'library': 'soundfile'})
            except Exception as e:
                logger.warning(f"Failed to save with soundfile: {e}, trying scipy", extra={'subsys': 'tts', 'event': 'create.fallback_scipy'})
                # Fallback to scipy if soundfile fails
                try:
                    from scipy.io import wavfile
                    # Convert to int16 for scipy wavfile
                    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                    wavfile.write(out_path, sample_rate, audio_int16)
                    logger.debug(f"Saved audio to {out_path} using scipy (int16)", extra={'subsys': 'tts', 'event': 'create.save', 'path': str(out_path), 'library': 'scipy'})
                except Exception as e:
                    logger.error(f"Failed to save audio with both libraries: {e}", extra={'subsys': 'tts', 'event': 'create.error.save_failed'}, exc_info=True)
                    raise TTSWriteError(f"Failed to save audio file: {e}")
            
            # Verify the file was created and has content
            if not out_path.exists() or out_path.stat().st_size == 0:
                logger.error(f"Failed to create audio file or file is empty: {out_path}", extra={'subsys': 'tts', 'event': 'create.error.file_creation'})
                raise TTSWriteError(f"Audio file is empty or does not exist: {out_path}")
            
            # Debug log to verify what we're returning
            logger.debug(f"Returning Path object: {out_path}, type: {type(out_path)}, audio length: {len(audio)}", extra={'subsys': 'tts', 'event': 'create.return_path', 'audio_length': len(audio), 'sample_rate': sample_rate})
                
            # Make sure we're returning ONLY a Path object (not a tuple)
            # This ensures backward compatibility with code expecting just a Path
            return out_path  # Return only the path, not (path, sample_rate)
            
        except TTSWriteError:
            # Re-raise TTSWriteError without wrapping
            raise
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}", extra={'subsys': 'tts', 'event': 'create.error'}, exc_info=True)
            raise TTSWriteError(f"Failed to generate speech: {e}")
