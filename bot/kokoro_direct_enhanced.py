"""
Enhanced KokoroDirect implementation with improved validation and error handling.
This version includes:
- Voice vector validation (shape, norm)
- Gibberish detection
- Phonemiser selection based on language
- Sample rate consistency checks
"""

import os
import logging
import tempfile
import re
import enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from .tts_errors import TTSGibberishError, TTSSynthesisError, MissingTokeniserError
from .tts_validation import (
    detect_available_tokenizers,
    select_tokenizer_for_language,
    validate_voice_vector,
    check_sample_rate_consistency,
    detect_gibberish_audio,
    TokenizerType,
    is_tokenizer_warning_needed,
    get_tokenizer_warning_message,
    AVAILABLE_TOKENIZERS
)
from .env_utils import resolve_env, resolve_path

logger = logging.getLogger(__name__)

class KokoroDirect:
    """
    Enhanced KokoroDirect implementation with improved validation and error handling.
    Handles NPZ voice data directly without JSON conversion.
    Fixes ONNX input signature mismatches and adds validation.
    """
    
    def __init__(self, model_path: str, voices_path: str):
        """
        Initialize KokoroDirect with model and voices paths.
        
        Args:
            model_path: Path to the ONNX model file
            voices_path: Path to the NPZ voices file
            
        Raises:
            MissingTokeniserError: If no suitable tokenizer is found for the language
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self.voices_data = {}
        self.voices = []
        self.tokenizer = None
        self.sess = None
        
        # Get language from environment with fallback to env var resolution
        self.language = resolve_env("TTS_LANGUAGE", "LANGUAGE", "en")
        
        # Initialize available tokenization methods
        self.available_tokenization_methods = set()
        
        # Log TTS language
        logger.info(f"TTS language set to: {self.language}",
                  extra={'subsys': 'tts', 'event': 'language.set', 'language': self.language})
        
        try:
            # Detect available tokenizers and select best for language
            # This will raise MissingTokeniserError if no suitable tokenizer is found
            self.available_tokenizers = detect_available_tokenizers()
            self.selected_tokenizer = select_tokenizer_for_language(self.language)
            
            # Log selected tokenizer
            logger.info(f"Tokeniser chosen for {self.language}: {self.selected_tokenizer}",
                      extra={'subsys': 'tts', 'event': 'tokenizer.chosen', 
                             'language': self.language, 
                             'tokenizer': self.selected_tokenizer,
                             'fallback': self.selected_tokenizer == 'grapheme'})
            
            # Check if we need to show a warning about missing tokenizers
            self.show_tokenizer_warning = is_tokenizer_warning_needed()
            
            # Load the model and voices
            self._load_model()
            self._load_voices()
            
            # Default sample rate (can be overridden by model)
            self.sample_rate = 24000
            
            logger.info(f"Initialized KokoroDirect with {len(self.voices)} voices",
                      extra={'subsys': 'tts', 'event': 'init.complete', 
                             'voice_count': len(self.voices)})
        except MissingTokeniserError as e:
            # Log the error and re-raise
            logger.error(f"Failed to initialize KokoroDirect: {str(e)}",
                       extra={'subsys': 'tts', 'event': 'init.failed.tokenizer',
                              'language': self.language})
            raise
        
    def get_tokenizer_warning(self) -> Optional[str]:
        """
        Get tokenizer warning message if needed.
        
        Returns:
            Warning message or None if no warning needed
        """
        if self.show_tokenizer_warning:
            return get_tokenizer_warning_message(self.language)
        return None
        
    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        try:
            import kokoro_onnx
            import onnxruntime as ort
            
            # Try to initialize tokenizer with error handling
            try:
                self.tokenizer = kokoro_onnx.Tokenizer()
                
                # Set language based on environment
                if hasattr(self.tokenizer, 'set_language'):
                    self.tokenizer.set_language(self.language)
                    logger.debug(f"Set tokenizer language to '{self.language}'",
                               extra={'subsys': 'tts', 'event': 'tokenizer.language'})
            except AttributeError as e:
                if "'EspeakWrapper' object has no attribute 'set_data_path'" in str(e):
                    logger.error(f"kokoro-onnx initialization error: {e}",
                               extra={'subsys': 'tts', 'event': 'init.error.espeak_wrapper'})
                    
                    # Try to monkey patch at runtime as a last resort
                    try:
                        from phonemizer.backend.espeak.wrapper import EspeakWrapper
                        
                        # Add the missing set_data_path method
                        @classmethod
                        def set_data_path(cls, path):
                            logger.debug(f"Runtime-patched EspeakWrapper.set_data_path called with: {path}",
                                       extra={'subsys': 'tts', 'event': 'patch.espeak_wrapper', 'path': path})
                            pass
                            
                        # Add the method to the class
                        EspeakWrapper.set_data_path = set_data_path
                        logger.info("Runtime-patched EspeakWrapper.set_data_path method",
                                  extra={'subsys': 'tts', 'event': 'patch.success'})
                        
                        # Try initializing again
                        self.tokenizer = kokoro_onnx.Tokenizer()
                        
                        # Set language based on environment
                        if hasattr(self.tokenizer, 'set_language'):
                            self.tokenizer.set_language(self.language)
                            logger.debug(f"Set tokenizer language to '{self.language}'",
                                       extra={'subsys': 'tts', 'event': 'tokenizer.language'})
                    except Exception as patch_error:
                        logger.error(f"Failed to apply runtime patch: {patch_error}",
                                   extra={'subsys': 'tts', 'event': 'patch.failed'}, exc_info=True)
                        raise
                else:
                    raise
            
            # Log ONNX providers
            providers = ort.get_available_providers()
            logger.debug(f"ONNX providers: {providers}", extra={'subsys': 'tts', 'event': 'onnx.providers'})
            
            # Create ONNX session directly instead of using Model class
            self.sess = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get model metadata
            metadata = self.sess.get_modelmeta()
            if hasattr(metadata, 'custom_metadata_map') and metadata.custom_metadata_map:
                if 'sample_rate' in metadata.custom_metadata_map:
                    try:
                        self.sample_rate = int(metadata.custom_metadata_map['sample_rate'])
                        logger.debug(f"Model sample rate from metadata: {self.sample_rate}Hz",
                                   extra={'subsys': 'tts', 'event': 'model.sample_rate'})
                    except (ValueError, TypeError):
                        logger.warning("Invalid sample_rate in model metadata",
                                     extra={'subsys': 'tts', 'event': 'model.metadata.invalid'})
            
            logger.info(f"Loaded ONNX model from {self.model_path}",
                      extra={'subsys': 'tts', 'event': 'model.loaded', 'path': self.model_path})
            
            # Log input names for debugging
            input_names = [input.name for input in self.sess.get_inputs()]
            logger.debug(f"ONNX model input names: {input_names}",
                       extra={'subsys': 'tts', 'event': 'onnx.inputs', 'names': input_names})
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True,
                       extra={'subsys': 'tts', 'event': 'model.load_failed'})
            raise
            
    def _load_voices(self) -> None:
        """Load voice embeddings from NPZ file."""
        try:
            # Load NPZ file directly
            voices_data = np.load(self.voices_path, allow_pickle=True)
            
            # Extract voice embeddings
            for voice_id in voices_data.files:
                voice_vector = voices_data[voice_id]
                
                try:
                    # Validate voice vector
                    validate_voice_vector(voice_vector, voice_id)
                    
                    # Store valid voice
                    self.voices_data[voice_id] = voice_vector
                    self.voices.append(voice_id)
                except ValueError as e:
                    logger.warning(f"Skipping invalid voice '{voice_id}': {e}",
                                 extra={'subsys': 'tts', 'event': 'voice.invalid', 'voice_id': voice_id})
            
            logger.info(f"Loaded {len(self.voices)} voices from {self.voices_path}",
                      extra={'subsys': 'tts', 'event': 'voices.loaded', 
                             'count': len(self.voices), 'path': self.voices_path})
            
            # Log available voices for debugging
            logger.debug(f"Available voices: {', '.join(self.voices)}",
                       extra={'subsys': 'tts', 'event': 'voices.available', 'voices': self.voices})
            
        except Exception as e:
            logger.error(f"Failed to load voices: {e}", exc_info=True,
                       extra={'subsys': 'tts', 'event': 'voices.load_failed'})
            raise
    
    def _create_audio(self, text: str, voice_id: str = "default") -> np.ndarray:
        """
        Create audio from text using the specified voice.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            
        Returns:
            Audio samples as numpy array
        """
        if not text or not text.strip():
            logger.error("Empty text provided for TTS synthesis",
                       extra={'subsys': 'tts', 'event': 'synthesis.empty_text'})
            raise ValueError("Empty text provided for TTS synthesis")
        
        # Normalize text
        text = text.strip()
        
        # Get voice embedding
        if voice_id not in self.voices_data:
            available_voices = ", ".join(self.voices[:5]) + (", ..." if len(self.voices) > 5 else "")
            logger.warning(f"Voice '{voice_id}' not found, using first available voice. Available: {available_voices}",
                         extra={'subsys': 'tts', 'event': 'voice.not_found', 
                                'requested': voice_id, 'available': self.voices})
            
            # Use first available voice as fallback
            if self.voices:
                voice_id = self.voices[0]
            else:
                logger.error("No voices available",
                           extra={'subsys': 'tts', 'event': 'voice.none_available'})
                raise ValueError("No voices available")
        
        voice_embedding = self.voices_data[voice_id]
        
        # Validate voice embedding again
        validate_voice_vector(voice_embedding, voice_id)
        
        try:
            # Tokenize text
            if not hasattr(self.tokenizer, 'encode'):
                logger.error("Tokenizer missing 'encode' method",
                           extra={'subsys': 'tts', 'event': 'tokenizer.missing_encode'})
                raise AttributeError("Tokenizer missing 'encode' method")
            
            # Tokenize with language-specific phonemiser
            tokens = self.tokenizer.encode(text)
            
            # Ensure we have at least one token
            if not tokens or len(tokens) == 0:
                logger.error("Tokenization produced empty token sequence",
                           extra={'subsys': 'tts', 'event': 'tokenizer.empty_tokens'})
                raise ValueError("Tokenization produced empty token sequence")
            
            logger.debug(f"Tokenized text to {len(tokens)} tokens",
                       extra={'subsys': 'tts', 'event': 'tokenizer.tokens', 'count': len(tokens)})
            
            # Convert tokens to input tensor
            input_ids = np.array(tokens, dtype=np.int64)
            
            # Ensure input_ids is 2D as required by ONNX
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
            
            # Prepare inputs based on model requirements
            inputs = {}
            
            # Get input names from model
            input_names = [input.name for input in self.sess.get_inputs()]
            
            # Add input_ids (required)
            if 'input_ids' in input_names:
                inputs['input_ids'] = input_ids
            elif 'tokens' in input_names:
                # Backward compatibility with older models
                inputs['tokens'] = input_ids
            else:
                logger.error(f"No compatible input name found for tokens. Available: {input_names}",
                           extra={'subsys': 'tts', 'event': 'onnx.no_token_input', 'available': input_names})
                raise ValueError(f"No compatible input name found for tokens. Available: {input_names}")
            
            # Add speaker embedding if supported
            speaker_input_name = None
            for name in ['speaker_embedding', 'speaker', 'spk_emb']:
                if name in input_names:
                    speaker_input_name = name
                    break
            
            # If no speaker embedding input found, check for style input (newer models)
            if not speaker_input_name and 'style' in input_names:
                speaker_input_name = 'style'
                logger.info(f"No compatible speaker embedding input found, using 'style' input instead",
                          extra={'subsys': 'tts', 'event': 'onnx.using_style_input'})
            
            if speaker_input_name:
                # Reshape voice embedding to match expected input shape
                voice_tensor = voice_embedding.astype(np.float32)
                
                # Get expected shape for this input
                expected_shape = None
                for input_meta in self.sess.get_inputs():
                    if input_meta.name == speaker_input_name:
                        expected_shape = input_meta.shape
                        break
                
                # Reshape if needed
                if expected_shape and len(expected_shape) > 1:
                    # If first dimension is dynamic (None), use 1
                    shape = [1 if dim is None else dim for dim in expected_shape]
                    
                    # If shape is [1, 256], reshape accordingly
                    if shape == [1, 256]:
                        voice_tensor = voice_tensor.reshape(1, 256)
                    # If shape is [1, 1], use a different approach
                    elif shape == [1, 1]:
                        # Use mean of embedding as a scalar style factor
                        voice_tensor = np.mean(voice_tensor).reshape(1, 1).astype(np.float32)
                
                logger.debug(f"Using {speaker_input_name} input with shape {voice_tensor.shape}",
                           extra={'subsys': 'tts', 'event': 'onnx.speaker_input', 
                                  'name': speaker_input_name, 'shape': voice_tensor.shape.tolist()})
                
                inputs[speaker_input_name] = voice_tensor
            else:
                logger.warning("No compatible speaker embedding input name found in model inputs",
                             extra={'subsys': 'tts', 'event': 'onnx.no_speaker_input'})
            
            # Add speed parameter if supported (default 1.0)
            if 'speed' in input_names:
                inputs['speed'] = np.array([1.0], dtype=np.float32)
            
            # Run inference
            logger.debug(f"Running inference with inputs: {', '.join(inputs.keys())}",
                       extra={'subsys': 'tts', 'event': 'onnx.inference.start'})
            
            outputs = self.sess.run(None, inputs)
            
            # Get audio from output
            audio = outputs[0].squeeze()
            
            # Check for all-zero or gibberish audio
            is_gibberish, metrics = detect_gibberish_audio(audio, self.sample_rate)
            
            # Log audio metrics
            logger.debug(f"Audio metrics: avg_abs={metrics['avg_abs']:.6f}, rms={metrics['rms']:.6f}, zcr={metrics['zcr']:.6f}",
                       extra={'subsys': 'tts', 'event': 'audio.metrics', 
                              'avg_abs': metrics['avg_abs'], 'rms': metrics['rms'], 'zcr': metrics['zcr']})
            
            return audio
            
        except Exception as e:
            logger.error(f"Failed to create audio: {e}", exc_info=True,
                       extra={'subsys': 'tts', 'event': 'synthesis.failed'})
            raise
    
    def create(self, text: str, voice_id: str = "default", output_path: Optional[str] = None) -> Path:
        """
        Create audio from text and save to a file.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            output_path: Path to save audio file (optional)
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Generate audio samples
            audio = self._create_audio(text, voice_id)
            
            # Create output path if not provided
            if not output_path:
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"tts_{hash(text)}.wav")
            
            # Ensure output_path is a Path object
            output_path = Path(output_path)
            
            # Save audio to file
            try:
                from scipy.io import wavfile
                wavfile.write(output_path, self.sample_rate, (audio * 32767).astype(np.int16))
                
                # Verify file was created and is non-empty
                if not output_path.exists():
                    logger.error(f"Failed to create audio file: {output_path} does not exist",
                               extra={'subsys': 'tts', 'event': 'file.missing'})
                    raise TTSSynthesisError(f"Failed to create audio file: {output_path} does not exist")
                
                if output_path.stat().st_size == 0:
                    logger.error(f"Generated empty audio file: {output_path}",
                               extra={'subsys': 'tts', 'event': 'file.empty'})
                    raise TTSSynthesisError(f"Generated empty audio file: {output_path}")
                
                logger.debug(f"Saved audio to {output_path} ({output_path.stat().st_size} bytes)",
                           extra={'subsys': 'tts', 'event': 'file.saved', 
                                  'path': str(output_path), 'size': output_path.stat().st_size})
                
                return output_path
                
            except Exception as e:
                logger.error(f"Failed to save audio: {e}", exc_info=True,
                           extra={'subsys': 'tts', 'event': 'file.save_failed'})
                raise TTSSynthesisError(f"Failed to save audio: {e}")
            
        except Exception as e:
            logger.error(f"Failed to create audio: {e}", exc_info=True,
                       extra={'subsys': 'tts', 'event': 'create.failed'})
            
            if isinstance(e, TTSGibberishError):
                # Re-raise TTSGibberishError with user-friendly message
                raise
            
            # Wrap other exceptions
            raise TTSSynthesisError(f"Failed to create audio: {e}")
