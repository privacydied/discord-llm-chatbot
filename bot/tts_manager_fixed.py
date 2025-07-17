"""
TTSManager implementation with robust error handling and proper audio file validation.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Import custom exceptions and utilities
from .tts_errors import TTSWriteError, MissingTokeniserError
from .env_utils import resolve_env, resolve_path, get_config_singleton
from .tokenizer_registry import TokenizerRegistry, discover_tokenizers, select_tokenizer_for_language, is_tokenizer_warning_needed

# Setup logging
logger = logging.getLogger(__name__)

class TTSManager:
    """
    Manages TTS functionality with proper error handling and validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TTS manager with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backend = resolve_env('TTS_BACKEND', 'TTS_ENGINE', 'kokoro-onnx')
        self.voice = resolve_env('TTS_VOICE', 'VOICE_ID', 'default')
        self.available = False
        self.kokoro = None
        
        # Use resolve_path for cache directory
        cache_dir = resolve_env('XDG_CACHE_HOME', 'CACHE_DIR', 'tts/cache')
        self.cache_dir = Path(cache_dir)
        
        # Use resolve_path for model and voice paths
        # New format: TTS_MODEL_PATH, TTS_VOICES_PATH
        # Old format: TTS_MODEL_FILE, TTS_VOICE_FILE
        self.model_path = resolve_path(
            'TTS_MODEL_PATH',
            'TTS_MODEL_FILE',
            'tts/onnx/kokoro-v1.0.onnx'
        )
        
        self.voices_path = resolve_path(
            'TTS_VOICES_PATH',
            'TTS_VOICE_FILE',
            'tts/voices/voices-v1.0.bin'
        )
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Perform late-binding tokenizer discovery
        self._discover_tokenizers()
        
        # Check if TTS is available
        self._check_availability()
        
    # This method is now deprecated in favor of using env_utils.resolve_path
    def _get_env_path_with_fallback(self, primary: str, fallback: str, default: str) -> Path:
        """
        DEPRECATED: Use env_utils.resolve_path instead.
        Get path from environment variables with fallback support.
        
        Args:
            primary: Primary (new) environment variable name
            fallback: Fallback (old) environment variable name
            default: Default path if neither env var is set
            
        Returns:
            Path object for the requested resource
        """
        logger.warning(
            "TTSManager._get_env_path_with_fallback is deprecated, use env_utils.resolve_path instead",
            extra={'subsys': 'tts', 'event': 'method.deprecated'}
        )
        return resolve_path(primary, fallback, default)
    
    def _discover_tokenizers(self) -> None:
        """Perform late-binding tokenizer discovery."""
        # Get the tokenizer registry singleton and discover available tokenizers
        registry = TokenizerRegistry.get_instance()
        available = registry.discover_tokenizers()
        
        # Log the discovery results
        available_count = sum(1 for v in available.values() if v)
        logger.info(f"Tokeniser discovery completed post-boot: {available_count} tokenizers available", 
                  extra={'subsys': 'tts', 'event': 'tokenizer.discovery.complete'})
        
        # Log available tokenizers by type
        binaries = [name for name, avail in available.items() 
                   if avail and name in ('espeak', 'espeak-ng')]
        modules = [name for name, avail in available.items() 
                  if avail and name in ('phonemizer', 'g2p_en', 'misaki')]
        
        logger.info(f"Tokeniser binaries: {set(binaries)}", 
                  extra={'subsys': 'tts', 'event': 'tokenizer.discovery.binaries'})
        logger.info(f"Tokeniser modules: {set(modules)}", 
                  extra={'subsys': 'tts', 'event': 'tokenizer.discovery.modules'})
        
        # Try to select a tokenizer for the current language
        try:
            language = os.environ.get('TTS_LANGUAGE', 'en')
            selected = select_tokenizer_for_language(language)
            logger.info(f"Selected tokenizer for {language}: {selected}", 
                      extra={'subsys': 'tts', 'event': 'tokenizer.selection'})
        except MissingTokeniserError as e:
            logger.error(f"Failed to select tokenizer: {e}", 
                       extra={'subsys': 'tts', 'event': 'tokenizer.selection.error'})
        
        # Check if warning is needed
        if is_tokenizer_warning_needed():
            logger.warning("Tokenizer warning needed for user notification", 
                         extra={'subsys': 'tts', 'event': 'tokenizer.warning_needed'})
    
    def _check_availability(self) -> None:
        """Check if the TTS backend is available."""
        if self.backend == 'kokoro-onnx':
            try:
                # Try to import kokoro_onnx
                import kokoro_onnx
                self.available = True
                logger.info("kokoro-onnx TTS backend is available", 
                           extra={'subsys': 'tts', 'event': 'check.available'})
            except ImportError:
                self.available = False
                logger.warning("kokoro-onnx TTS backend is not available (ImportError)", 
                              extra={'subsys': 'tts', 'event': 'check.unavailable'})
        else:
            self.available = False
            logger.warning(f"Unsupported TTS backend: {self.backend}", 
                          extra={'subsys': 'tts', 'event': 'check.unsupported'})
    
    async def load_model(self) -> None:
        """
        Load the TTS model asynchronously.
        """
        if not self.available:
            logger.warning("TTS is not available, skipping model loading", 
                          extra={'subsys': 'tts', 'event': 'load.skipped'})
            return
            
        try:
            # Import here to avoid dependency issues
            from .kokoro_direct_fixed import KokoroDirect
            
            # Initialize KokoroDirect with our model and voice paths
            # These already handle both old and new environment variable formats
            self.kokoro = KokoroDirect(
                model_path=str(self.model_path),
                voices_path=str(self.voices_path),
                voice_name=self.voice
            )
            
            logger.info(
                f"Initialized kokoro-onnx with model={self.model_path}, voices={self.voices_path}, voice={self.voice}", 
                extra={'subsys': 'tts', 'event': 'kokoro.init'}
            )
            
            # Log available voices
            voices = self.kokoro.get_voice_names()
            logger.info(f"TTS model loaded with {len(voices)} voices: {', '.join(voices[:5])}{'...' if len(voices) > 5 else ''}", 
                       extra={'subsys': 'tts', 'event': 'load.success'})
            
            # Verify voice exists
            if self.voice not in voices and self.voice != 'default':
                logger.warning(f"Requested voice '{self.voice}' not found, will use fallback", 
                              extra={'subsys': 'tts', 'event': 'load.voice_missing'})
            
            # Set availability based on successful loading
            self.available = True
            
        except Exception as e:
            self.available = False
            logger.error(f"Failed to load TTS model: {e}", 
                        extra={'subsys': 'tts', 'event': 'load.error'}, exc_info=True)
    
    async def generate_speech(self, text: str, voice: Optional[str] = None, 
                             out_path: Optional[Path] = None, pcm16: bool = False) -> Optional[Path]:
        """
        Generate speech from text and return path to the audio file.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID to use (defaults to self.voice)
            out_path: Optional output path for the WAV file
            pcm16: Whether to generate PCM16 format (for legacy compatibility)
            
        Returns:
            Path to the generated audio file or None if generation failed
            
        Raises:
            TTSWriteError: If audio generation or file writing fails
        """
        if not self.available or self.kokoro is None:
            logger.warning("TTS is not available, cannot generate speech", 
                          extra={'subsys': 'tts', 'event': 'generate.unavailable'})
            return None
            
        if not text:
            logger.warning("Empty text provided for TTS, skipping", 
                          extra={'subsys': 'tts', 'event': 'generate.empty_text'})
            return None
            
        # Use provided voice or default
        voice_id = voice or self.voice
        
        # Create temporary output path if not provided
        if out_path is None:
            try:
                fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=self.cache_dir)
                os.close(fd)  # Close the file descriptor
                out_path = Path(temp_path)
            except Exception as e:
                logger.error(f"Failed to create temporary file: {e}", 
                            extra={'subsys': 'tts', 'event': 'generate.temp_error'}, exc_info=True)
                raise TTSWriteError(f"Failed to create temporary file: {e}")
        
        try:
            # Generate speech using kokoro
            logger.debug(f"Generating speech for text: '{text[:50]}...' with voice '{voice_id}'", 
                        extra={'subsys': 'tts', 'event': 'generate.start'})
            
            # Call kokoro.create with the text and voice
            result_path = self.kokoro.create(text, voice_id, out_path=out_path)
            
            # Verify the file exists and has content
            if not result_path.exists():
                logger.error(f"Generated audio file does not exist: {result_path}", 
                            extra={'subsys': 'tts', 'event': 'generate.file_missing'})
                raise TTSWriteError(f"Generated audio file does not exist: {result_path}")
                
            if result_path.stat().st_size == 0:
                logger.error(f"Generated audio file is empty: {result_path}", 
                            extra={'subsys': 'tts', 'event': 'generate.file_empty'})
                raise TTSWriteError(f"Generated audio file is empty: {result_path}")
                
            logger.info(f"Successfully generated speech to {result_path} ({result_path.stat().st_size} bytes)", 
                       extra={'subsys': 'tts', 'event': 'generate.success'})
            
            return result_path
            
        except TTSWriteError:
            # Re-raise TTSWriteError without wrapping
            raise
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}", 
                        extra={'subsys': 'tts', 'event': 'generate.error'}, exc_info=True)
            raise TTSWriteError(f"Failed to generate speech: {e}")
