"""
TTSManager implementation with robust error handling and proper audio file validation.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Import custom exceptions
from .tts_errors import TTSWriteError

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
        self.backend = os.environ.get('TTS_BACKEND', 'kokoro-onnx')
        self.voice = os.environ.get('TTS_VOICE', 'default')
        self.available = False
        self.kokoro = None
        self.cache_dir = Path(os.environ.get('XDG_CACHE_HOME', 'tts/cache'))
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if TTS is available
        self._check_availability()
        
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
            # Use our fixed KokoroDirect implementation
            from .kokoro_direct_fixed import KokoroDirect
            
            # Get model and voices paths from config or environment
            model_path = os.environ.get('TTS_MODEL_PATH', self.config.get('tts', {}).get('model_path', 'models/kokoro.onnx'))
            voices_path = os.environ.get('TTS_VOICES_PATH', self.config.get('tts', {}).get('voices_path', 'models/voices.npz'))
            
            # Log model and voices paths
            logger.debug(f"Loading TTS model from {model_path} and voices from {voices_path}", 
                        extra={'subsys': 'tts', 'event': 'load.paths'})
            
            # Initialize KokoroDirect
            self.kokoro = KokoroDirect(model_path, voices_path)
            
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
