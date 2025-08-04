"""Text-to-Speech manager for the Discord bot."""

import asyncio
import io
import logging
import os
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union, List

import discord
import numpy as np
from discord.ext import commands

from .config import load_config
from .kokoro_direct_fixed import KokoroDirect
from .tokenizer_registry import TokenizerRegistry, discover_tokenizers, is_tokenizer_warning_needed, get_tokenizer_warning_message
from .tts_errors import MissingTokeniserError, TTSWriteError

logger = logging.getLogger(__name__)


class TTSManager:
    """Manages Text-to-Speech functionality for the bot."""
    
    def __init__(self, bot: commands.Bot):
        """Initialize the TTS manager.
        
        Args:
            bot: The bot instance
        """
        self.bot = bot
        self.config = load_config()
        self.kokoro = None
        self.voices = []
        self.voice_cache = {}
        self._default_voice = None
        self.temp_dir = Path(self.config.get("TEMP_DIR", "temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kokoro TTS
        self._init_kokoro()
    
    def _init_tokenizer_registry(self) -> None:
        """Initialize the tokenizer registry to ensure tokenizers are available for TTS."""
        try:
            # Get the tokenizer registry instance
            registry = TokenizerRegistry.get_instance()
            
            # Discover available tokenizers
            available_tokenizers = registry.discover_tokenizers()
            
            if not available_tokenizers:
                logger.warning("No tokenizers found during initialization. TTS may not work correctly.",
                             extra={'subsys': 'tts', 'event': 'init.no_tokenizers'})
            else:
                logger.info(f"Found {len(available_tokenizers)} available tokenizers: {', '.join(sorted(available_tokenizers))}",
                          extra={'subsys': 'tts', 'event': 'init.tokenizers_found'})
                
            # Check if we need to warn about missing tokenizers for English
            language = os.environ.get("TTS_LANGUAGE", "en").lower()
            if language == "en" and is_tokenizer_warning_needed(language):
                warning_message = get_tokenizer_warning_message(language)
                logger.warning(warning_message, extra={'subsys': 'tts', 'event': 'init.tokenizer_warning'})
                
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer registry: {e}",
                       extra={'subsys': 'tts', 'event': 'init.tokenizer_registry_error'}, exc_info=True)
            # Continue without failing - we'll use fallback tokenizers
    
    def _init_kokoro(self) -> None:
        """Initialize the Kokoro TTS engine with direct NPZ support."""
        try:
            # Initialize tokenizer registry first to ensure it's available for TTS
            self._init_tokenizer_registry()
            
            # Check for new environment variables first (preferred)
            model_path = os.environ.get("TTS_MODEL_PATH")
            voices_path = os.environ.get("TTS_VOICES_PATH")
            
            # Fall back to old environment variables if new ones aren't set
            if not model_path:
                model_path = os.environ.get("TTS_MODEL_FILE")
                if model_path:
                    logger.warning("Using deprecated TTS_MODEL_FILE environment variable. Please use TTS_MODEL_PATH instead.",
                                 extra={'subsys': 'tts', 'event': 'init.deprecated_env_var', 'var': 'TTS_MODEL_FILE'})
            
            if not voices_path:
                voices_path = os.environ.get("TTS_VOICE_FILE")
                if voices_path:
                    logger.warning("Using deprecated TTS_VOICE_FILE environment variable. Please use TTS_VOICES_PATH instead.",
                                 extra={'subsys': 'tts', 'event': 'init.deprecated_env_var', 'var': 'TTS_VOICE_FILE'})
            
            # Fall back to config values if environment variables aren't set
            if not model_path:
                model_path = self.config.get("tts", {}).get("model_path") or self.config.get("TTS_MODEL_FILE", "tts/onnx/kokoro-v1.0.onnx")
            
            if not voices_path:
                voices_path = self.config.get("tts", {}).get("voices_path") or self.config.get("TTS_VOICE_FILE", "tts/voices/voices-v1.0.bin")
            
            # Log paths with detailed info for debugging
            logger.debug(f"Using TTS model path: {model_path}", 
                        extra={'subsys': 'tts', 'event': 'init.paths', 'path_type': 'model'})
            logger.debug(f"Using TTS voice path: {voices_path}", 
                        extra={'subsys': 'tts', 'event': 'init.paths', 'path_type': 'voices'})
            
            # Validate paths
            model_path_obj = Path(model_path)
            voices_path_obj = Path(voices_path)
            
            # Try alternative paths if files don't exist
            if not model_path_obj.exists():
                # Try common alternative locations
                alt_paths = [
                    Path("models/kokoro.onnx"),
                    Path("tts/onnx/model.onnx"),
                    Path("tts/onnx/kokoro-v1.0.onnx"),
                    Path("tts/model.onnx")
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        logger.warning(f"Model not found at {model_path}, using alternative path: {alt_path}",
                                     extra={'subsys': 'tts', 'event': 'init.alt_path', 'path_type': 'model'})
                        model_path = str(alt_path)
                        model_path_obj = alt_path
                        break
            
            if not model_path_obj.exists():
                logger.error(f"Kokoro model not found at {model_path} or any alternative paths", 
                           extra={'subsys': 'tts', 'event': 'init.error.model_missing'})
                return
            
            # Try alternative paths for voices file
            if not voices_path_obj.exists():
                # Try common alternative locations
                alt_paths = [
                    Path("models/voices.npz"),
                    Path("tts/voices/voices.npz"),
                    Path("tts/voices/voices-v1.0.bin"),
                    Path("tts/voices.npz")
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        logger.warning(f"Voices not found at {voices_path}, using alternative path: {alt_path}",
                                     extra={'subsys': 'tts', 'event': 'init.alt_path', 'path_type': 'voices'})
                        voices_path = str(alt_path)
                        voices_path_obj = alt_path
                        break
            
            if not voices_path_obj.exists():
                logger.error(f"Voices file not found at {voices_path} or any alternative paths", 
                           extra={'subsys': 'tts', 'event': 'init.error.voices_missing'})
                return
            
            # Log file info for debugging
            logger.debug(f"Model file size: {model_path_obj.stat().st_size} bytes", 
                        extra={'subsys': 'tts', 'event': 'init.file_info', 'file_type': 'model'})
            logger.debug(f"Voices file size: {voices_path_obj.stat().st_size} bytes", 
                        extra={'subsys': 'tts', 'event': 'init.file_info', 'file_type': 'voices'})
            
            # Calculate SHA-256 of voice file for verification
            try:
                with open(voices_path_obj, 'rb') as f:
                    voice_file_hash = hashlib.sha256(f.read()).hexdigest()
                    logger.debug(f"Voices file SHA-256: {voice_file_hash[:8]}...", 
                                extra={'subsys': 'tts', 'event': 'init.file_hash', 'file_type': 'voices'})
            except Exception as e:
                logger.warning(f"Could not calculate voice file hash: {e}", 
                             extra={'subsys': 'tts', 'event': 'init.hash_error'})
            
            # Initialize Kokoro with the model and voices paths
            try:
                logger.debug("Initializing KokoroDirect TTS engine", 
                           extra={'subsys': 'tts', 'event': 'init.kokoro_start'})
                self.kokoro = KokoroDirect(model_path, voices_path)
                logger.info("KokoroDirect TTS engine initialized successfully", 
                          extra={'subsys': 'tts', 'event': 'init.kokoro_success'})
                
                # Check if tokenizer warning is needed and log it
                if is_tokenizer_warning_needed():
                    language = os.environ.get('TTS_LANGUAGE', 'en')
                    warning_msg = get_tokenizer_warning_message(language)
                    logger.warning(f"Tokenizer warning: {warning_msg.splitlines()[0]}", 
                                 extra={'subsys': 'tts', 'event': 'init.tokenizer_warning'})
                    
                    # Store warning message for potential display to user
                    self.tokenizer_warning = warning_msg
                else:
                    self.tokenizer_warning = None
                
                # Get available voices
                self.voices = self.kokoro.get_voice_names()
                logger.info(f"Loaded {len(self.voices)} voices", 
                          extra={'subsys': 'tts', 'event': 'init.voices_loaded', 'voice_count': len(self.voices)})
                
                # Set default voice
                default_voice = os.environ.get("TTS_VOICE", "default")
                if default_voice not in self.voices and default_voice != "default":
                    logger.warning(f"Default voice '{default_voice}' not found, using first available voice",
                                 extra={'subsys': 'tts', 'event': 'init.default_voice_not_found'})
                    default_voice = self.voices[0] if self.voices else "default"
                
                self._default_voice = default_voice
                logger.info(f"Using default voice: {self._default_voice}", 
                          extra={'subsys': 'tts', 'event': 'init.default_voice_set'})
                
            except Exception as e:
                logger.error(f"Failed to initialize KokoroDirect TTS engine: {e}", 
                           extra={'subsys': 'tts', 'event': 'init.kokoro_error'}, exc_info=True)
                self.kokoro = None
                raise
                
        except ImportError as e:
            logger.error(f"Failed to import required modules for TTS: {e}", 
                       extra={'subsys': 'tts', 'event': 'init.import_error'}, exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}", 
                       extra={'subsys': 'tts', 'event': 'init.error'}, exc_info=True)
            raise
            
    def is_available(self) -> bool:
        """Check if TTS is available for use.
        
        Returns:
            bool: True if TTS is available, False otherwise
        """
        # TTS is available if we have a kokoro instance and at least one voice
        is_available = self.kokoro is not None and len(self.voices) > 0
        logger.debug(f"TTS availability check: {is_available}", extra={'subsys': 'tts', 'event': 'check_available'})
        return is_available
        
    @property
    def voice(self) -> str:
        """Get the default voice ID to use for TTS.
        
        Returns:
            str: The default voice ID, or the first available voice if no default is set
        """
        # If we have a default voice and it's in the available voices, use it
        if self._default_voice and self._default_voice in self.voices:
            return self._default_voice
            
        # Otherwise, use the first available voice
        if self.voices:
            # Set this as the default for future use
            self._default_voice = self.voices[0]
            return self._default_voice
            
        # No voices available
        logger.warning("No voices available for TTS", extra={'subsys': 'tts', 'event': 'voice.none_available'})
        return None
    
    async def synthesize(self, text: str, voice_id: str = None) -> Optional[discord.File]:
        """Synthesize text to speech and return as a Discord file.
        
        Args:
            text: The text to synthesize
            voice_id: The voice ID to use (optional)
            
        Returns:
            A Discord file containing the audio, or None if synthesis failed
        """
        if not self.kokoro:
            logger.error("TTS not initialized")
            # Raise exception instead of returning None
            raise ValueError("TTS generation failed: TTS not initialized")
            
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty text provided for TTS")
            # Raise exception instead of returning None
            raise ValueError("TTS generation failed: Empty text provided")
            
        # Truncate text if it's too long
        max_length = self.config.get("TTS_MAX_LENGTH", 500)
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        # Use default voice if none specified
        if not voice_id and self.voices:
            voice_id = self.voices[0]
            
        if not voice_id or voice_id not in self.voices:
            if not self.voices:
                logger.error("No voices available for TTS")
                # Raise exception instead of returning None
                raise ValueError("TTS generation failed: No voices available")
            voice_id = self.voices[0]
            logger.warning(f"Voice {voice_id} not found, using {voice_id} instead")
        
        try:
            # Generate audio
            logger.debug(f"Synthesizing text with voice {voice_id}: {text[:50]}...")
            audio, sample_rate = await asyncio.to_thread(self.kokoro.create, text, voice_id)
            
            # Convert to WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.temp_dir) as temp_file:
                temp_path = temp_file.name
                
            # Save as WAV using scipy
            try:
                from scipy.io import wavfile
                wavfile.write(temp_path, sample_rate, audio)
            except ImportError:
                # Fallback to soundfile if scipy not available
                import soundfile as sf
                sf.write(temp_path, audio, sample_rate)
                
            # Create Discord file
            discord_file = discord.File(temp_path, filename="tts_output.wav")
            
            # Schedule file deletion
            asyncio.create_task(self._cleanup_file(temp_path))
            
            return discord_file
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            # Re-raise the exception with more context
            raise ValueError(f"TTS generation failed: {e}")
    
    async def _cleanup_file(self, file_path: str, delay: float = 60.0) -> None:
        """Clean up a temporary file after a delay.
        
        Args:
            file_path: Path to the file to delete
            delay: Delay in seconds before deletion
        """
        try:
            await asyncio.sleep(delay)
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {e}")
    
    async def generate_tts(self, text: str, voice_id: str = None) -> Path:
        """Generate TTS audio and return the path to the audio file.
        
        Args:
            text: The text to synthesize
            voice_id: The voice ID to use (optional)
            
        Returns:
            Path to the generated audio file
            
        Raises:
            RuntimeError: If TTS is not available
        """
        # Check if TTS is available using the is_available method
        if not self.is_available():
            logger.error("TTS not available", 
                       extra={'subsys': 'tts', 'event': 'generate.error.not_available'})
            raise RuntimeError("TTS system is not available")
        
        # Additional check for kokoro initialization (redundant but kept for backward compatibility)
        if not self.kokoro:
            logger.error("TTS not initialized", 
                       extra={'subsys': 'tts', 'event': 'generate.error.not_initialized'})
            raise ValueError("TTS system is not initialized")
            
        # Truncate text if it's too long
        max_length = self.config.get("TTS_MAX_LENGTH", 500)
        original_length = len(text)
        if original_length > max_length:
            text = text[:max_length] + "..."
            logger.debug(f"Truncated text from {original_length} to {max_length} characters", 
                       extra={'subsys': 'tts', 'event': 'generate.truncate', 'original_length': original_length, 'new_length': len(text)})
            
        # Use default voice if none specified
        if not voice_id:
            voice_id = self.voice
            logger.debug(f"No voice specified, using default voice: {voice_id}", 
                       extra={'subsys': 'tts', 'event': 'generate.default_voice'})
            
        if not voice_id or voice_id not in self.voices:
            if not self.voices:
                logger.error("No voices available for TTS", 
                           extra={'subsys': 'tts', 'event': 'generate.error.no_voices'})
                raise ValueError("No voices available for TTS synthesis")
                
            # Fall back to first available voice
            old_voice_id = voice_id
            voice_id = self.voices[0]
            logger.warning(f"Voice '{old_voice_id}' not found, falling back to '{voice_id}'", 
                         extra={'subsys': 'tts', 'event': 'generate.fallback_voice', 'requested': old_voice_id, 'using': voice_id})
        
        # Generate audio using KokoroDirect via KokoroWrapper
        # KokoroDirect now handles both voice IDs and embeddings directly
        logger.debug(f"Synthesizing text with voice '{voice_id}': '{text[:50]}{'...' if len(text) > 50 else ''}''", 
                   extra={'subsys': 'tts', 'event': 'generate.start', 'voice_id': voice_id, 'text_length': len(text)})
        
        # Create a temporary file with a unique name in the temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.temp_dir) as temp_file:
            temp_path = Path(temp_file.name)
        
        # Pass voice_id directly to KokoroDirect.create which returns a Path
        start_time = asyncio.get_event_loop().time()
        try:
            wav_path = await asyncio.to_thread(self.kokoro.create, text, voice_id, out_path=temp_path)
            generation_time = asyncio.get_event_loop().time() - start_time
            
            # Validate the return type is Path
            if not isinstance(wav_path, Path):
                logger.error("KokoroDirect.create returned unexpected type: {type(wav_path)}",
                           extra={'subsys': 'tts', 'event': 'generate.error.invalid_return_type'})
                raise TypeError(f"Expected Path, got {type(wav_path)}")
            
            # Verify the file exists and is not empty
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                logger.error("Generated audio file is missing or empty",
                           extra={'subsys': 'tts', 'event': 'generate.error.empty_file', 'path': str(wav_path)})
                raise TTSWriteError("Generated audio file is missing or empty")
            
            logger.info(f"Generated TTS audio in {generation_time:.2f}s: {wav_path}",
                      extra={'subsys': 'tts', 'event': 'generate.success', 'duration': generation_time, 'path': str(wav_path)})
            return wav_path
        except Exception as e:
            generation_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"TTS generation failed after {generation_time:.2f}s: {e}", 
                       extra={'subsys': 'tts', 'event': 'generate.error'}, exc_info=True)
            
            # Clean up the temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temp file {temp_path} after error",
                               extra={'subsys': 'tts', 'event': 'generate.cleanup'})
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}",
                                 extra={'subsys': 'tts', 'event': 'generate.cleanup_error'})
            
            # Re-raise the exception with more context
            raise ValueError(f"TTS generation failed: {e}")
    
    async def close(self) -> None:
        """Clean up resources when shutting down."""
        try:
            logger.debug("[TTS] Closing TTS manager")
            
            # Clean up Kokoro resources
            if self.kokoro:
                try:
                    # Clear voice cache to free memory
                    self.voice_cache.clear()
                    
                    # Clear references
                    self.kokoro = None
                    self.voices = []
                    self._default_voice = None
                    
                    logger.debug("[TTS] Kokoro resources cleared")
                except Exception as e:
                    logger.warning(f"[TTS] Error cleaning up Kokoro: {e}")
            
            # Clean up temporary files
            try:
                if self.temp_dir.exists():
                    import shutil
                    temp_files = list(self.temp_dir.glob("*.wav"))
                    if temp_files:
                        logger.debug(f"[TTS] Cleaning up {len(temp_files)} temporary audio files")
                        for temp_file in temp_files:
                            try:
                                temp_file.unlink()
                            except Exception:
                                pass  # Ignore individual file cleanup errors
            except Exception as e:
                logger.warning(f"[TTS] Error cleaning up temp files: {e}")
            
            logger.info("[TTS] ✔ TTS manager closed successfully")
            
        except Exception as e:
            logger.warning(f"[TTS] Error during TTS manager shutdown: {e}")
