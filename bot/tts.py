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
    
    def _init_kokoro(self) -> None:
        """Initialize the Kokoro TTS engine with direct NPZ support."""
        try:
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
            
            # Initialize with KokoroDirect directly
            try:
                logger.info("Initializing KokoroDirect TTS engine",
                          extra={'subsys': 'tts', 'event': 'init.kokoro_direct'})
                self.kokoro = KokoroDirect(model_path, voices_path)
                
                # Get available voices
                self.voices = self.kokoro.get_voice_names() if hasattr(self.kokoro, 'get_voice_names') else []
                
                # Log success with detailed info
                if self.voices:
                    logger.info(f"TTS initialized with {len(self.voices)} voices: {', '.join(self.voices[:5])}{'...' if len(self.voices) > 5 else ''}", 
                               extra={'subsys': 'tts', 'event': 'init.success', 'voice_count': len(self.voices)})
                else:
                    logger.warning("TTS initialized but no voices available", 
                                 extra={'subsys': 'tts', 'event': 'init.no_voices'})
            except Exception as e:
                logger.error(f"Failed to initialize KokoroDirect: {e}",
                           extra={'subsys': 'tts', 'event': 'init.kokoro_error'}, exc_info=True)
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
        
        # Pass voice_id directly to KokoroDirect.create which should return a Path
        start_time = asyncio.get_event_loop().time()
        try:
            result = await asyncio.to_thread(self.kokoro.create, text, voice_id, out_path=temp_path)
            generation_time = asyncio.get_event_loop().time() - start_time
            
            # Handle potential tuple return value (audio, sample_rate) instead of Path
            if isinstance(result, tuple) and len(result) == 2:
                # Extract the components from the tuple
                first_elem, second_elem = result
                
                # Check if the first element is a Path or string
                if isinstance(first_elem, (str, Path)):
                    logger.debug("kokoro.create returned a tuple (Path, sample_rate). Using the Path component.",
                                extra={'subsys': 'tts', 'event': 'generate.tuple_path_return'})
                    wav_path = first_elem
                # Check if the first element is a NumPy ndarray (audio data)
                elif isinstance(first_elem, np.ndarray):
                    logger.debug("kokoro.create returned a tuple (ndarray, sample_rate). Writing to file.",
                                extra={'subsys': 'tts', 'event': 'generate.tuple_ndarray_return'})
                    # Get the sample rate from the second element if it's a number
                    sample_rate = second_elem if isinstance(second_elem, (int, float)) else 24000
                    
                    # Write the ndarray to the temp file
                    try:
                        from scipy.io import wavfile
                        # Normalize and convert to int16 for scipy wavfile
                        if np.max(np.abs(first_elem)) > 0:  # Avoid division by zero
                            audio_normalized = first_elem / np.max(np.abs(first_elem))
                        else:
                            audio_normalized = first_elem
                        audio_int16 = (np.clip(audio_normalized, -1.0, 1.0) * 32767).astype(np.int16)
                        wavfile.write(temp_path, sample_rate, audio_int16)
                        wav_path = temp_path
                        logger.debug(f"Wrote audio ndarray to temp file: {temp_path}",
                                    extra={'subsys': 'tts', 'event': 'generate.tuple_ndarray_written'})
                    except Exception as e:
                        logger.error(f"Failed to write audio ndarray to file: {e}",
                                    extra={'subsys': 'tts', 'event': 'generate.error.write_tuple_ndarray'})
                        raise ValueError(f"Failed to write audio ndarray to file: {e}")
                else:
                    # Unexpected first element type
                    logger.error(f"First element of tuple is not a Path, string, or ndarray: {type(first_elem)}",
                                extra={'subsys': 'tts', 'event': 'generate.error.tuple_unexpected_type'})
                    raise ValueError(f"TTS generation failed: First element of tuple is not a Path, string, or ndarray: {type(first_elem)}")
                    
            elif isinstance(result, np.ndarray):
                # Handle case where raw audio array is returned instead of Path
                logger.warning("kokoro.create returned a NumPy array instead of a Path. Writing to temp file.",
                              extra={'subsys': 'tts', 'event': 'generate.array_return'})
                # Write the array to the temp file
                try:
                    from scipy.io import wavfile
                    # Convert to int16 for scipy wavfile
                    audio_int16 = (np.clip(result, -1.0, 1.0) * 32767).astype(np.int16)
                    wavfile.write(temp_path, 24000, audio_int16)  # Assume 24kHz sample rate
                    wav_path = temp_path
                    logger.debug(f"Wrote audio array to temp file: {temp_path}",
                                extra={'subsys': 'tts', 'event': 'generate.array_written'})
                except Exception as e:
                    logger.error(f"Failed to write audio array to file: {e}",
                                extra={'subsys': 'tts', 'event': 'generate.error.write_array'})
                    # Raise exception instead of returning None
                    raise ValueError(f"Failed to write audio array to file: {e}")
            else:
                # Handle case where result might be None or other non-Path type
                if result is None:
                    logger.error("kokoro.create returned None instead of a Path",
                                extra={'subsys': 'tts', 'event': 'generate.error.none_result'})
                    raise ValueError("TTS generation failed: kokoro.create returned None instead of a Path")
                elif not isinstance(result, (str, Path)):
                    logger.error(f"kokoro.create returned unexpected type: {type(result)}",
                                extra={'subsys': 'tts', 'event': 'generate.error.unexpected_type'})
                    raise ValueError(f"TTS generation failed: kokoro.create returned unexpected type: {type(result)}")
                wav_path = result
                
            # Debug log the result type and path
            logger.debug(f"TTS result: type={type(wav_path)}, path={wav_path}",
                        extra={'subsys': 'tts', 'event': 'generate.result_path'})
            
            # Ensure wav_path is a Path object
            if wav_path is not None and not isinstance(wav_path, Path):
                try:
                    wav_path = Path(wav_path)
                    logger.debug(f"Converted wav_path to Path object: {wav_path}",
                                extra={'subsys': 'tts', 'event': 'generate.path_convert'})
                except Exception as e:
                    logger.error(f"Failed to convert wav_path to Path object: {e}",
                                extra={'subsys': 'tts', 'event': 'generate.error.path_convert'})
                    # Raise exception instead of returning None
                    raise ValueError(f"Failed to convert TTS output to Path object: {e}")
            
            # Verify the file was created and has content
            # First ensure wav_path is not None and is a Path object
            if wav_path is None:
                logger.error("TTS generated None instead of a Path", 
                           extra={'subsys': 'tts', 'event': 'generate.error.none_path'})
                
                # Create a fallback silent audio file instead of raising an exception
                try:
                    logger.warning("Creating fallback silent audio file",
                                 extra={'subsys': 'tts', 'event': 'generate.fallback_silent'})
                    # Generate 1 second of silence
                    from scipy.io import wavfile
                    sample_rate = 24000
                    silent_audio = np.zeros(sample_rate, dtype=np.int16)
                    wavfile.write(temp_path, sample_rate, silent_audio)
                    wav_path = temp_path
                    logger.debug(f"Created fallback silent audio file: {temp_path}",
                                extra={'subsys': 'tts', 'event': 'generate.fallback_silent_created'})
                except Exception as fallback_error:
                    logger.error(f"Failed to create fallback silent audio: {fallback_error}",
                                extra={'subsys': 'tts', 'event': 'generate.error.fallback_silent_failed'})
                    raise ValueError("TTS generation failed: returned None instead of a Path and fallback creation failed")
                
            # Now check if the file exists and has content
            try:
                if not isinstance(wav_path, Path):
                    logger.warning(f"wav_path is not a Path object, attempting conversion: {type(wav_path)}", 
                                 extra={'subsys': 'tts', 'event': 'generate.warning.not_path'})
                    wav_path = Path(str(wav_path))
                
                if not wav_path.exists():
                    logger.error(f"TTS output file does not exist: {wav_path}", 
                                extra={'subsys': 'tts', 'event': 'generate.error.missing_file'})
                    
                    # Create a fallback silent audio file
                    logger.warning("Creating fallback silent audio file for missing output",
                                 extra={'subsys': 'tts', 'event': 'generate.fallback_silent_missing'})
                    from scipy.io import wavfile
                    sample_rate = 24000
                    silent_audio = np.zeros(sample_rate, dtype=np.int16)
                    wavfile.write(temp_path, sample_rate, silent_audio)
                    wav_path = temp_path
                elif wav_path.stat().st_size == 0:
                    logger.error(f"TTS generated empty file: {wav_path}", 
                                extra={'subsys': 'tts', 'event': 'generate.error.empty_file'})
                    
                    # Create a fallback silent audio file
                    logger.warning("Creating fallback silent audio file for empty output",
                                 extra={'subsys': 'tts', 'event': 'generate.fallback_silent_empty'})
                    from scipy.io import wavfile
                    sample_rate = 24000
                    silent_audio = np.zeros(sample_rate, dtype=np.int16)
                    wavfile.write(temp_path, sample_rate, silent_audio)
                    wav_path = temp_path
            except Exception as e:
                logger.error(f"Error checking file existence: {e}", 
                            extra={'subsys': 'tts', 'event': 'generate.error.file_check'})
                
                # Create a fallback silent audio file
                try:
                    logger.warning("Creating fallback silent audio after file check error",
                                 extra={'subsys': 'tts', 'event': 'generate.fallback_silent_error'})
                    from scipy.io import wavfile
                    sample_rate = 24000
                    silent_audio = np.zeros(sample_rate, dtype=np.int16)
                    wavfile.write(temp_path, sample_rate, silent_audio)
                    wav_path = temp_path
                except Exception as fallback_error:
                    logger.error(f"Failed to create fallback silent audio: {fallback_error}",
                                extra={'subsys': 'tts', 'event': 'generate.error.fallback_silent_failed'})
                    raise ValueError(f"TTS generation failed: error checking file existence: {e}")
                
            
            # Try to get audio duration for logging
            try:
                import soundfile as sf
                info = sf.info(wav_path)
                audio_duration = info.duration
                sample_rate = info.samplerate
                logger.debug(f"Audio info: {audio_duration:.2f}s, {sample_rate}Hz, {info.channels} channels",
                           extra={'subsys': 'tts', 'event': 'generate.audio_info'})
            except ImportError:
                # Fallback to scipy if soundfile is not available
                try:
                    from scipy.io import wavfile
                    sample_rate, audio = wavfile.read(wav_path)
                    audio_duration = len(audio) / sample_rate
                    logger.debug(f"Audio info (scipy): {audio_duration:.2f}s, {sample_rate}Hz, shape: {audio.shape}",
                               extra={'subsys': 'tts', 'event': 'generate.audio_info_scipy'})
                except Exception as e:
                    # If we can't get audio info, just log file size
                    audio_duration = 0
                    logger.debug(f"Could not get audio info: {e}, file size: {wav_path.stat().st_size} bytes",
                               extra={'subsys': 'tts', 'event': 'generate.audio_info_fallback'})
            
            # Log performance metrics
            logger.debug(f"Generated {audio_duration:.2f}s audio in {generation_time:.2f}s ({audio_duration/generation_time:.2f}x real-time if positive)", 
                       extra={'subsys': 'tts', 'event': 'generate.complete', 
                              'audio_duration': audio_duration, 
                              'generation_time': generation_time, 
                              'speedup': audio_duration/generation_time if generation_time > 0 and audio_duration > 0 else 0})
            
            # Return the path as a Path object
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
        # Nothing to clean up for now
        logger.info("TTS manager shutting down")
