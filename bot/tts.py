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
from .kokoro_wrapper import KokoroWrapper
from .kokoro_direct import KokoroDirect

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
            # Get model and voice paths from environment variables or config
            model_path = os.environ.get("TTS_MODEL_FILE") or self.config.get("TTS_MODEL_FILE", "tts/onnx/kokoro-v1.0.onnx")
            voices_path = os.environ.get("TTS_VOICE_FILE") or self.config.get("TTS_VOICE_FILE", "tts/voices/voices-v1.0.bin")
            
            # Log paths with detailed info for debugging
            logger.debug(f"Using TTS model path: {model_path}", 
                        extra={'subsys': 'tts', 'event': 'init.paths', 'path_type': 'model'})
            logger.debug(f"Using TTS voice path: {voices_path}", 
                        extra={'subsys': 'tts', 'event': 'init.paths', 'path_type': 'voices'})
            
            # Validate paths
            model_path_obj = Path(model_path)
            voices_path_obj = Path(voices_path)
            
            if not model_path_obj.exists():
                logger.error(f"Kokoro model not found at {model_path}", 
                           extra={'subsys': 'tts', 'event': 'init.error.model_missing'})
                return
                
            if not voices_path_obj.exists():
                logger.error(f"Voices file not found at {voices_path}", 
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
            
            # Initialize with KokoroWrapper which now uses KokoroDirect internally
            self.kokoro = KokoroWrapper(model_path, voices_path)
            
            # Get available voices
            self.voices = self.kokoro.voices if hasattr(self.kokoro, 'voices') else []
            
            # Log success with detailed info
            if self.voices:
                logger.info(f"TTS initialized with {len(self.voices)} voices: {', '.join(self.voices[:5])}{'...' if len(self.voices) > 5 else ''}", 
                           extra={'subsys': 'tts', 'event': 'init.success', 'voice_count': len(self.voices)})
            else:
                logger.warning("TTS initialized but no voices available", 
                             extra={'subsys': 'tts', 'event': 'init.no_voices'})
                
        except ImportError as e:
            logger.error(f"Failed to import required modules for TTS: {e}", 
                       extra={'subsys': 'tts', 'event': 'init.import_error'}, exc_info=True)
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}", 
                       extra={'subsys': 'tts', 'event': 'init.error'}, exc_info=True)
            
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
            return None
            
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
            
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
                return None
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
            return None
    
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
    
    async def generate_tts(self, text: str, voice_id: str = None) -> Optional[str]:
        """Generate TTS audio and return the path to the audio file.
        
        Args:
            text: The text to synthesize
            voice_id: The voice ID to use (optional)
            
        Returns:
            Path to the generated audio file, or None if synthesis failed
        """
        if not self.kokoro:
            logger.error("TTS not initialized", 
                       extra={'subsys': 'tts', 'event': 'generate.error.not_initialized'})
            return None
            
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty text provided for TTS", 
                         extra={'subsys': 'tts', 'event': 'generate.error.empty_text'})
            return None
            
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
                return None
                
            # Fall back to first available voice
            old_voice_id = voice_id
            voice_id = self.voices[0]
            logger.warning(f"Voice '{old_voice_id}' not found, falling back to '{voice_id}'", 
                         extra={'subsys': 'tts', 'event': 'generate.fallback_voice', 'requested': old_voice_id, 'using': voice_id})
        
        try:
            # Generate audio using KokoroDirect via KokoroWrapper
            # KokoroDirect now handles both voice IDs and embeddings directly
            logger.debug(f"Synthesizing text with voice '{voice_id}': '{text[:50]}{'...' if len(text) > 50 else ''}''", 
                       extra={'subsys': 'tts', 'event': 'generate.start', 'voice_id': voice_id, 'text_length': len(text)})
            
            # Pass voice_id directly - KokoroDirect will handle the lookup
            start_time = asyncio.get_event_loop().time()
            audio, sample_rate = await asyncio.to_thread(self.kokoro.create, text, voice_id)
            generation_time = asyncio.get_event_loop().time() - start_time
            
            # Log performance metrics
            audio_duration = len(audio) / sample_rate if audio is not None else 0
            logger.debug(f"Generated {audio_duration:.2f}s audio in {generation_time:.2f}s ({audio_duration/generation_time:.2f}x real-time)", 
                       extra={'subsys': 'tts', 'event': 'generate.complete', 
                              'audio_duration': audio_duration, 
                              'generation_time': generation_time, 
                              'speedup': audio_duration/generation_time if generation_time > 0 else 0})
            
            if audio is None or len(audio) == 0:
                logger.error("Generated empty audio", 
                           extra={'subsys': 'tts', 'event': 'generate.error.empty_audio'})
                return None
            
            # Create a temporary file with a unique name in the temp directory
            os.makedirs(self.temp_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.temp_dir) as temp_file:
                temp_path = temp_file.name
                
            # Ensure audio is not empty and has valid data
            if audio is None or audio.size == 0:
                logger.error("Cannot save empty audio", 
                           extra={'subsys': 'tts', 'event': 'generate.error.empty_audio'})
                return None
                
            # Log audio info for debugging
            logger.debug(f"Audio shape: {audio.shape}, dtype: {audio.dtype}, min: {np.min(audio)}, max: {np.max(audio)}",
                       extra={'subsys': 'tts', 'event': 'generate.audio_info'})
                   
            # Ensure audio is a 1D array (WAV files expect 1D arrays)
            if len(audio.shape) > 1:
                # Reshape to 1D if it's a 2D array with shape like (1, N)
                audio = audio.reshape(-1)
                logger.debug(f"Reshaped audio to 1D: {audio.shape}",
                           extra={'subsys': 'tts', 'event': 'generate.reshape'})
            
            # Save as WAV using soundfile (more reliable for float32 audio)
            try:
                import soundfile as sf
                # Ensure audio is float32 and in range [-1, 1]
                if not np.issubdtype(audio.dtype, np.floating):
                    audio = audio.astype(np.float32)
                    if np.max(np.abs(audio)) > 1.0:
                        audio = audio / 32767.0  # Convert from int16 scale if needed
                        
                # Write as WAV file with subtype 'FLOAT' for float32 data
                sf.write(temp_path, audio, sample_rate, subtype='FLOAT')
                logger.debug(f"Saved audio to {temp_path} using soundfile (FLOAT subtype)", 
                           extra={'subsys': 'tts', 'event': 'generate.save', 'path': temp_path, 'library': 'soundfile'})
            except Exception as e:
                logger.warning(f"Failed to save with soundfile: {e}, trying scipy", 
                             extra={'subsys': 'tts', 'event': 'generate.fallback_scipy'})
                # Fallback to scipy if soundfile fails
                try:
                    from scipy.io import wavfile
                    # Convert to int16 for scipy wavfile
                    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                    wavfile.write(temp_path, sample_rate, audio_int16)
                    logger.debug(f"Saved audio to {temp_path} using scipy (int16)", 
                               extra={'subsys': 'tts', 'event': 'generate.save', 'path': temp_path, 'library': 'scipy'})
                except Exception as e:
                    logger.error(f"Failed to save audio with both libraries: {e}", 
                               extra={'subsys': 'tts', 'event': 'generate.error.save_failed'}, exc_info=True)
                    return None
                    
            # Verify the file was created and has content
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                logger.error(f"Failed to create audio file or file is empty: {temp_path}", 
                           extra={'subsys': 'tts', 'event': 'generate.error.file_creation'})
                return None
                
            # Schedule file deletion
            asyncio.create_task(self._cleanup_file(temp_path))
            
            return temp_path
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", 
                       extra={'subsys': 'tts', 'event': 'generate.error'}, exc_info=True)
            return None
    
    async def close(self) -> None:
        """Clean up resources when shutting down."""
        # Nothing to clean up for now
        logger.info("TTS manager shutting down")
