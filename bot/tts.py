"""
TTS client for Kokoro-ONNX TTS microservice.
"""
import hashlib
import logging
import time
from pathlib import Path
from typing import Union
import numpy as np

# Import TTS utilities for automatic model downloads
from .tts_utils import ensure_tts_files

# Configuration constants [CMV]
MAX_CACHE_AGE_DAYS = 7

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TTSManager:
    """Text-to-Speech manager using kokoro-onnx package. [PA][CA][IV]"""
    
    def __init__(self, bot: "LLMBot"):
        """Initialize TTS manager with lightweight kokoro-onnx implementation. [PA][CA]"""
        self.bot = bot
        self.logger = bot.logger
        self.config = bot.config
        self.backend = self.config.get("TTS_BACKEND", "kokoro-onnx")
        self.voice = self.config.get("TTS_VOICE", "am_michael")
        self.cache_dir = Path(self.config.get("TTS_CACHE_DIR", "tts_cache"))
        self.cache_dir.mkdir(exist_ok=True)

        # Auto-download model files if missing
        self._ensure_models_exist()

        self.tts_instance = None
        self.load_tts_instance()
        
        # TTS engine state
        self.engine = None
        self.g2p = None
        self.available = False
        
        # Initialize backend [CA]
        if self.backend == "kokoro-onnx":
            try:
                from .kokoro_direct import KokoroDirect # Use direct implementation
                
                # Verify model and voice directories exist [IV]
                onnx_dir = Path(self.config.get("TTS_ONNX_DIR", "tts/onnx"))
                voices_dir = Path(self.config.get("TTS_VOICES_DIR", "tts/voices"))

                if not onnx_dir.exists() or not onnx_dir.is_dir():
                    raise FileNotFoundError(f"ONNX model directory not found: {onnx_dir}")
                if not voices_dir.exists() or not voices_dir.is_dir():
                    raise FileNotFoundError(f"Voices directory not found: {voices_dir}")

                logger.debug(f"✅ Model and voice directories verified: {onnx_dir}, {voices_dir}")

                # Initialize G2P with fallback [REH]
                try:
                    from misaki import en, espeak
                    fallback = espeak.EspeakFallback(british=False)
                    self.g2p = en.G2P(trf=False, british=False, fallback=fallback)
                    logger.debug("✅ Misaki G2P initialized with espeak fallback")
                except Exception as g2p_error:
                    logger.error(f"Failed to initialize G2P: {g2p_error}")
                    raise

                # Initialize Kokoro engine using KokoroDirect [CA]
                try:
                    self.engine = KokoroDirect(onnx_dir=str(onnx_dir), voices_dir=str(voices_dir))
                    logger.debug("✅ KokoroDirect engine initialized")
                    
                    # Verify voices are available [IV]
                    if hasattr(self.engine, 'available_voices') and self.engine.available_voices:
                        available_voices = self.engine.available_voices
                        logger.info(f"Available voices ({len(available_voices)}): {available_voices[:5]}{'...' if len(available_voices) > 5 else ''}")
                        
                        # Validate requested voice [IV]
                        if self.voice in available_voices:
                            logger.info(f"Using requested voice: {self.voice}")
                        else:
                            # Use first available voice as fallback [REH]
                            self.voice = available_voices[0]
                            logger.warning(f"Requested voice '{config.get('TTS_VOICE')}' not found, using fallback: {self.voice}")
                    else:
                        raise RuntimeError("No voices found in Kokoro engine")
                    
                    self.available = True
                    logger.info("✅ Kokoro-ONNX TTS initialized successfully")
                    
                except Exception as kokoro_error:
                    logger.error(f"Failed to initialize Kokoro engine: {kokoro_error}")
                    raise
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize Kokoro-ONNX: {e}")
                self.available = False
        else:
            logger.error(f"❌ Unsupported TTS backend: {self.backend}")
            self.available = False
    
    def is_available(self) -> bool:
        """Check if TTS backend is available and ready. [CA]"""
        return self.available and self.engine is not None
    
    def synthesize(self, text: str, output_path: Path, voice: Union[str, np.ndarray] = None):
        """Generate speech from text using Kokoro-ONNX. [PA][CA][REH]"""
        # Log entry to method
        logger.debug(f"Entering synthesize method with text: {text[:50]}...")
        
        if voice is None:
            voice = self.voice
        
        if not self.engine:
            raise RuntimeError("TTS engine not initialized")
            
        try:
            # Debug logging for key operations [CDiP]
            logger.debug(f"Starting TTS synthesis: text='{text[:50]}...', voice='{voice}', output='{output_path}'")
            
            # Generate audio samples using correct kokoro-onnx API [PA][REH]
            try:
                # DEBUG: Log voice parameter details to identify numpy.float32 bug [REH]
                logger.error(f"DEBUG TTS: voice type={type(voice)}, voice value={repr(voice)}")
                if hasattr(voice, 'dtype'):
                    logger.error(f"DEBUG TTS: voice dtype={voice.dtype}, shape={getattr(voice, 'shape', 'N/A')}")
                
                # Resolve voice string to vector if needed [IV]
                if isinstance(voice, str):
                    logger.debug(f"Voice before resolving: {voice} (type: {type(voice)})")
                    try:
                        voice = self.engine.get_voice_style(voice)
                        logger.debug(f"Voice after resolving: type={type(voice)}, shape={voice.shape if hasattr(voice, 'shape') else 'unknown'}")
                    except Exception as e:
                        logger.error(f"Error getting voice style: {e}")
                        # Create a default 256-dimension vector as fallback
                        logger.warning("Using default 256-dim voice vector as fallback")
                        voice = np.zeros((1, 256), dtype=np.float32)
                
                # Debug voice value and type
                logger.debug(f"Voice value type: {type(voice)}")
                logger.debug(f"Voice is numpy array: {isinstance(voice, np.ndarray)}")
                
                # Ensure voice is a numpy array
                if not isinstance(voice, np.ndarray):
                    logger.warning("Converting non-numpy voice to numpy array")
                    try:
                        voice = np.array(voice, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"Error converting voice to numpy array: {e}")
                        voice = np.zeros((1, 256), dtype=np.float32)
                
                if isinstance(voice, np.ndarray):
                    logger.debug(f"Voice numpy array shape: {voice.shape}, ndim: {voice.ndim}, dtype: {voice.dtype}")
                    
                    # Process the voice vector to ensure correct dimensions
                    original_shape = voice.shape
                    original_ndim = voice.ndim
                    logger.debug(f"Original voice shape: {original_shape}, ndim: {original_ndim}")
                    
                    # First flatten the array completely
                    voice = voice.flatten()
                    
                # With KokoroDirect, we pass the voice_id (string) directly
                samples_tensor, sample_rate = self.engine.create(
                    text,
                    voice_id=self.voice,
                    speed=1.0
                )
                
                # Handle both PyTorch tensors and NumPy arrays
                if hasattr(samples_tensor, 'detach'):
                    # It's a PyTorch tensor
                    logger.debug("Converting PyTorch tensor to NumPy array")
                    samples = samples_tensor.detach().cpu().numpy()
                else:
                    # It's already a NumPy array
                    logger.debug(f"Using NumPy array directly, type: {type(samples_tensor)}")
                    samples = samples_tensor
                
                logger.debug(f"Audio generated: {len(samples)} samples at {sample_rate}Hz")
            except Exception as e:
                logger.error(f"Failed to generate audio: {e}")
                raise
            
            # Save to output file [RM]
            try:
                import soundfile as sf
                sf.write(str(output_path), samples, sample_rate)
                file_size = output_path.stat().st_size
                logger.debug(f"Audio saved: {output_path} ({file_size} bytes)")
            except Exception as e:
                logger.error(f"Failed to save audio file: {e}")
                raise
                
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def _setup_cache(self):
        """Ensure cache directory exists [RM]"""
        self.cache_dir.mkdir(exist_ok=True)
        
    def purge_old_cache(self):
        """
        Remove cache files older than MAX_CACHE_AGE_DAYS.
        This is called automatically during maintenance tasks.
        """
        current_time = time.time()
        cutoff_time = current_time - (MAX_CACHE_AGE_DAYS * 24 * 60 * 60)
        
        for file in self.cache_dir.iterdir():
            if file.is_file() and file.stat().st_mtime < cutoff_time:
                try:
                    file.unlink()
                    logger.info(f"Purged old TTS cache file: {file.name}")
                except Exception as e:
                    logger.error(f"Error purging cache file {file.name}: {e}")
                    
    def get_cache_stats(self):
        """Get cache statistics including file count and total size."""
        file_count = 0
        total_size = 0  # in bytes
        
        for file in self.cache_dir.iterdir():
            if file.is_file():
                file_count += 1
                total_size += file.stat().st_size
                
        return {
            'files': file_count,
            'size_mb': total_size / (1024 * 1024),  # convert to MB
            'cache_dir': str(self.cache_dir)
        }
    
    async def generate_tts(self, text: str, voice_id: str = "default") -> Path:
        """Generate TTS audio for the given text and return the file path. [PA][CA][REH]"""
        if not self.is_available():
            raise RuntimeError("TTS service is not available")
            
        # Generate unique filename based on text and voice [CMV]
        filename = f"{voice_id}_{hashlib.md5(text.encode()).hexdigest()}.wav"
        file_path = self.cache_dir / filename
        
        # Check cache first [PA]
        if file_path.exists():
            logger.debug(f"Using cached TTS: {file_path}")
            return file_path
            
        # Generate TTS using synchronous method [CA]
        try:
            self.synthesize(text, file_path)
            logger.debug(f"TTS generated: {file_path} ({file_path.stat().st_size} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
