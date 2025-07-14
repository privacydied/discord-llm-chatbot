"""
TTS client for Kokoro-ONNX TTS microservice.
"""
import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Union, Optional

# Configuration constants [CMV]
MAX_CACHE_AGE_DAYS = 7

class TTSManager:
    """Text-to-Speech manager using kokoro-onnx package. [PA][CA][IV]"""
    
    def __init__(self, config: dict):
        """Initialize TTS manager with lightweight kokoro-onnx implementation. [PA][CA]"""
        self.config = config
        self.backend = config.get("TTS_BACKEND", "kokoro-onnx")
        self.voice = config.get("TTS_VOICE", "am_michael")
        self.cache_dir = Path(config.get("TTS_CACHE_DIR", "tts_cache"))
        self.cache_dir.mkdir(exist_ok=True)
        
        # TTS engine state
        self.engine = None
        self.g2p = None
        self.available = False
        
        # Initialize backend [CA]
        if self.backend == "kokoro-onnx":
            try:
                # Check if kokoro_onnx is available [IV]
                try:
                    import kokoro_onnx
                    logging.debug("✅ kokoro_onnx package imported successfully")
                except ImportError as e:
                    raise RuntimeError(f"❌ Kokoro-ONNX not installed. Please install it with 'uv pip install kokoro-onnx soundfile misaki[en]': {e}")
                
                # Verify model files exist [IV]
                model_path = Path(config.get("TTS_MODEL_FILE", "tts/kokoro-v1.0.onnx"))
                voices_path = Path(config.get("TTS_VOICE_FILE", "tts/voices.json"))
                
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                if not voices_path.exists():
                    raise FileNotFoundError(f"Voices file not found: {voices_path}")
                
                # Validate model file size [IV]
                if model_path.stat().st_size < 1000000:  # Should be > 1MB
                    raise ValueError(f"Model file too small (corrupted): {model_path} ({model_path.stat().st_size} bytes)")
                
                logging.debug(f"✅ Model files verified: {model_path} ({model_path.stat().st_size} bytes), {voices_path}")
                
                # Initialize G2P with fallback [REH]
                try:
                    from misaki import en, espeak
                    fallback = espeak.EspeakFallback(british=False)
                    self.g2p = en.G2P(trf=False, british=False, fallback=fallback)
                    logging.debug("✅ Misaki G2P initialized with espeak fallback")
                except Exception as g2p_error:
                    logging.error(f"Failed to initialize G2P: {g2p_error}")
                    raise
                
                # Initialize Kokoro engine [CA]
                try:
                    from kokoro_onnx import Kokoro
                    self.engine = Kokoro(str(model_path), str(voices_path))
                    logging.debug(f"✅ Kokoro engine initialized with correct API")
                    
                    # Verify voices are available [IV]
                    if hasattr(self.engine, 'voices') and self.engine.voices:
                        available_voices = list(self.engine.voices)
                        logging.info(f"Available voices ({len(available_voices)}): {available_voices[:5]}{'...' if len(available_voices) > 5 else ''}")
                        
                        # Validate requested voice [IV]
                        if self.voice in available_voices:
                            logging.info(f"Using requested voice: {self.voice}")
                        else:
                            # Use first available voice as fallback [REH]
                            self.voice = available_voices[0]
                            logging.warning(f"Requested voice '{config.get('TTS_VOICE')}' not found, using fallback: {self.voice}")
                    else:
                        raise RuntimeError("No voices found in Kokoro engine")
                    
                    self.available = True
                    logging.info("✅ Kokoro-ONNX TTS initialized successfully")
                    
                except Exception as kokoro_error:
                    logging.error(f"Failed to initialize Kokoro engine: {kokoro_error}")
                    raise
                    
            except Exception as e:
                logging.error(f"❌ Failed to initialize Kokoro-ONNX: {e}")
                self.available = False
        else:
            logging.error(f"❌ Unsupported TTS backend: {self.backend}")
            self.available = False
    
    def is_available(self) -> bool:
        """Check if TTS backend is available and ready. [CA]"""
        return self.available and self.engine is not None
    
    def synthesize(self, text: str, output_path: Path) -> None:
        """Generate speech from text using Kokoro-ONNX. [PA][CA][REH]"""
        if not self.is_available():
            raise RuntimeError(f"TTS backend '{self.backend}' is not available. Please check your configuration.")
        
        try:
            # Debug logging for key operations [CDiP]
            logging.debug(f"Starting TTS synthesis: text='{text[:50]}...', voice='{self.voice}', output='{output_path}'")
            
            # Generate audio samples using correct kokoro-onnx API [PA][REH]
            try:
                samples, sample_rate = self.engine.create(
                    text, 
                    voice=self.voice, 
                    speed=1.0, 
                    lang="en-us"
                )
                logging.debug(f"Audio generated: {len(samples)} samples at {sample_rate}Hz")
            except Exception as e:
                logging.error(f"Failed to generate audio: {e}")
                raise
            
            # Save to output file [RM]
            try:
                import soundfile as sf
                sf.write(str(output_path), samples, sample_rate)
                file_size = output_path.stat().st_size
                logging.debug(f"Audio saved: {output_path} ({file_size} bytes)")
            except Exception as e:
                logging.error(f"Failed to save audio file: {e}")
                raise
                
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
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
                    logging.info(f"Purged old TTS cache file: {file.name}")
                except Exception as e:
                    logging.error(f"Error purging cache file {file.name}: {e}")
                    
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
            logging.debug(f"Using cached TTS: {file_path}")
            return file_path
            
        # Generate TTS using synchronous method [CA]
        try:
            self.synthesize(text, file_path)
            logging.debug(f"TTS generated: {file_path} ({file_path.stat().st_size} bytes)")
            return file_path
            
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
            raise
