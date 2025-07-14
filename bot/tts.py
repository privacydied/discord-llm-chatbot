"""
TTS client for Kokoro-ONNX TTS microservice.
"""
import asyncio
import hashlib
import logging
import aiohttp
import time
import os
import subprocess
import requests
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import soundfile as sf

# Configuration
CACHE_DIR = Path("tts_cache")
MAX_CACHE_AGE_DAYS = 7
MAX_TEXT_LENGTH = 500  # Prevent abuse
TTS_SERVICE_URL = "http://localhost:5000/synthesize"

# Kokoro-ONNX model files
MODEL_DIR = Path("tts")
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
# Default paths for model files
DEFAULT_MODEL_PATH = MODEL_DIR / "kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = MODEL_DIR / "voices.json"
# Fallback URLs for download if files don't exist
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

class TTSManager:
    def __init__(self, config: dict):
        self.config = config
        self.voice_dir = Path(config.get("TTS_CACHE_DIR", "tts_cache"))
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        self.engine = None
        self.available = False  # Default to unavailable until initialized
        
        # Get TTS backend and voice from config/env
        self.backend = config.get("TTS_BACKEND", "kokoro-onnx").lower()
        self.voice = config.get("TTS_VOICE", "af")
        logging.debug(f"✅ TTSManager initialized with backend={self.backend}, voice={self.voice}")
        
        # Initialize TTS engine based on backend
        if self.backend == "kokoro-onnx":
            try:
                # Ensure model files are available
                model_path, voices_path = self.ensure_model_files()
                logging.debug(f"Using model files: {model_path}, {voices_path}")
                
                # Import and initialize Kokoro
                logging.debug("Attempting to import kokoro_onnx")
                from kokoro_onnx import Kokoro
                logging.debug("Successfully imported kokoro_onnx")
                
                # Import Misaki G2P with espeak fallback
                try:
                    from misaki import en, espeak
                    fallback = espeak.EspeakFallback(british=False)
                    self.g2p = en.G2P(trf=False, british=False, fallback=fallback)
                    logging.debug("✅ Misaki G2P initialized with espeak fallback")
                except Exception as e:
                    logging.error(f"❌ Failed to initialize Misaki G2P: {e}")
                    raise
                
                # Initialize Kokoro with model paths
                try:
                    logging.debug(f"Initializing Kokoro with model={model_path}, voices={voices_path}")
                    self.engine = Kokoro(str(model_path), str(voices_path))
                    
                    # Log available voices
                    if hasattr(self.engine, 'voices'):
                        available_voices = list(self.engine.voices)
                        logging.debug(f"Available voices: {available_voices}")
                        if len(available_voices) > 0:
                            logging.debug(f"First voice: {available_voices[0]}")
                            logging.debug(f"Voice type: {type(available_voices[0])}")
                    else:
                        logging.warning("Kokoro engine has no 'voices' attribute")
                    
                    # Check if configured voice is available
                    if hasattr(self.engine, 'voices') and self.voice not in self.engine.voices:
                        logging.error(f"Configured voice '{self.voice}' not found in available voices: {list(self.engine.voices)}")
                        # Set to a default voice from available voices
                        if len(self.engine.voices) > 0:
                            self.voice = list(self.engine.voices)[0]
                            logging.info(f"Falling back to voice: {self.voice}")
                    
                    self.available = True
                    logging.debug("✅ Kokoro-ONNX TTS engine initialized")
                except Exception as e:
                    logging.error(f"❌ Failed to initialize Kokoro: {e}", exc_info=True)
                    # Check if this is a binary format error
                    if "utf-8" in str(e) and "decode" in str(e):
                        logging.error("The voices file appears to be corrupted or in an unexpected format.")
                        logging.info("Attempting to re-download the voices file...")
                        # Force re-download of voices file
                        if VOICES_MODEL_PATH.exists():
                            VOICES_MODEL_PATH.unlink()
                        self._download_file(VOICES_MODEL_URL, VOICES_MODEL_PATH)
                        # Try initialization again
                        self.engine = Kokoro(str(model_path), str(voices_path))
                        self.available = True
                        logging.debug("✅ Kokoro-ONNX TTS engine initialized after re-downloading voices file")
                    else:
                        raise
            except ImportError as e:
                logging.error(f"❌ Kokoro-ONNX import failed: {e}")
                import sys
                logging.error(f"Python path: {sys.path}")
            except Exception as e:
                logging.error(f"❌ Failed to initialize Kokoro-ONNX: {e}", exc_info=True)
        else:
            logging.error(f"❌ Unsupported TTS backend: {self.backend}. Currently only 'kokoro-onnx' is supported.")
            
    def _setup_cache(self):
        """Ensure cache directory exists"""
        self.voice_dir.mkdir(exist_ok=True)
        
    def ensure_model_files(self) -> Tuple[Path, Path]:
        """Ensure model files exist, download if missing, and verify they are valid.
        
        Returns:
            Tuple[Path, Path]: Paths to the model and voices files
        """
        # Read paths from environment variables with defaults
        model_path = Path(self.config.get("TTS_MODEL_FILE", str(DEFAULT_MODEL_PATH)))
        voices_path = Path(self.config.get("TTS_VOICE_FILE", str(DEFAULT_VOICES_PATH)))
        
        # Log the paths being used
        logging.debug(f"Using model path: {model_path}")
        logging.debug(f"Using voices path: {voices_path}")
        
        # Create model directory if it doesn't exist
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check if model files exist, download if missing or corrupted
        try:
            # Download model file if missing
            if not model_path.exists() or model_path.stat().st_size == 0:
                logging.info(f"Model file not found at {model_path}, downloading from {KOKORO_MODEL_URL}")
                self._download_file(KOKORO_MODEL_URL, model_path)
            
            # Check if voices.json exists - we'll use our already created one
            if not voices_path.exists() or voices_path.stat().st_size == 0:
                logging.warning(f"Voices file not found at {voices_path}. Please run scripts/convert_voices_to_json.py")
                # If no voices.json exists, but we have the binary file, convert it
                bin_path = MODEL_DIR / "voices-v1.0.bin"
                if bin_path.exists() and bin_path.stat().st_size > 0:
                    logging.info(f"Attempting to convert binary voices file to JSON")
                    try:
                        # Convert binary to JSON (simplified version)
                        import numpy as np
                        with open(bin_path, 'rb') as f:
                            voices_data = np.load(f)
                        voice_names = list(voices_data.files)
                        with open(voices_path, 'w') as f:
                            import json
                            json.dump({"voices": voice_names}, f, indent=2)
                        logging.info(f"Successfully converted voices file to JSON format at {voices_path}")
                    except Exception as e:
                        logging.error(f"Failed to convert voices file: {e}")
            
            # Verify model file is valid (attempt to load it)
            try:
                import onnxruntime as ort
                # Just check if the model can be loaded
                _ = ort.InferenceSession(str(model_path))
                logging.debug(f"✅ Successfully verified ONNX model at {model_path}")
            except Exception as e:
                logging.error(f"❌ Invalid ONNX model file: {e}")
                # Remove corrupted file and re-download
                model_path.unlink(missing_ok=True)
                logging.info(f"Re-downloading Kokoro model file from {KOKORO_MODEL_URL}")
                self._download_file(KOKORO_MODEL_URL, model_path)
            
            # Verify JSON voices file - try to load it as JSON
            try:
                import json
                with open(voices_path, 'r') as f:
                    voices_data = json.load(f)
                # Check if it has the expected structure
                if "voices" not in voices_data:
                    logging.warning(f"Voices file at {voices_path} doesn't have 'voices' key")
                else:
                    logging.debug(f"✅ Successfully verified JSON voices file at {voices_path} with {len(voices_data['voices'])} voices")
            except Exception as e:
                logging.error(f"❌ Invalid JSON voices file: {e}")
            
            # Final verification after potential re-downloads
            if not model_path.exists() or model_path.stat().st_size == 0:
                raise FileNotFoundError(f"Kokoro model file not found or empty at {model_path}")
                
            if not voices_path.exists() or voices_path.stat().st_size == 0:
                raise FileNotFoundError(f"Voices file not found or empty at {voices_path}")
                
            return model_path, voices_path
        except Exception as e:
            logging.error(f"❌ Failed to ensure model files: {e}", exc_info=True)
            raise
        
    def _download_file(self, url: str, path: Path) -> None:
        """Download a file from a URL to a path.
        
        Args:
            url: URL to download from
            path: Path to save the file to
        """
        try:
            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file with progress
            logging.info(f"Downloading {url} to {path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Use a temporary file for download to avoid partial/corrupted files
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Log progress every 10%
                        if total_size > 0 and downloaded % (total_size // 10) < 8192:
                            percent = (downloaded / total_size) * 100
                            logging.info(f"Downloaded {percent:.1f}% of {path.name}")
            
            # Only move the file to its final location if the download completed successfully
            if temp_path.exists():
                # Remove existing file if it exists
                if path.exists():
                    path.unlink()
                # Rename temp file to final path
                temp_path.rename(path)
                logging.info(f"✅ Successfully downloaded {path.name}")
            else:
                raise FileNotFoundError(f"Downloaded file not found at {temp_path}")
            
        except Exception as e:
            logging.error(f"❌ Failed to download {url}: {e}")
            # Remove partial file if download failed
            if path.exists():
                path.unlink(missing_ok=True)
            # Remove temp file if it exists
            temp_path = path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise
    
    def purge_old_cache(self):
        """
        Remove cache files older than MAX_CACHE_AGE_DAYS.
        This is called automatically during maintenance tasks.
        """
        current_time = time.time()
        cutoff_time = current_time - (MAX_CACHE_AGE_DAYS * 24 * 60 * 60)
        
        for file in self.voice_dir.iterdir():
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
        
        for file in self.voice_dir.iterdir():
            if file.is_file():
                file_count += 1
                total_size += file.stat().st_size
                
        return {
            'files': file_count,
            'size_mb': total_size / (1024 * 1024),  # convert to MB
            'cache_dir': str(self.voice_dir)
        }
        
    def set_available(self, available: bool):
        """Set the availability of the TTS service."""
        self.available = available
        
    def is_available(self):
        """Check if TTS service is available."""
        return self.available
        
    async def load_model(self):
        """Initialize TTS model asynchronously with timeout."""
        try:
            # Use asyncio.to_thread for the blocking call
            providers = await asyncio.wait_for(
                asyncio.to_thread(self._get_onnx_providers),
                timeout=5.0
            )
            logging.debug(f"ONNX providers: {providers}")
            self.set_available(True)
            logging.debug("✅ TTS model loaded")
        except asyncio.TimeoutError:
            logging.warning("ONNX provider probe timed out, falling back to CPU")
            self.set_available(False)
        except Exception as e:
            logging.error(f"ONNX initialization failed: {e}", exc_info=True)
            self.set_available(False)
            
    def _get_onnx_providers(self):
        """Helper method to get ONNX providers in a synchronous context."""
        import onnxruntime as ort
        return ort.get_available_providers()
            
    async def synthesize_async(self, text: str, path: Path) -> Optional[Path]:
        """Synthesize text to speech asynchronously."""
        if not self.available:
            raise RuntimeError("TTS service unavailable")
            
        try:
            return await asyncio.to_thread(self.synthesize, text, path)
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            return None

    async def generate_tts(self, text: str, voice_id: str = "default") -> Path:
        """
        Generate TTS audio for the given text and return the file path.
        
        Args:
            text: The text to convert to speech.
            voice_id: The voice ID to use (default: "default").
            
        Returns:
            Path to the generated audio file.
        """
        if not self.available:
            raise RuntimeError("TTS service is not available")
            
        # Generate unique filename based on text and voice
        filename = f"{voice_id}_{hashlib.md5(text.encode()).hexdigest()}.wav"
        file_path = self.voice_dir / filename
        
        # Check cache first
        if file_path.exists():
            return file_path
            
        # Generate TTS
        try:
            if not self.engine or not self.available:
                raise RuntimeError(f"TTS backend '{self.backend}' is not available. Please check your configuration.")
                
            # Use configured TTS engine
            if self.backend == "kokoro-onnx":
                # Phonemize text using Misaki G2P
                phonemes, _ = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.g2p(text)
                )
                
                # Generate audio using Kokoro
                samples, sample_rate = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.engine.create(text, self.voice, phonemes=phonemes)
                )
                
                # Debug audio properties
                logging.debug(f"Audio generated: shape={samples.shape}, dtype={samples.dtype}, sample_rate={sample_rate}")
                logging.debug(f"Audio stats: min={samples.min()}, max={samples.max()}, mean={samples.mean()}, non-zero={np.count_nonzero(samples)}")
                
                # Check for silent audio
                if np.count_nonzero(samples) < 100:
                    logging.warning(f"Generated audio appears to be silent! Only {np.count_nonzero(samples)} non-zero samples detected.")
                
                # Save audio to WAV file
                wav_path = file_path.with_suffix('.wav')
                sf.write(str(wav_path), samples, sample_rate, format='WAV')
                
                # Convert to Opus for Discord voice messages
                opus_path = file_path.with_suffix('.opus')
                try:
                    # Use ffmpeg to convert to Opus format
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', str(wav_path),
                        '-c:a', 'libopus',
                        '-b:a', '64k',
                        str(opus_path)
                    ]
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        logging.error(f"ffmpeg conversion failed: {stderr.decode()}")
                        return wav_path  # Fall back to WAV if conversion fails
                    
                    return opus_path  # Return the Opus file path
                except Exception as e:
                    logging.error(f"Error converting to Opus: {e}")
                    return wav_path  # Fall back to WAV if conversion fails
            else:
                # Fallback for other backends (should not reach here)
                audio = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.engine.synthesize(text, voice=self.voice)
                )
                sf.write(str(file_path), audio, 22050, format='WAV')
                
            return file_path
            
        except Exception as e:
            logging.error(f"TTS generation failed: {e}", exc_info=True)
            raise Exception(f"TTS generation failed: {e}") from e
