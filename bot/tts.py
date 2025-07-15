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

        self._setup_cache()
        self._ensure_models_exist()

        # TTS engine state
        self.tts_instance = None
        self.available = False

        self.load_tts_instance()

    def _setup_cache(self):
        """Ensure cache directory exists [RM]"""
        self.cache_dir.mkdir(exist_ok=True)

    def _ensure_models_exist(self):
        """Check for and download TTS model files if they are missing. [RM]"""
        # This is a placeholder. A real implementation would download files.
        onnx_dir = Path(self.config.get("TTS_ONNX_DIR", "tts/onnx"))
        if not onnx_dir.exists():
            self.logger.warning(f"TTS ONNX directory not found at {onnx_dir}. Model files may be missing.")
        pass

    def load_tts_instance(self):
        """Load the appropriate TTS engine instance based on config. [CA]"""
        if self.backend == "kokoro-onnx":
            try:
                from .kokoro_direct import KokoroDirect

                onnx_dir = Path(self.config.get("TTS_ONNX_DIR", "tts/onnx"))
                voices_dir = Path(self.config.get("TTS_VOICES_DIR", "tts/voices"))

                if not onnx_dir.is_dir():
                    raise FileNotFoundError(f"ONNX model directory not found: {onnx_dir}")
                if not voices_dir.is_dir():
                    raise FileNotFoundError(f"Voices directory not found: {voices_dir}")

                self.logger.info(f"✅ Model and voice directories verified: {onnx_dir}, {voices_dir}", extra={'subsys': 'tts', 'event': 'init_verify'})

                self.tts_instance = KokoroDirect(onnx_dir=onnx_dir, voices_dir=voices_dir)

                if not self.tts_instance.is_available():
                    raise RuntimeError("KokoroDirect engine failed to initialize.")

                self.available = True
                self.logger.info(f"✅ Kokoro-ONNX TTS initialized successfully", extra={'subsys': 'tts', 'event': 'init_success'})

            except ImportError:
                self.logger.warning("Kokoro-ONNX not installed. Please run 'uv pip install kokoro-onnx'", extra={'subsys': 'tts', 'event': 'import_error'})
                self.available = False
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Kokoro-ONNX: {e}", exc_info=True, extra={'subsys': 'tts', 'event': 'init_fail'})
                self.available = False
        else:
            self.logger.warning(f"TTS backend '{self.backend}' is not supported.", extra={'subsys': 'tts', 'event': 'unsupported_backend'})
            self.available = False

    def is_available(self) -> bool:
        """Check if TTS backend is available and ready. [CA]"""
        return self.available and self.tts_instance is not None

    async def synthesize_async(self, text: str, output_path: Path, voice: Union[str, np.ndarray] = None):
        """Run synchronous synthesis in an executor to avoid blocking. [PA][REH]"""
        if not self.is_available():
            raise RuntimeError("TTS is not available.")
        loop = self.bot.loop
        await loop.run_in_executor(None, self.synthesize, text, output_path, voice)
        return output_path

    def synthesize(self, text: str, output_path: Path, voice: Union[str, np.ndarray] = None):
        """Generate speech from text using the loaded TTS instance. [PA][CA][REH]"""
        if not self.is_available():
            raise RuntimeError("TTS engine not initialized")

        try:
            self.logger.debug(f"Starting TTS synthesis: text='{text[:50]}...', output='{output_path}'")
            samples, sample_rate = self.tts_instance.create(text, self.voice)
            
            import soundfile as sf
            sf.write(str(output_path), samples, sample_rate)
            self.logger.debug(f"Audio saved: {output_path} ({output_path.stat().st_size} bytes)")
        except Exception as e:
            self.logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            raise

    def purge_old_cache(self):
        """Remove cache files older than MAX_CACHE_AGE_DAYS. [RM]"""
        cutoff = time.time() - (MAX_CACHE_AGE_DAYS * 24 * 60 * 60)
        for file in self.cache_dir.iterdir():
            if file.is_file() and file.stat().st_mtime < cutoff:
                try:
                    file.unlink()
                    self.logger.info(f"Purged old TTS cache file: {file.name}")
                except Exception as e:
                    self.logger.error(f"Error purging cache file {file.name}: {e}")

    def get_cache_stats(self) -> dict:
        """Get cache statistics. [CA]"""
        files = list(self.cache_dir.iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        return {
            'files': len(files),
            'size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

    async def generate_tts(self, text: str, voice_id: str = "default") -> Path:
        """Generate TTS audio, cache it, and return the file path. [PA][CA][REH]"""
        if not self.is_available():
            raise RuntimeError("TTS service is not available")

        filename = f"{voice_id}_{hashlib.md5(text.encode()).hexdigest()}.wav"
        file_path = self.cache_dir / filename

        if file_path.exists():
            self.logger.debug(f"Using cached TTS: {file_path}")
            return file_path

        await self.synthesize_async(text, file_path)
        self.logger.debug(f"TTS generated: {file_path} ({file_path.stat().st_size} bytes)")
        return file_path
