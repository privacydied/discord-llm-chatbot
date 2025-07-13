"""
TTS client for Kokoro-ONNX TTS microservice.
"""
import asyncio
import hashlib
import logging
import aiohttp
import time
from pathlib import Path
from typing import Optional

# Configuration
CACHE_DIR = Path("tts_cache")
MAX_CACHE_AGE_DAYS = 7
MAX_TEXT_LENGTH = 500  # Prevent abuse
TTS_SERVICE_URL = "http://localhost:5000/synthesize"

class TTSManager:
    def __init__(self):
        self._setup_cache()
        self.available = True  # Assume service is available
    
    def _setup_cache(self):
        """Ensure cache directory exists"""
        CACHE_DIR.mkdir(exist_ok=True)
    
    def purge_old_cache(self):
        """
        Remove cache files older than MAX_CACHE_AGE_DAYS.
        This is called automatically during maintenance tasks.
        """
        current_time = time.time()
        cutoff_time = current_time - (MAX_CACHE_AGE_DAYS * 24 * 60 * 60)
        
        for file in CACHE_DIR.iterdir():
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
        
        for file in CACHE_DIR.iterdir():
            if file.is_file():
                file_count += 1
                total_size += file.stat().st_size
                
        return {
            'files': file_count,
            'size_mb': total_size / (1024 * 1024),  # convert to MB
            'cache_dir': str(CACHE_DIR)
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
