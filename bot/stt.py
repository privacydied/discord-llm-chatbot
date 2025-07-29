"""
Speech-to-text module for Discord bot using faster-whisper and whisper.cpp
"""
import os
import json
import asyncio
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache

import torch
from faster_whisper import WhisperModel

from .util.logging import get_logger

logger = get_logger(__name__)

# Get environment variables
_ENGINE = os.getenv("STT_ENGINE", "faster-whisper")
_FALLBACK = os.getenv("STT_FALLBACK", "whispercpp")
_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base-int8")

@lru_cache
def _load_fw():
    """Load faster-whisper model with caching"""
    logger.info(f"Loading faster-whisper {_SIZE}")
    return WhisperModel(
        _SIZE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="int8"
    )


class STTManager:
    """Manages STT operations with support for multiple backends."""
    
    def __init__(self):
        self.available = False
        self.engine = _ENGINE
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the STT model."""
        try:
            if self.engine == "faster-whisper":
                self.model = _load_fw()
                self.available = True
                logger.info("✅ Initialized faster-whisper STT model")
            else:
                logger.warning(f"Unsupported STT engine: {self.engine}")
                self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize STT: {str(e)}")
            self.available = False
    
    def is_available(self) -> bool:
        """Check if STT is available."""
        return self.available
    
    async def transcribe_async(self, audio_path: Path) -> str:
        """Transcribe audio file asynchronously."""
        if not self.available:
            raise RuntimeError("STT engine not available")
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,  # Use default executor
                self._transcribe_sync,
                audio_path
            )
        except Exception as e:
            logger.error(f"STT transcription failed: {str(e)}")
            raise
    
    def _transcribe_sync(self, audio_path: Path) -> str:
        """Synchronous transcription wrapper."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            if self.engine == "faster-whisper" and self.model:
                segments, _ = self.model.transcribe(
                    str(audio_path),
                    beam_size=5,
                    language="en"
                )
                return " ".join(segment.text for segment in segments)
            else:
                raise RuntimeError(f"Unsupported STT engine: {self.engine}")
        except Exception as e:
            logger.error(f"STT transcription error: {str(e)}")
            raise

# Create a single instance of STTManager
stt_manager = STTManager()


async def transcribe_wav(path: Path) -> str:
    """Transcribe WAV file using configured STT engine"""
    loop = asyncio.get_running_loop()
    
    if _ENGINE == "faster-whisper":
        try:
            model = _load_fw()
            segments, _ = await loop.run_in_executor(
                None, 
                lambda: model.transcribe(str(path), vad_filter=True)
            )
            return " ".join(seg.text for seg in segments)
        except Exception as e:
            logger.error(f"faster-whisper failed: {e}")
            if _FALLBACK == "none":
                raise
    
    # Fallback to whisper.cpp binary
    if _FALLBACK == "whispercpp":
        model_path = Path(os.getenv("WHISPER_CPP_MODEL", "models/ggml-medium.bin"))
        if not model_path.exists():
            logger.error(f"whisper.cpp model not found: {model_path}")
            raise FileNotFoundError(f"whisper.cpp model not found: {model_path}")
        
        cmd = [
            "whisper.cpp", 
            "-m", str(model_path),
            "-f", str(path),
            "-of", "json"
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"whisper.cpp failed: {err.decode()}")
            raise RuntimeError(f"whisper.cpp failed: {err.decode()}")
        
        try:
            result = json.loads(out)
            return result["text"]
        except json.JSONDecodeError:
            logger.error("Failed to parse whisper.cpp output")
            raise
    
    raise RuntimeError("No valid STT engine available")

async def normalise_to_wav(attachment) -> Path:
    """Normalize audio attachment to 16kHz mono WAV"""
    try:
        # Save original attachment to temp file
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            await attachment.save(tmp.name)
            orig_path = Path(tmp.name)
        
        # Create output WAV file
        wav_path = orig_path.with_suffix(".wav")
        
        # Convert to 16kHz mono WAV using ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", str(orig_path),
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            str(wav_path)
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        await proc.wait()
        
        # Clean up original file
        orig_path.unlink()
        
        return wav_path
    except Exception as e:
        logger.error(f"Audio normalization failed: {e}")
        raise