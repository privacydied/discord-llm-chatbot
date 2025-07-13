"""
Speech-to-text module for Discord bot using faster-whisper and whisper.cpp
"""
import os
import json
import logging
import asyncio
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache

import torch
from faster_whisper import WhisperModel

# Get environment variables
_ENGINE = os.getenv("STT_ENGINE", "faster-whisper")
_FALLBACK = os.getenv("STT_FALLBACK", "whispercpp")
_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base-int8")

logger = logging.getLogger(__name__)

@lru_cache
def _load_fw():
    """Load faster-whisper model with caching"""
    logger.info(f"Loading faster-whisper {_SIZE}")
    return WhisperModel(
        _SIZE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="int8"
    )

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