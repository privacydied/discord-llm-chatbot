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
import threading

import torch
from faster_whisper import WhisperModel

from .util.logging import get_logger

logger = get_logger(__name__)

# Get environment variables
_ENGINE = os.getenv("STT_ENGINE", "faster-whisper")
_FALLBACK = os.getenv("STT_FALLBACK", "whispercpp")
_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
# Optional performance/network controls
_FW_CACHE_DIR = os.getenv("STT_CACHE_DIR", "stt/cache")
_FW_LOCAL_ONLY = os.getenv("STT_LOCAL_ONLY", "0").lower() in ("1", "true", "yes", "y")
_FW_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
_FW_INIT_TIMEOUT = float(os.getenv("STT_INIT_TIMEOUT", "8"))

# Accept common patterns like "base-int8" -> (size=base, compute_type=int8)
_ALLOWED_CT = {
    "int8", "int8_float16", "int8_float32",
    "int16", "float16", "float32"
}

def _resolve_size_and_ct(size_str: str, default_ct: str) -> tuple[str, str]:
    s = (size_str or "").strip()
    ct = default_ct
    if "-" in s:
        cand_size, cand_ct = s.split("-", 1)
        if cand_ct in _ALLOWED_CT:
            s = cand_size
            ct = cand_ct
    return s, ct

@lru_cache
def _load_fw():
    """Load faster-whisper model with caching"""
    model_name, compute_type = _resolve_size_and_ct(_SIZE, _FW_COMPUTE_TYPE)
    logger.info(
        f"Loading faster-whisper model={model_name} compute_type={compute_type} "
        f"local_only={_FW_LOCAL_ONLY} cache={_FW_CACHE_DIR}"
    )
    # Prefer local CPU/GPU auto-detect but keep compute type configurable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=_FW_CACHE_DIR,
            local_files_only=_FW_LOCAL_ONLY,
        )
    except Exception:
        # Fallback without download options to preserve legacy behavior
        return WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )


class STTManager:
    """Manages STT operations with support for multiple backends."""
    
    def __init__(self):
        self.available = False
        self.engine = _ENGINE
        self.model = None
        # Background init thread/event to avoid blocking startup [REH]
        self._init_thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._init_model()
    
    def _init_model(self):
        """Initialize the STT model."""
        try:
            if self.engine == "faster-whisper":
                # Initialize in a background thread to prevent startup hangs
                def _bg_init():
                    try:
                        mdl = _load_fw()
                        self.model = mdl
                        self.available = True
                        logger.info("✅ Initialized faster-whisper STT model")
                    except Exception as e:
                        logger.error(f"Failed to initialize STT: {str(e)}")
                        self.available = False
                    finally:
                        self._ready_event.set()
                self._init_thread = threading.Thread(target=_bg_init, name="stt-fw-init", daemon=True)
                self._init_thread.start()
            else:
                logger.warning(f"Unsupported STT engine: {self.engine}")
                self.available = False
                self._ready_event.set()
        except Exception as e:
            logger.error(f"Failed to initialize STT: {str(e)}")
            self.available = False
            self._ready_event.set()
    
    def is_available(self) -> bool:
        """Check if STT is available."""
        return self.available
    
    async def transcribe_async(self, audio_path: Path) -> str:
        """Transcribe audio file asynchronously."""
        # Wait briefly for background init if needed to avoid false negatives [REH]
        if self.engine == "faster-whisper" and not self.available:
            logger.info("ℹ [STT] Waiting up to %.1fs for faster-whisper to initialize", _FW_INIT_TIMEOUT)
            loop = asyncio.get_running_loop()
            ready = await loop.run_in_executor(None, self._ready_event.wait, _FW_INIT_TIMEOUT)
            if not ready or not self.available:
                raise RuntimeError("STT engine not ready after init timeout")
        
        try:
            loop = asyncio.get_running_loop()
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
            # Prefer the manager's background-initialized model to avoid blocking here
            if not stt_manager.available:
                await loop.run_in_executor(None, stt_manager._ready_event.wait, _FW_INIT_TIMEOUT)
            if stt_manager.available and stt_manager.model:
                segments, _ = await loop.run_in_executor(
                    None,
                    lambda: stt_manager.model.transcribe(str(path), vad_filter=True)
                )
                return " ".join(seg.text for seg in segments)
            else:
                logger.error("✖ [STT] faster-whisper not ready after init timeout")
                if _FALLBACK == "none":
                    raise RuntimeError("faster-whisper not ready and no fallback enabled")
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
    """Normalize audio attachment to 16kHz mono WAV with smart preprocessing"""
    try:
        # Save original attachment to temp file
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            await attachment.save(tmp.name)
            orig_path = Path(tmp.name)
        
        logger.debug(f"[STT] Starting smart preprocessing for: {orig_path.name}")
        
        # Create output WAV file
        wav_path = orig_path.with_suffix(".wav")
        
        # Smart preprocessing pipeline:
        # 1. loudnorm - normalize audio levels first to prevent distortion
        # 2. atempo=1.5 - increase speed by 50% for faster processing
        # 3. silenceremove - cut silence so Whisper doesn't waste cycles on dead air
        # 4. Convert to 16kHz mono WAV format required by Whisper
        audio_filters = [
            "loudnorm=I=-16:TP=-1.5:LRA=11",  # Normalize loudness (EBU R128)
            "atempo=1.5",                      # Speed up by 50%
            "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-40dB:detection=peak,aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono"  # Remove silence + format
        ]
        
        cmd = [
            "ffmpeg", "-y", "-i", str(orig_path),
            "-af", ",".join(audio_filters),
            "-acodec", "pcm_s16le",
            str(wav_path)
        ]
        
        logger.debug(f"[STT] Running ffmpeg with filters: {','.join(audio_filters)}")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"[STT] ffmpeg preprocessing failed: {stderr.decode()}")
            # Fallback to basic conversion if smart preprocessing fails
            logger.warning("[STT] Falling back to basic audio conversion")
            return await _basic_normalise_to_wav(orig_path, wav_path)
        
        logger.debug(f"[STT] Smart preprocessing completed successfully")
        
        # Clean up original file
        orig_path.unlink()
        
        return wav_path
    except Exception as e:
        logger.error(f"[STT] Audio normalization failed: {e}")
        raise


async def _basic_normalise_to_wav(orig_path: Path, wav_path: Path) -> Path:
    """Fallback basic audio conversion without smart preprocessing"""
    try:
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
        logger.error(f"[STT] Basic audio conversion failed: {e}")
        raise