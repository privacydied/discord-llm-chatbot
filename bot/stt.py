"""
Speech-to-text runtime primitives built around faster-whisper.

Provides lazy model loading with CPU-friendly defaults and utilities that higher-level
pipelines (hear.py) use to orchestrate preprocessing, adaptive model selection, and
chunked inference.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from faster_whisper import WhisperModel

from .utils.logging import get_logger
from .utils.torch_compat import ensure_reduce_op_alias

logger = get_logger(__name__)

# Ensure third-party whisper dependencies do not trigger torch distributed warnings.
ensure_reduce_op_alias()

# ---------------------------------------------------------------------------
# Environment controls — keep numeric libraries single-threaded on CPU
# ---------------------------------------------------------------------------

_THREAD_LOCK = threading.Lock()
_CPU_THREADS = 2
_THREAD_ENVS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

with _THREAD_LOCK:
    for _env in _THREAD_ENVS:
        if not os.getenv(_env):
            os.environ[_env] = "1"
    if not os.getenv("CT_NUM_THREADS"):
        os.environ["CT_NUM_THREADS"] = str(_CPU_THREADS)
    try:
        torch.set_num_threads(_CPU_THREADS)
    except Exception:
        # Some Torch builds may not expose set_num_threads; ignore
        pass


# ---------------------------------------------------------------------------
# Configuration from environment (kept minimal to avoid new env surface area)
# ---------------------------------------------------------------------------

_ENGINE = os.getenv("STT_ENGINE", "faster-whisper")
_DEFAULT_MODEL_DECL = os.getenv("WHISPER_MODEL_SIZE", "base")
_CACHE_DIR = os.getenv("STT_CACHE_DIR", "stt/cache")
_LOCAL_ONLY = os.getenv("STT_LOCAL_ONLY", "0").lower() in ("1", "true", "yes", "y")
_COMPUTE_TYPE_DECL = os.getenv("STT_COMPUTE_TYPE", "int8")
_INIT_TIMEOUT = float(os.getenv("STT_INIT_TIMEOUT", "8"))


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model configuration."""

    size: str
    compute_type: str


def _normalize_compute_type(ct: str) -> str:
    allowed = {
        "int8",
        "int8_float16",
        "int8_float32",
        "int16",
        "float16",
        "float32",
    }
    candidate = (ct or "").strip()
    return candidate if candidate in allowed else "int8"


@lru_cache
def _resolve_spec(declaration: str, default_compute: str) -> ModelSpec:
    """
    Resolve a declaration like 'base-int8' into a ModelSpec.
    """
    decl = (declaration or "").strip()
    compute_type = _normalize_compute_type(default_compute)
    if "-" in decl:
        size_candidate, ct_candidate = decl.split("-", 1)
        size_candidate = size_candidate.strip()
        ct_candidate = _normalize_compute_type(ct_candidate.strip())
        if size_candidate:
            return ModelSpec(size_candidate, ct_candidate)
    if decl:
        return ModelSpec(decl, compute_type)
    return ModelSpec("base", compute_type)


def _device_for_runtime() -> str:
    # Even though optimised for CPU, keep CUDA detection for environments that may supply GPUs.
    if torch.cuda.is_available():
        try:
            return "cuda"
        except Exception:
            pass
    return "cpu"


def _model_ladder() -> Tuple[str, ...]:
    # Prioritise smaller CPU-friendly variants for downgrade logic.
    return (
        "large-v3",
        "large-v2",
        "large",
        "medium",
        "small",
        "base",
        "tiny",
        "tiny.en",
    )


def _downgrade(size: str) -> Optional[str]:
    ladder = _model_ladder()
    try:
        idx = ladder.index(size)
    except ValueError:
        return None
    if idx == len(ladder) - 1:
        return None
    return ladder[idx + 1]


class STTManager:
    """Lazy-loading faster-whisper manager with light concurrency controls."""

    def __init__(self) -> None:
        self.engine = _ENGINE
        self._model_cache: Dict[ModelSpec, WhisperModel] = {}
        self._model_locks: Dict[ModelSpec, threading.Lock] = {}
        self._ready_event = threading.Event()
        self._default_spec = _resolve_spec(_DEFAULT_MODEL_DECL, _COMPUTE_TYPE_DECL)
        self._available = False
        self._init_thread: Optional[threading.Thread] = None
        self._warm_default_async()

    # ------------------------------------------------------------------ Utils

    def _get_lock_for(self, spec: ModelSpec) -> threading.Lock:
        lock = self._model_locks.get(spec)
        if lock is None:
            lock = threading.Lock()
            self._model_locks[spec] = lock
        return lock

    # --------------------------------------------------------------- Warm load

    def _warm_default_async(self) -> None:
        if self.engine != "faster-whisper":
            logger.warning("Unsupported STT engine: %s", self.engine)
            self._available = False
            self._ready_event.set()
            return

        def _loader() -> None:
            try:
                self._load_model(self._default_spec)
                self._available = True
                logger.info(
                    "✅ Initialized faster-whisper model=%s compute_type=%s device=%s",
                    self._default_spec.size,
                    self._default_spec.compute_type,
                    _device_for_runtime(),
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to initialize STT: %s", exc)
                self._available = False
            finally:
                self._ready_event.set()

        self._init_thread = threading.Thread(
            target=_loader, name="stt-fw-init", daemon=True
        )
        self._init_thread.start()

    def _load_model(self, spec: ModelSpec) -> WhisperModel:
        lock = self._get_lock_for(spec)
        with lock:
            model = self._model_cache.get(spec)
            if model:
                return model
            device = _device_for_runtime()
            try:
                model = WhisperModel(
                    spec.size,
                    device=device,
                    compute_type=spec.compute_type,
                    download_root=_CACHE_DIR,
                    local_files_only=_LOCAL_ONLY,
                )
            except Exception as exc:
                # retry without cache hints for parity with legacy path
                logger.warning(
                    "Primary model load failed (size=%s compute=%s): %s. Retrying without cache hints.",
                    spec.size,
                    spec.compute_type,
                    exc,
                )
                model = WhisperModel(
                    spec.size,
                    device=device,
                    compute_type=spec.compute_type,
                )
            self._model_cache[spec] = model
            return model

    # -------------------------------------------------------------- Public API

    @property
    def available(self) -> bool:
        return self._available

    @property
    def default_spec(self) -> ModelSpec:
        return self._default_spec

    @property
    def cpu_threads(self) -> int:
        return _CPU_THREADS

    async def ensure_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Await readiness of the default model. Returns True if ready.
        """
        timeout = timeout if timeout is not None else _INIT_TIMEOUT
        loop = asyncio.get_running_loop()
        ready = await loop.run_in_executor(None, self._ready_event.wait, timeout)
        return bool(ready and self.available)

    async def ensure_model(self, spec: ModelSpec) -> WhisperModel:
        """
        Ensure model for the given spec exists, loading lazily via executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._load_model, spec)

    def downgrade_spec(self, spec: ModelSpec) -> Optional[ModelSpec]:
        nxt = _downgrade(spec.size)
        if not nxt:
            return None
        return ModelSpec(nxt, spec.compute_type)

    # Backwards-compatible helpers -------------------------------------------------

    async def transcribe_async(self, audio_path: Path) -> str:
        """
        Legacy compatibility: transcribe from a file path using default CPU-friendly params.
        Newer code should orchestrate preprocessing + chunking explicitly (see hear.py).
        """
        if self.engine != "faster-whisper":
            raise RuntimeError(f"Unsupported STT engine: {self.engine}")

        ready = await self.ensure_ready()
        if not ready:
            raise RuntimeError("STT engine not ready after init timeout")

        model = await self.ensure_model(self.default_spec)
        loop = asyncio.get_running_loop()

        def _transcribe() -> str:
            segments, _info = model.transcribe(
                str(audio_path),
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                word_timestamps=False,
                task="transcribe",
                language=None,
                cpu_threads=_CPU_THREADS,
            )
            return " ".join(segment.text for segment in segments)

        return await loop.run_in_executor(None, _transcribe)


# Global singleton used throughout the bot
stt_manager = STTManager()


# Convenience shim maintained for backwards compatibility -----------------------

async def transcribe_wav(path: Path) -> str:
    """Alias to stt_manager.transcribe_async for historical callers."""
    return await stt_manager.transcribe_async(path)
