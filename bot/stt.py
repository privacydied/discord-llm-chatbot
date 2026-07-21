"""Speech-to-text runtime primitives built around faster-whisper.

Provides lazy model loading with CPU-friendly defaults and utilities that higher-level
pipelines (hear.py) use to orchestrate preprocessing, adaptive model selection, and
chunked inference.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from faster_whisper import WhisperModel

logger = get_logger(__name__)

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
    # Defer torch.set_num_threads to when torch is actually loaded (during STT).
    # Setting it here via module-level import torch loads ~600 modules even when
    # STT is disabled. Applied lazily in _preload_cpu_threads().


def _preload_cpu_threads() -> None:
    """Apply torch.set_num_threads IF torch is already in the process. [PA]

    faster-whisper runs on ctranslate2 and does not need torch at all; importing
    torch here just to set a thread count used to pin ~300-400 MB of RSS for the
    process lifetime. Only touch it when some other subsystem already paid that
    cost (checked via sys.modules — never triggers the import ourselves).
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        torch.set_num_threads(_CPU_THREADS)
    except Exception as exc:
        logger.debug(f"torch thread setting failed: {exc}")


# ---------------------------------------------------------------------------
# Configuration from environment (kept minimal to avoid new env surface area)
# ---------------------------------------------------------------------------

_ENGINE = os.getenv("STT_ENGINE", "faster-whisper")
_DEFAULT_MODEL_DECL = os.getenv("WHISPER_MODEL_SIZE", "base")
_CACHE_DIR = os.getenv("STT_CACHE_DIR", "stt/cache")
_LOCAL_ONLY = os.getenv("STT_LOCAL_ONLY", "0").lower() in ("1", "true", "yes", "y")
_COMPUTE_TYPE_DECL = os.getenv("STT_COMPUTE_TYPE", "int8")
_INIT_TIMEOUT = float(os.getenv("STT_INIT_TIMEOUT", "8"))
# Each cached whisper model variant pins its full weights + ctranslate2 buffers
# in memory for the process lifetime; downgrades (base->tiny on slow_decode)
# previously stacked indefinitely. Cap how many distinct specs stay resident
# at once -- the default model always counts as one slot. [PA][CMV]
_MODEL_CACHE_MAX = max(1, int(os.getenv("STT_MODEL_CACHE_MAX", "2")))


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
    """Resolve a declaration like 'base-int8' into a ModelSpec."""
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
    """CUDA detection via ctranslate2 (the actual inference runtime). [PA]

    Previously probed torch.cuda, which imported the full torch runtime
    (~300-400 MB RSS) even on CPU-only hosts. ctranslate2 is already a hard
    dependency of faster-whisper, so this adds nothing to the footprint.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception as exc:
        logger.debug(f"CUDA detection failed: {exc}")
    return "cpu"


def _model_ladder() -> tuple[str, ...]:
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


def _downgrade(size: str) -> str | None:
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
        # OrderedDict for LRU eviction -- each entry pins a full model's weights
        # + ctranslate2 buffers in memory; unbounded growth (e.g. repeated
        # base->tiny downgrades) was the largest contributor to RSS growth. [PA]
        self._model_cache: OrderedDict[ModelSpec, WhisperModel] = OrderedDict()
        self._model_locks: dict[ModelSpec, threading.Lock] = {}
        self._ready_event = threading.Event()
        self._default_spec = _resolve_spec(_DEFAULT_MODEL_DECL, _COMPUTE_TYPE_DECL)
        self._available = False
        self._init_thread: threading.Thread | None = None
        self._last_used = time.monotonic()
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

            _preload_cpu_threads()

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
                logger.exception("Failed to initialize STT: %s", exc)
                self._available = False
            finally:
                self._ready_event.set()

        self._init_thread = threading.Thread(target=_loader, name="stt-fw-init", daemon=True)
        self._init_thread.start()

    def _evict_lru_locked(self) -> None:
        """Drop least-recently-used cached models beyond the cap. Caller holds the lock.

        The default spec is never evicted -- it's reloaded on essentially every
        request, so evicting it would just trade memory for repeated reload cost.
        """
        while len(self._model_cache) > _MODEL_CACHE_MAX:
            for victim_spec in self._model_cache:
                if victim_spec != self._default_spec:
                    break
            else:
                break  # only the default spec remains; nothing safe to evict
            victim = self._model_cache.pop(victim_spec)
            del victim
            gc.collect()
            logger.info(
                "stt.model_cache.evict | spec=%s-%s cache_size=%s",
                victim_spec.size,
                victim_spec.compute_type,
                len(self._model_cache),
                extra={"event": "stt.model_cache.evict", "subsys": "stt"},
            )

    def _load_model(self, spec: ModelSpec) -> WhisperModel:
        from faster_whisper import WhisperModel

        self._last_used = time.monotonic()
        lock = self._get_lock_for(spec)
        with lock:
            model = self._model_cache.get(spec)
            if model:
                self._model_cache.move_to_end(spec)
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
            self._evict_lru_locked()
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

    async def ensure_ready(self, timeout: float | None = None) -> bool:
        """Await readiness of the default model. Returns True if ready."""
        timeout = timeout if timeout is not None else _INIT_TIMEOUT
        loop = asyncio.get_running_loop()
        ready = await loop.run_in_executor(None, self._ready_event.wait, timeout)
        return bool(ready and self.available)

    async def ensure_model(self, spec: ModelSpec) -> WhisperModel:
        """Ensure model for the given spec exists, loading lazily via executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._load_model, spec)

    def downgrade_spec(self, spec: ModelSpec) -> ModelSpec | None:
        nxt = _downgrade(spec.size)
        if not nxt:
            return None
        return ModelSpec(nxt, spec.compute_type)

    def evict_idle_models(self) -> int:
        """Thread-safe eviction of all cached non-default models. Returns count evicted.

        Public entry point for callers outside this module (e.g. the periodic
        health check reclaiming memory under pressure) -- acquires each
        victim's own lock before dropping it, unlike the internal
        `_evict_lru_locked` which assumes the caller already holds the lock
        for the spec being loaded. [PA][REH]
        """
        victims = [spec for spec in list(self._model_cache) if spec != self._default_spec]
        evicted = 0
        for spec in victims:
            with self._get_lock_for(spec):
                model = self._model_cache.pop(spec, None)
            if model is not None:
                del model
                evicted += 1
        if evicted:
            gc.collect()
            logger.info(
                "stt.model_cache.evict_idle | count=%s",
                evicted,
                extra={"event": "stt.model_cache.evict_idle", "subsys": "stt"},
            )
        return evicted

    def evict_if_idle(self, idle_seconds: float) -> int:
        """Drop ALL cached models — including the default — after prolonged idle. [PA]

        Unlike `evict_idle_models` (memory-pressure path, spares the default
        spec), this is the idle-TTL path: when no STT request has touched the
        manager for `idle_seconds`, every resident model's weights are released.
        The next request lazily reloads via `_load_model` at the cost of a few
        seconds. An in-flight transcription keeps its own strong reference to
        the model object, so eviction here never invalidates active work — the
        memory is simply reclaimed once that call finishes. Returns count evicted.
        """
        if idle_seconds <= 0 or not self._model_cache:
            return 0
        if time.monotonic() - self._last_used < idle_seconds:
            return 0
        evicted = 0
        for spec in list(self._model_cache):
            with self._get_lock_for(spec):
                model = self._model_cache.pop(spec, None)
            if model is not None:
                del model
                evicted += 1
        if evicted:
            gc.collect()
            logger.info(
                "stt.model_cache.evict_idle_ttl | count=%s idle_s=%.0f",
                evicted,
                idle_seconds,
                extra={"event": "stt.model_cache.evict_idle_ttl", "subsys": "stt"},
            )
        return evicted

    # Backwards-compatible helpers -------------------------------------------------

    async def transcribe_async(self, audio_path: Path) -> str:
        """Legacy compatibility: transcribe from a file path using default CPU-friendly params.
        Newer code should orchestrate preprocessing + chunking explicitly (see hear.py).
        """
        if self.engine != "faster-whisper":
            msg = f"Unsupported STT engine: {self.engine}"
            raise RuntimeError(msg)

        ready = await self.ensure_ready()
        if not ready:
            msg = "STT engine not ready after init timeout"
            raise RuntimeError(msg)

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


# Global singleton — created lazily to avoid spinning up a thread + importing
# torch/faster_whisper at module import time. [IV][REH]
_stt_manager: STTManager | None = None


def get_stt_manager() -> STTManager:
    """Return the global STTManager, creating it on first call."""
    global _stt_manager
    if _stt_manager is None:
        _stt_manager = STTManager()
    return _stt_manager


def get_stt_manager_if_initialized() -> STTManager | None:
    """Return the global STTManager only if already created; never instantiates one.

    For callers that want to act on an existing manager (e.g. evicting idle
    cached models under memory pressure) without the side effect of cold-
    starting STT -- which spins up a background model-load thread -- for a
    bot that has never actually used speech-to-text. [PA][REH]
    """
    return _stt_manager


# Backwards-compatible alias for existing callers that access stt_manager directly.
# This is a module-level property-like shim — direct access from other modules
# will create the manager at access time, NOT at import time.
def __getattr__(name: str):
    if name == "stt_manager":
        return get_stt_manager()
    raise AttributeError(name)


# Convenience shim maintained for backwards compatibility -----------------------


async def transcribe_wav(path: Path) -> str:
    """Alias to stt_manager.transcribe_async for historical callers."""
    return await get_stt_manager().transcribe_async(path)
