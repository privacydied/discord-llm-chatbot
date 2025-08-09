"""
STT Orchestrator: multi-provider asynchronous speech-to-text with caching and basic modes.

Initial modes implemented:
- single (default): call primary provider only (wraps existing stt_manager)
- cascade_primary_then_fallbacks: try providers in order until acceptable

Stubs in place for:
- parallel_first_acceptable
- parallel_best_of
- hybrid_draft_then_finalize

This module integrates non-invasively: if STT_ENABLE=false or misconfigured, it
falls back to the existing `stt_manager.transcribe_async()` path.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .util.logging import get_logger
from .config import load_config
from .stt import stt_manager  # Leverage existing local STT

logger = get_logger(__name__)

# Constants [CMV]
DEFAULT_MODE = "single"  # single | cascade_primary_then_fallbacks | parallel_first_acceptable | parallel_best_of | hybrid_draft_then_finalize
CACHE_TTL_DEFAULT = 600  # seconds
CONF_MIN_DEFAULT = 0.0   # Accept any if provider does not supply confidence


@dataclass
class TranscriptResult:
    provider: str
    text: str
    latency_ms: int
    confidence: Optional[float] = None
    cost_usd: Optional[float] = None
    success: bool = True
    detail: Optional[str] = None

    def acceptable(self, min_conf: float) -> bool:
        # If provider does not report confidence, treat as acceptable under current policy.
        return self.success and (self.confidence is None or self.confidence >= min_conf)


class AbstractSTTProvider:
    name: str = "abstract"

    def __init__(self, concurrency: int = 2):
        self._sema = asyncio.Semaphore(concurrency)

    async def transcribe(self, audio_path: Path, deadline_ms: Optional[int], config: Dict[str, Any]) -> TranscriptResult:
        raise NotImplementedError


class LocalWhisperProvider(AbstractSTTProvider):
    name: str = "local_whisper"

    async def transcribe(self, audio_path: Path, deadline_ms: Optional[int], config: Dict[str, Any]) -> TranscriptResult:
        start = time.time()
        try:
            async with self._sema:
                # We ignore deadline_ms for now; future: enforce via timeout.
                text = await stt_manager.transcribe_async(audio_path)
            latency_ms = int((time.time() - start) * 1000)
            # No confidence available from current local path
            return TranscriptResult(provider=self.name, text=text, latency_ms=latency_ms, confidence=None, cost_usd=0.0, success=True)
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            logger.error(f"[{self.name}] transcription failed: {e}")
            return TranscriptResult(provider=self.name, text="", latency_ms=latency_ms, confidence=None, cost_usd=0.0, success=False, detail=str(e))


class STTOrchestrator:
    def __init__(self) -> None:
        self._providers: List[AbstractSTTProvider] = []
        self._singleflight: Dict[str, asyncio.Future] = {}
        self._cache: Dict[str, Tuple[float, TranscriptResult]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._reload_config()

    def _reload_config(self) -> None:
        cfg = load_config()
        # Config [CMV]
        self.enabled: bool = bool(cfg.get("STT_ENABLE", True))
        self.mode: str = str(cfg.get("STT_MODE", DEFAULT_MODE))
        self.min_conf: float = float(cfg.get("STT_CONFIDENCE_MIN", CONF_MIN_DEFAULT))
        self.cache_ttl: int = int(cfg.get("STT_CACHE_TTL", CACHE_TTL_DEFAULT))
        active = cfg.get("STT_ACTIVE_PROVIDERS", ["local_whisper"]) or ["local_whisper"]
        local_conc = int(cfg.get("STT_LOCAL_CONCURRENCY", 2))

        # Build providers list
        providers: List[AbstractSTTProvider] = []
        for p in active:
            p = (p or "").strip().lower()
            if p in ("local", "local_whisper", "whisper", "faster_whisper"):
                providers.append(LocalWhisperProvider(concurrency=local_conc))
            else:
                logger.warning(f"Unknown STT provider '{p}' — skipping")
        if not providers:
            providers.append(LocalWhisperProvider(concurrency=local_conc))
        self._providers = providers

    @staticmethod
    def _hash_audio_file(path: Path) -> str:
        # Robust cache key based on bytes; may be large but ensures correctness. [PA]
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _cache_get(self, key: str) -> Optional[TranscriptResult]:
        ent = self._cache.get(key)
        if not ent:
            return None
        ts, res = ent
        if (time.time() - ts) <= self.cache_ttl:
            return res
        # expired
        self._cache.pop(key, None)
        return None

    def _cache_set(self, key: str, res: TranscriptResult) -> None:
        self._cache[key] = (time.time(), res)

    async def transcribe(self, audio_path: Path) -> str:
        """
        Main entry. Returns transcript text.
        Obeys STT_ENABLE and falls back to stt_manager when disabled.
        """
        # Reload config periodically could be added; for now rely on cached config utility [PA]
        self._reload_config()

        if not self.enabled:
            logger.debug("[STT-Orch] Disabled — using legacy stt_manager path")
            return await stt_manager.transcribe_async(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cache_key = self._hash_audio_file(audio_path)
        cached = self._cache_get(cache_key)
        if cached:
            logger.info(f"[STT-Orch] Cache hit by {cached.provider} ({cached.latency_ms}ms)")
            return cached.text

        # Single-flight [PA]
        lock = self._get_lock(cache_key)
        async with lock:
            # Check cache again after awaiting lock (thundering herd) [PA]
            cached2 = self._cache_get(cache_key)
            if cached2:
                logger.info(f"[STT-Orch] Cache hit (post-lock) by {cached2.provider}")
                return cached2.text

            # Dispatch based on mode
            if self.mode == "single":
                res = await self._run_single(audio_path)
            elif self.mode == "cascade_primary_then_fallbacks":
                res = await self._run_cascade(audio_path)
            elif self.mode == "parallel_first_acceptable":
                res = await self._run_parallel_first(audio_path)
            elif self.mode == "parallel_best_of":
                res = await self._run_parallel_best(audio_path)
            elif self.mode == "hybrid_draft_then_finalize":
                res = await self._run_hybrid(audio_path)
            else:
                logger.warning(f"[STT-Orch] Unknown mode '{self.mode}', defaulting to single")
                res = await self._run_single(audio_path)

            if not res.success:
                raise RuntimeError(res.detail or "STT orchestration failed")

            # Cache acceptable result only [IV]
            if res.acceptable(self.min_conf):
                self._cache_set(cache_key, res)
            return res.text

    async def _run_single(self, audio_path: Path) -> TranscriptResult:
        provider = self._providers[0]
        logger.info(f"[STT-Orch] single via {provider.name}")
        return await provider.transcribe(audio_path, deadline_ms=None, config={})

    async def _run_cascade(self, audio_path: Path) -> TranscriptResult:
        logger.info("[STT-Orch] cascade providers")
        last_error: Optional[TranscriptResult] = None
        for idx, p in enumerate(self._providers):
            logger.info(f"[STT-Orch] cascade try {idx+1}/{len(self._providers)}: {p.name}")
            res = await p.transcribe(audio_path, deadline_ms=None, config={})
            if res.acceptable(self.min_conf):
                return res
            last_error = res
        return last_error or TranscriptResult(provider="none", text="", latency_ms=0, success=False, detail="No providers configured")

    async def _run_parallel_first(self, audio_path: Path) -> TranscriptResult:
        logger.info("[STT-Orch] parallel_first_acceptable")
        # Minimal viable implementation: run all, pick first acceptable to finish.
        # For now, since only local provider exists, behave like single.
        return await self._run_single(audio_path)

    async def _run_parallel_best(self, audio_path: Path) -> TranscriptResult:
        logger.info("[STT-Orch] parallel_best_of")
        # Placeholder: run providers, then select by confidence/quality. Currently only local provider — return it.
        return await self._run_single(audio_path)

    async def _run_hybrid(self, audio_path: Path) -> TranscriptResult:
        logger.info("[STT-Orch] hybrid_draft_then_finalize")
        # Placeholder: produce quick draft then finalize with higher quality provider.
        return await self._run_single(audio_path)


# Global orchestrator instance
stt_orchestrator = STTOrchestrator()
