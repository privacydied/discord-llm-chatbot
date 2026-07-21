"""ONNX-based embedding backend via fastembed — a torch-free drop-in for
SentenceTransformer embeddings. [PA]

Rationale: sentence-transformers drags the full torch runtime (~300-500 MB RSS)
into the process to run a ~90 MB MiniLM model. fastembed executes the same
model through onnxruntime, which this bot already ships for Kokoro TTS, so the
marginal memory cost is just the model weights themselves.

Vectors are dimensionally identical (384 for all-MiniLM-L6-v2) and
directionally near-identical to the torch fp32 outputs, so existing Chroma
collections keep working without re-ingestion.
"""

import asyncio
import os
import time
import weakref

import numpy as np

from bot.rag.embedding_interface import EmbeddingInterface
from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Serialize first-load per process; fastembed model init is not re-entrant.
_load_lock = asyncio.Lock()

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE_DIR_ENV = "FASTEMBED_CACHE_DIR"

# Live instances for idle-TTL unload (mirrors the whisper/kokoro pattern). [PA]
_instances: weakref.WeakSet = weakref.WeakSet()


def unload_idle_models(idle_seconds: float) -> int:
    """Release ONNX sessions of embedders idle for idle_seconds. Returns count.

    The MiniLM session pins ~180 MB; embedding traffic is bursty (memory
    curation, RAG queries), so quiet periods reclaim it and the next encode
    lazily reloads from the on-disk model cache in ~1-2s. Never raises.
    """
    if idle_seconds <= 0:
        return 0
    unloaded = 0
    now = time.monotonic()
    for inst in list(_instances):
        try:
            if inst.model is not None and (now - inst._last_used) >= idle_seconds:
                inst.model = None
                unloaded += 1
        except Exception as exc:
            logger.debug(f"fastembed idle unload skipped one instance: {exc}")
    if unloaded:
        logger.info(
            "fastembed.unload_idle | count=%s idle_s=%.0f",
            unloaded,
            idle_seconds,
            extra={"event": "fastembed.unload_idle", "subsys": "rag"},
        )
    return unloaded


def fastembed_supports(model_name: str) -> bool:
    """True if fastembed can run this model (import is cheap — no torch)."""
    try:
        from fastembed import TextEmbedding

        return any(m["model"] == model_name for m in TextEmbedding.list_supported_models())
    except Exception as exc:  # ImportError or registry API drift [REH]
        logger.debug(f"fastembed availability check failed: {exc}")
        return False


class FastEmbedEmbedding(EmbeddingInterface):
    """onnxruntime (fastembed) implementation of the embedding interface."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, normalize: bool = True) -> None:
        super().__init__(model_name, normalize)
        self.model = None
        self._last_used = time.monotonic()
        _instances.add(self)

    async def _initialize(self) -> None:
        if self.model is not None:
            return
        async with _load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from fastembed import TextEmbedding

        cache_dir = os.getenv(_CACHE_DIR_ENV) or None
        session_options = self._session_options()
        model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir, extra_session_options=session_options)
        dim = next(m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == self.model_name)
        self.model = model
        self.embedding_dim = dim
        logger.info(f"✅ Initialized fastembed {self.model_name} [dim={dim}, torch-free]")

    @staticmethod
    def _session_options() -> dict | None:
        """ORT session tuning: basic graph optimization saves ~25 MB RSS vs
        full optimization, with negligible speed impact on a 90 MB model. [PA]
        """
        try:
            import onnxruntime as ort

            return {"graph_optimization_level": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC}
        except Exception:  # pragma: no cover - onnxruntime always present in prod
            return None

    async def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode texts via onnxruntime off the event loop."""
        self._last_used = time.monotonic()
        await self._initialize()
        # Local strong reference: idle-TTL unload may null self.model from the
        # event loop while this runs in a worker thread. [REH]
        model = self.model
        if model is None:  # unloaded between init and here — reload once
            await self._initialize()
            model = self.model
        if isinstance(texts, str):
            texts = [texts]
        try:
            embeddings = await asyncio.to_thread(lambda: np.vstack(list(model.embed(texts))))
            embeddings = self._normalize_embeddings(embeddings)
            logger.debug(f"[RAG] Encoded {len(texts)} texts via fastembed [shape={embeddings.shape}]")
            return embeddings
        except Exception as e:
            logger.exception(f"[RAG] fastembed encoding failed: {e}")
            raise

    async def get_embedding_dimension(self) -> int:
        await self._initialize()
        return self.embedding_dim
