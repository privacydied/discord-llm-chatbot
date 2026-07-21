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

import numpy as np

from bot.rag.embedding_interface import EmbeddingInterface
from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Serialize first-load per process; fastembed model init is not re-entrant.
_load_lock = asyncio.Lock()

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE_DIR_ENV = "FASTEMBED_CACHE_DIR"


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
        model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)
        dim = next(m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == self.model_name)
        self.model = model
        self.embedding_dim = dim
        logger.info(f"✅ Initialized fastembed {self.model_name} [dim={dim}, torch-free]")

    async def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode texts via onnxruntime off the event loop."""
        await self._initialize()
        if isinstance(texts, str):
            texts = [texts]
        try:
            embeddings = await asyncio.to_thread(lambda: np.vstack(list(self.model.embed(texts))))
            embeddings = self._normalize_embeddings(embeddings)
            logger.debug(f"[RAG] Encoded {len(texts)} texts via fastembed [shape={embeddings.shape}]")
            return embeddings
        except Exception as e:
            logger.exception(f"[RAG] fastembed encoding failed: {e}")
            raise

    async def get_embedding_dimension(self) -> int:
        await self._initialize()
        return self.embedding_dim
