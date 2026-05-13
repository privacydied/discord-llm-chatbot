"""Thin ChromaDB wrapper for curated memory semantic recall."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..rag.chroma_backend import ChromaRAGBackend
from ..rag.embedding_interface import EmbeddingInterface

logger = logging.getLogger(__name__)


class CuratedMemorySemanticStore:
    """Wrapper over the existing ChromaDB stack for curated memories only."""

    def __init__(
        self,
        db_path: str | Path = "./chroma_db",
        collection_name: str = "curated_memories",
        embedding_model: Optional[EmbeddingInterface] = None,
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._backend: Optional[ChromaRAGBackend] = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._backend = ChromaRAGBackend(
            db_path=str(self.db_path),
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        await self._backend.initialize()
        self._initialized = True

    @property
    def collection(self):
        if not self._backend:
            return None
        return self._backend.collection

    @property
    def embedding(self) -> EmbeddingInterface:
        if not self._backend:
            raise RuntimeError("Curated memory semantic store has not been initialized")
        return self._backend.embedding_model

    async def upsert(
        self,
        memory_id: str,
        document: str,
        metadata: Dict[str, Any],
    ) -> str:
        await self.initialize()
        if not document.strip():
            return memory_id

        # Cap text length before embedding to reduce CPU/memory [Phase 6-9]
        from ..config import load_config as _upsert_load_config

        _cfg = _upsert_load_config()
        max_chars = int(_cfg.get("MEMORY_MAX_TEXT_CHARS", 500))
        document = document[:max_chars]

        embedding = await self.embedding.encode_single(document)
        sanitized = self._sanitize_metadata(metadata)

        await asyncio.to_thread(
            self.collection.upsert,
            ids=[memory_id],
            documents=[document],
            embeddings=[embedding.tolist()],
            metadatas=[sanitized],
        )
        return memory_id

    async def delete(self, memory_id: str) -> None:
        await self.initialize()
        await asyncio.to_thread(self.collection.delete, ids=[memory_id])

    async def delete_many(self, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        await self.initialize()
        await asyncio.to_thread(self.collection.delete, ids=list(memory_ids))

    async def query(
        self,
        query: str,
        *,
        top_k: int = 6,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()
        if not query.strip():
            return []

        # Low-resource top_k cap [Phase 6-9]
        from ..config import load_config as _query_load_config

        _qc = _query_load_config()
        chroma_max = int(_qc.get("CHROMADB_MAX_RESULTS", 5))
        semantic_top_k_cfg = int(_qc.get("MEMORY_SEMANTIC_TOP_K", 5))
        effective_top_k = min(int(top_k), semantic_top_k_cfg, chroma_max)
        effective_top_k = max(1, effective_top_k)

        # Keyword prefilter before embedding to reduce search surface [Phase 6-9]
        query_words = set(query.lower().split())
        kw_tokens = {w for w in query_words if len(w) > 2}
        kw_where_doc: Optional[Dict[str, Any]] = where_document
        if kw_tokens and kw_where_doc is None:
            # Require at least one keyword token to appear in stored document
            kw_where_doc = {"$contains": " ".join(kw_tokens)}

        query_embedding = await self.embedding.encode_single(query)
        result = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding.tolist()],
            n_results=effective_top_k,
            where=where,
            where_document=kw_where_doc,
            include=["metadatas", "documents", "distances"],
        )

        if not result or not result.get("ids") or not result["ids"]:
            return []

        ids = result["ids"][0] or []
        documents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []

        items: List[Dict[str, Any]] = []
        for memory_id, document_text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            similarity = 1.0 / (1.0 + float(distance))
            items.append(
                {
                    "memory_id": memory_id,
                    "document": document_text,
                    "metadata": metadata,
                    "semantic_score": similarity,
                }
            )
        return items

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
        return sanitized
