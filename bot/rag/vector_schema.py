"""Vector document schema and configuration for RAG system."""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Schema for vector documents stored in the RAG system."""

    id: str  # UUID for chunk
    source_id: str  # Original file/context identifier
    chunk_text: str  # Actual text content
    embedding: list[float]  # L2-normalized vector
    metadata: dict[str, Any]  # Additional metadata
    version_hash: str  # SHA256 of source content
    chunk_index: int  # Position in original document
    confidence_score: float = 1.0  # Quality/relevance score
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        source_id: str,
        chunk_text: str,
        embedding: np.ndarray,
        chunk_index: int,
        metadata: dict[str, Any] | None = None,
        confidence_score: float = 1.0,
    ) -> "VectorDocument":
        """Create a new VectorDocument with auto-generated fields.

        Args:
            source_id: Identifier for the source document
            chunk_text: Text content of the chunk
            embedding: Embedding vector as numpy array
            chunk_index: Position in original document
            metadata: Optional metadata dictionary
            confidence_score: Quality score for the chunk

        Returns:
            VectorDocument instance

        """
        if metadata is None:
            metadata = {}

        # Generate unique ID
        doc_id = str(uuid.uuid4())

        # Create version hash from source content
        version_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        # Convert embedding to list
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        # Add standard metadata fields
        metadata.update(
            {
                "chunk_length": len(chunk_text),
                "embedding_dim": len(embedding_list),
                "source_type": metadata.get("source_type", "unknown"),
            },
        )

        return cls(
            id=doc_id,
            source_id=source_id,
            chunk_text=chunk_text,
            embedding=embedding_list,
            metadata=metadata,
            version_hash=version_hash,
            chunk_index=chunk_index,
            confidence_score=confidence_score,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "chunk_text": self.chunk_text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "version_hash": self.version_hash,
            "chunk_index": self.chunk_index,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorDocument":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            chunk_text=data["chunk_text"],
            embedding=data["embedding"],
            metadata=data["metadata"],
            version_hash=data["version_hash"],
            chunk_index=data["chunk_index"],
            confidence_score=data.get("confidence_score", 1.0),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
        )

    def get_embedding_array(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid vector + keyword search."""

    # Vector search parameters
    vector_confidence_threshold: float = 0.3  # Lowered from 0.7 to work with L2 distance conversion
    max_vector_results: int = 5
    vector_weight: float = 0.7

    # Keyword search parameters
    max_keyword_results: int = 3
    keyword_weight: float = 0.3

    # Fallback behavior
    fallback_to_keyword_on_failure: bool = True
    fallback_to_keyword_on_low_confidence: bool = True
    min_results_threshold: int = 1

    # Result combination
    combine_results: bool = True
    max_combined_results: int = 5
    deduplication_threshold: float = 0.9  # Cosine similarity threshold for dedup

    # Logging and monitoring
    log_retrieval_paths: bool = True
    log_confidence_scores: bool = True

    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100

    # Access control
    enforce_user_scoping: bool = True
    enforce_guild_scoping: bool = True

    # Performance & Loading [RAG]
    eager_vector_load: bool = True  # Load vector index at startup vs lazy load on first search
    background_indexing: bool = True  # Process new documents asynchronously vs synchronously

    # Background Processing [RAG]
    indexing_queue_size: int = 1000  # Maximum items in indexing queue
    indexing_workers: int = 2  # Number of background indexing workers
    indexing_batch_size: int = 10  # Documents to process per batch
    lazy_load_timeout: float = 30.0  # Timeout for lazy vector index loading (seconds)
    # Document caps [Phase 6-9]
    max_doc_bytes: int = 1048576  # 1 MB hard limit for RAG documents
    max_snippet_chars: int = 2000  # Max chars per retrieved snippet
    dedup_chunks: int = 500  # Max chunks to index per document

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.vector_confidence_threshold <= 1:
            msg = "vector_confidence_threshold must be between 0 and 1"
            raise ValueError(msg)

        if not 0 <= self.vector_weight <= 1:
            msg = "vector_weight must be between 0 and 1"
            raise ValueError(msg)

        if not 0 <= self.keyword_weight <= 1:
            msg = "keyword_weight must be between 0 and 1"
            raise ValueError(msg)

        if abs(self.vector_weight + self.keyword_weight - 1.0) > 1e-6:
            msg = "vector_weight + keyword_weight must equal 1.0"
            raise ValueError(msg)

        if self.chunk_size <= self.chunk_overlap:
            msg = "chunk_size must be greater than chunk_overlap"
            raise ValueError(msg)

        if self.min_chunk_size <= 0:
            msg = "min_chunk_size must be positive"
            raise ValueError(msg)

        # Validate new RAG performance fields [RAG]
        if self.indexing_queue_size <= 0:
            msg = "indexing_queue_size must be positive"
            raise ValueError(msg)

        if self.indexing_workers <= 0:
            msg = "indexing_workers must be positive"
            raise ValueError(msg)

        if self.indexing_batch_size <= 0:
            msg = "indexing_batch_size must be positive"
            raise ValueError(msg)

        if self.lazy_load_timeout <= 0:
            msg = "lazy_load_timeout must be positive"
            raise ValueError(msg)


@dataclass
class SearchResult:
    """Enhanced search result with vector similarity scores."""

    document: VectorDocument
    similarity_score: float
    search_type: str  # "vector", "keyword", "hybrid"
    rank: int
    explanation: str | None = None

    @property
    def title(self) -> str:
        """Get title from metadata or generate from source."""
        return self.document.metadata.get("title", f"Chunk {self.document.chunk_index}")

    @property
    def snippet(self) -> str:
        """Get text snippet, capped by RAG_MAX_SNIPPET_CHARS."""
        text = self.document.chunk_text
        # Snippet cap from config [Phase 6-9]
        max_snippet = 200
        try:
            from bot.config import load_config as _rag_snippet_config

            _rc = _rag_snippet_config()
            max_snippet = int(_rc.get("RAG_MAX_SNIPPET_CHARS", 200))
        except (ValueError, TypeError, OSError, ImportError, AttributeError) as exc:
            logger.debug(f"Failed to load RAG_MAX_SNIPPET_CHARS: {exc}")
        if len(text) > max_snippet:
            return text[: max_snippet - 3] + "..."
        return text

    @property
    def source(self) -> str:
        """Get source information."""
        return self.document.metadata.get("filename", self.document.source_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.document.id,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
            "similarity_score": self.similarity_score,
            "search_type": self.search_type,
            "rank": self.rank,
            "explanation": self.explanation,
            "metadata": self.document.metadata,
            "confidence_score": self.document.confidence_score,
        }


@dataclass
class ChunkingResult:
    """Result of document chunking operation."""

    chunks: list[str]
    metadata: dict[str, Any]
    source_hash: str

    @classmethod
    def create(
        cls,
        text: str,
        chunks: list[str],
        source_metadata: dict[str, Any] | None = None,
    ) -> "ChunkingResult":
        """Create chunking result with auto-generated hash."""
        if source_metadata is None:
            source_metadata = {}

        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        metadata = {
            "original_length": len(text),
            "num_chunks": len(chunks),
            "avg_chunk_length": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            **source_metadata,
        }

        return cls(chunks=chunks, metadata=metadata, source_hash=source_hash)
