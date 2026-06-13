"""RAG (Retrieval Augmented Generation) module for vector-based document retrieval."""

from .chroma_backend import ChromaRAGBackend
from .embedding_interface import EmbeddingInterface
from .vector_schema import HybridSearchConfig, VectorDocument

__all__ = [
    "ChromaRAGBackend",
    "EmbeddingInterface",
    "HybridSearchConfig",
    "VectorDocument",
]
