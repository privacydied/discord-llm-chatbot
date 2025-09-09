"""
RAG (Retrieval Augmented Generation) module for vector-based document retrieval.
"""

from .embedding_interface import EmbeddingInterface
from .vector_schema import VectorDocument, HybridSearchConfig
from .chroma_backend import ChromaRAGBackend

__all__ = [
    "EmbeddingInterface",
    "VectorDocument",
    "HybridSearchConfig",
    "ChromaRAGBackend",
]
