"""
Abstract interface for embedding models with pluggable implementations.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np
from ..util.logging import get_logger

logger = get_logger(__name__)


class EmbeddingInterface(ABC):
    """Abstract base class for embedding model implementations."""
    
    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.embedding_dim: Optional[int] = None
        
    @abstractmethod
    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into embedding vector(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this model."""
        pass
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply L2 normalization to embeddings."""
        if not self.normalize:
            return embeddings
            
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    async def encode_single(self, text: str) -> np.ndarray:
        """Convenience method to encode a single text."""
        result = await self.encode([text])
        return result[0]


class SentenceTransformerEmbedding(EmbeddingInterface):
    """Sentence-transformers implementation of embedding interface."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True):
        super().__init__(model_name, normalize)
        self.model = None
        self._initialized = False
        
    async def _initialize(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[1]
            
            self._initialized = True
            logger.info(f"✔ Initialized {self.model_name} [dim={self.embedding_dim}]")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize sentence-transformers model: {e}")
            raise
    
    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts using sentence-transformers."""
        await self._initialize()
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            )
            
            # Apply normalization if enabled
            embeddings = self._normalize_embeddings(embeddings)
            
            logger.debug(f"[RAG] Encoded {len(texts)} texts [shape={embeddings.shape}]")
            return embeddings
            
        except Exception as e:
            logger.error(f"[RAG] Embedding encoding failed: {e}")
            raise
    
    async def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        await self._initialize()
        return self.embedding_dim


class OpenAIEmbedding(EmbeddingInterface):
    """OpenAI API implementation of embedding interface."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None, normalize: bool = True):
        super().__init__(model_name, normalize)
        self.api_key = api_key
        self.client = None
        self._initialized = False
        
        # Known dimensions for OpenAI models
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
    async def _initialize(self):
        """Initialize OpenAI client."""
        if self._initialized:
            return
            
        try:
            import openai
            
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.embedding_dim = self._model_dimensions.get(self.model_name, 1536)
            
            self._initialized = True
            logger.info(f"✔ Initialized OpenAI {self.model_name} [dim={self.embedding_dim}]")
            
        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding client: {e}")
            raise
    
    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts using OpenAI API."""
        await self._initialize()
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = np.array([item.embedding for item in response.data])
            
            # Apply normalization if enabled
            embeddings = self._normalize_embeddings(embeddings)
            
            logger.debug(f"[RAG] Encoded {len(texts)} texts via OpenAI [shape={embeddings.shape}]")
            return embeddings
            
        except Exception as e:
            logger.error(f"[RAG] OpenAI embedding failed: {e}")
            raise
    
    async def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        await self._initialize()
        return self.embedding_dim


def create_embedding_model(model_type: str = "sentence-transformers", **kwargs) -> EmbeddingInterface:
    """
    Factory function to create embedding models.
    
    Args:
        model_type: Type of model ("sentence-transformers" or "openai")
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        EmbeddingInterface implementation
    """
    if model_type == "sentence-transformers":
        return SentenceTransformerEmbedding(**kwargs)
    elif model_type == "openai":
        return OpenAIEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")
