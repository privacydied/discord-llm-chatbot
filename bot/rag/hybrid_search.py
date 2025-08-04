"""
Hybrid search integration that combines vector and keyword search with fallback logic.
"""
import os
from typing import List, Dict, Any, Optional, Union
import asyncio
from dataclasses import dataclass

from .chroma_backend import ChromaRAGBackend
from .vector_schema import HybridSearchConfig, SearchResult as RAGSearchResult
from .bootstrap import create_rag_system
from ..search import SearchResult as LegacySearchResult, search_memories, search_files, web_search
from ..util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HybridSearchResult:
    """Combined result from hybrid search with provenance tracking."""
    
    title: str
    snippet: str
    source: str
    score: float
    search_type: str  # "vector", "keyword", "hybrid", "fallback"
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_legacy_result(self) -> LegacySearchResult:
        """Convert to legacy SearchResult format for backward compatibility."""
        return LegacySearchResult(
            title=self.title,
            url="",  # RAG results don't have URLs
            snippet=self.snippet,
            source=f"{self.source} ({self.search_type})"
        )


class HybridRAGSearch:
    """Hybrid search system combining vector RAG with legacy keyword search."""
    
    def __init__(
        self,
        kb_path: str = "kb",
        db_path: str = "./chroma_db",
        config: Optional[HybridSearchConfig] = None,
        enable_rag: bool = True
    ):
        self.kb_path = kb_path
        self.db_path = db_path
        self.config = config or HybridSearchConfig()
        self.enable_rag = enable_rag and os.getenv("ENABLE_RAG", "true").lower() == "true"
        
        # RAG components (lazy initialization)
        self.rag_backend: Optional[ChromaRAGBackend] = None
        self.bootstrap = None
        self._initialized = False
        
        # Performance tracking
        self.search_stats = {
            "vector_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "fallback_searches": 0,
            "total_searches": 0
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the RAG system. Returns True if successful, False if fallback needed.
        """
        if self._initialized:
            return self.rag_backend is not None
        
        if not self.enable_rag:
            logger.info("[RAG] RAG system disabled via configuration")
            self._initialized = True
            return False
        
        try:
            # Create RAG system
            self.rag_backend, self.bootstrap = await create_rag_system(
                kb_path=self.kb_path,
                db_path=self.db_path,
                config=self.config
            )
            
            # Check if we need to bootstrap
            stats = await self.rag_backend.get_collection_stats()
            if stats.get("total_chunks", 0) == 0:
                logger.info("[RAG] Empty collection detected, running bootstrap...")
                bootstrap_result = await self.bootstrap.bootstrap_knowledge_base()
                logger.info(f"[RAG] Bootstrap completed: {bootstrap_result}")
            
            self._initialized = True
            logger.info("✔ Hybrid RAG search system initialized")
            return True
            
        except Exception as e:
            logger.error(f"[RAG] Failed to initialize RAG system: {e}")
            logger.warning("[RAG] Falling back to legacy search only")
            self._initialized = True
            return False
    
    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        max_results: int = 5,
        search_type: str = "hybrid"
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search across vector and keyword sources.
        
        Args:
            query: Search query
            user_id: Optional user ID for scoped search
            guild_id: Optional guild ID for scoped search
            max_results: Maximum results to return
            search_type: Type of search ("vector", "keyword", "hybrid")
            
        Returns:
            List of HybridSearchResult objects
        """
        self.search_stats["total_searches"] += 1
        
        # Initialize if needed
        rag_available = await self.initialize()
        
        if search_type == "hybrid" and rag_available:
            return await self._hybrid_search(query, user_id, guild_id, max_results)
        elif search_type == "vector" and rag_available:
            return await self._vector_search(query, user_id, guild_id, max_results)
        else:
            return await self._keyword_search(query, user_id, guild_id, max_results)
    
    async def _hybrid_search(
        self,
        query: str,
        user_id: Optional[str],
        guild_id: Optional[str],
        max_results: int
    ) -> List[HybridSearchResult]:
        """Perform hybrid vector + keyword search."""
        self.search_stats["hybrid_searches"] += 1
        
        try:
            # Perform both searches concurrently
            vector_task = self._vector_search(query, user_id, guild_id, self.config.max_vector_results)
            keyword_task = self._keyword_search(query, user_id, guild_id, self.config.max_keyword_results)
            
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.error(f"[RAG] Vector search failed in hybrid mode: {vector_results}")
                vector_results = []
            
            if isinstance(keyword_results, Exception):
                logger.error(f"[RAG] Keyword search failed in hybrid mode: {keyword_results}")
                keyword_results = []
            
            # Combine and rank results
            combined_results = self._combine_results(vector_results, keyword_results)
            
            # Apply deduplication
            deduplicated_results = self._deduplicate_results(combined_results)
            
            # Limit results
            final_results = deduplicated_results[:max_results]
            
            if self.config.log_retrieval_paths:
                logger.debug(f"[RAG] Hybrid search: '{query}' → {len(vector_results)} vector + "
                           f"{len(keyword_results)} keyword = {len(final_results)} final")
            
            return final_results
            
        except Exception as e:
            logger.error(f"[RAG] Hybrid search failed: {e}")
            if self.config.fallback_to_keyword_on_failure:
                logger.warning("[RAG] Falling back to keyword search")
                return await self._keyword_search(query, user_id, guild_id, max_results)
            raise
    
    async def _vector_search(
        self,
        query: str,
        user_id: Optional[str],
        guild_id: Optional[str],
        max_results: int
    ) -> List[HybridSearchResult]:
        """Perform vector similarity search."""
        self.search_stats["vector_searches"] += 1
        
        if not self.rag_backend:
            logger.warning("[RAG] Vector search requested but RAG backend not available")
            return []
        
        try:
            rag_results = await self.rag_backend.search(
                query=query,
                n_results=max_results,
                user_id=user_id,
                guild_id=guild_id
            )
            
            # Convert to HybridSearchResult
            hybrid_results = []
            for result in rag_results:
                hybrid_result = HybridSearchResult(
                    title=result.document.metadata.get('filename', f"Chunk {result.document.chunk_index}"),
                    snippet=result.snippet,
                    source=result.source,
                    score=result.similarity_score,
                    search_type="vector",
                    explanation=result.explanation,
                    metadata=result.document.metadata
                )
                hybrid_results.append(hybrid_result)
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"[RAG] Vector search failed: {e}")
            if self.config.fallback_to_keyword_on_failure:
                logger.warning("[RAG] Vector search failed, falling back to keyword")
                return await self._keyword_search(query, user_id, guild_id, max_results)
            raise
    
    async def _keyword_search(
        self,
        query: str,
        user_id: Optional[str],
        guild_id: Optional[str],
        max_results: int
    ) -> List[HybridSearchResult]:
        """Perform legacy keyword search."""
        self.search_stats["keyword_searches"] += 1
        
        try:
            # Use existing search functions
            memory_results = await search_memories(query, user_id, guild_id)
            
            # Convert legacy results to hybrid format
            hybrid_results = []
            
            for i, result in enumerate(memory_results[:max_results]):
                hybrid_result = HybridSearchResult(
                    title=result.title,
                    snippet=result.snippet,
                    source=result.source,
                    score=0.5,  # Default score for keyword results
                    search_type="keyword",
                    explanation="Legacy keyword search match",
                    metadata={"legacy_result": True}
                )
                hybrid_results.append(hybrid_result)
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"[RAG] Keyword search failed: {e}")
            return []
    
    def _combine_results(
        self,
        vector_results: List[HybridSearchResult],
        keyword_results: List[HybridSearchResult]
    ) -> List[HybridSearchResult]:
        """Combine vector and keyword results with weighted scoring."""
        combined = []
        
        # Re-score vector results
        for result in vector_results:
            result.score = result.score * self.config.vector_weight
            result.search_type = "hybrid"
            combined.append(result)
        
        # Re-score keyword results
        for result in keyword_results:
            result.score = result.score * self.config.keyword_weight
            result.search_type = "hybrid"
            combined.append(result)
        
        # Sort by combined score
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined
    
    def _deduplicate_results(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """Remove duplicate results based on content similarity."""
        if not results:
            return results
        
        deduplicated = [results[0]]  # Always keep the first (highest scored) result
        
        for result in results[1:]:
            is_duplicate = False
            
            for existing in deduplicated:
                # Simple text similarity check (could be enhanced with embedding similarity)
                similarity = self._text_similarity(result.snippet, existing.snippet)
                
                if similarity > self.config.deduplication_threshold:
                    is_duplicate = True
                    if self.config.log_retrieval_paths:
                        logger.debug(f"[RAG] Deduplicating similar result: {similarity:.3f}")
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard similarity of words)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def add_document(
        self,
        source_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_type: str = "text"
    ) -> bool:
        """
        Add a document to the RAG system.
        
        Args:
            source_id: Unique identifier for the document
            text: Document content
            metadata: Optional metadata
            file_type: File type for appropriate chunking
            
        Returns:
            True if successful, False otherwise
        """
        rag_available = await self.initialize()
        
        if not rag_available:
            logger.warning("[RAG] Cannot add document - RAG system not available")
            return False
        
        try:
            await self.rag_backend.add_document(source_id, text, metadata, file_type)
            logger.info(f"[RAG] Added document to RAG system: {source_id}")
            return True
        except Exception as e:
            logger.error(f"[RAG] Failed to add document {source_id}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get search statistics and system status."""
        stats = {
            "rag_enabled": self.enable_rag,
            "rag_initialized": self._initialized,
            "rag_available": self.rag_backend is not None,
            "search_stats": self.search_stats.copy()
        }
        
        if self.rag_backend:
            try:
                collection_stats = await self.rag_backend.get_collection_stats()
                stats["collection_stats"] = collection_stats
            except Exception as e:
                stats["collection_error"] = str(e)
        
        return stats


# Global instance for the bot
_hybrid_search: Optional[HybridRAGSearch] = None


async def get_hybrid_search() -> HybridRAGSearch:
    """Get or create the global hybrid search instance."""
    global _hybrid_search
    
    if _hybrid_search is None:
        _hybrid_search = HybridRAGSearch()
        await _hybrid_search.initialize()
    
    return _hybrid_search


async def hybrid_search(
    query: str,
    user_id: Optional[str] = None,
    guild_id: Optional[str] = None,
    max_results: int = 5,
    search_type: str = "hybrid"
) -> List[LegacySearchResult]:
    """
    Convenience function for hybrid search that returns legacy SearchResult format.
    
    This function maintains backward compatibility with existing code.
    """
    search_engine = await get_hybrid_search()
    
    hybrid_results = await search_engine.search(
        query=query,
        user_id=user_id,
        guild_id=guild_id,
        max_results=max_results,
        search_type=search_type
    )
    
    # Convert to legacy format
    legacy_results = [result.to_legacy_result() for result in hybrid_results]
    
    return legacy_results
