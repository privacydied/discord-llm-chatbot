"""
Hybrid search integration that combines vector and keyword search with fallback logic.
"""

import os
import time
import threading
from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass
from enum import Enum

from .chroma_backend import ChromaRAGBackend
from .vector_schema import HybridSearchConfig
from .bootstrap import create_rag_system
from .indexing_queue import IndexingQueue, IndexingTask
from ..search import SearchResult as LegacySearchResult
from ..search.factory import get_search_provider
from ..search.types import SearchQueryParams, SafeSearch
from ..utils.logging import get_logger

logger = get_logger(__name__)


class IndexState(Enum):
    """Vector index loading state."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


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
            source=f"{self.source} ({self.search_type})",
        )


class HybridRAGSearch:
    """Hybrid search system combining vector RAG with legacy keyword search."""

    def __init__(
        self,
        kb_path: str = "kb",
        db_path: str = "./chroma_db",
        config: Optional[HybridSearchConfig] = None,
        enable_rag: bool = True,
    ):
        self.kb_path = kb_path
        self.db_path = db_path
        self.config = config or HybridSearchConfig()
        self.enable_rag = (
            enable_rag and os.getenv("ENABLE_RAG", "true").lower() == "true"
        )

        # RAG components (lazy initialization)
        self.rag_backend: Optional[ChromaRAGBackend] = None
        self.bootstrap = None
        self._initialized = False

        # Index state management [RAG]
        self._index_state = IndexState.NOT_LOADED
        self._index_load_lock = threading.Lock()
        self._index_load_start_time: Optional[float] = None
        self._reranker_load_time: Optional[float] = None
        self._vector_load_time: Optional[float] = None

        # Background indexing [RAG]
        self._indexing_queue: Optional[IndexingQueue] = None

        # Performance tracking
        self.search_stats = {
            "vector_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "fallback_searches": 0,
            "total_searches": 0,
            "first_query_lazy_loads": 0,
            "lazy_load_time_ms": 0.0,
        }

        # Log configuration at startup [RAG]
        logger.info(
            f"[RAG] Hybrid search configured: eager_vector_load={self.config.eager_vector_load}, "
            f"background_indexing={self.config.background_indexing}, "
            f"indexing_workers={self.config.indexing_workers}"
        )

    async def initialize(self) -> bool:
        """
        Initialize the RAG system with reranker loading and optional eager vector loading.
        Returns True if successful, False if fallback needed.
        """
        if self._initialized:
            return self.rag_backend is not None

        if not self.enable_rag:
            logger.info("[RAG] RAG system disabled via configuration")
            self._initialized = True
            return False

        try:
            reranker_start = time.time()

            # Always load reranker immediately (lightweight) [RAG]
            logger.info("[RAG] Loading reranker (lightweight component)...")
            # Note: Reranker loading would happen in create_rag_system

            if self.config.eager_vector_load:
                # Eager mode: load vector index at startup [RAG]
                logger.info(
                    "[RAG] Eager vector loading enabled, initializing full RAG system..."
                )
                await self._load_vector_index()
            else:
                # Lazy mode: defer vector index loading [RAG]
                logger.info(
                    "[RAG] Lazy vector loading enabled, deferring index load until first search"
                )
                self._index_state = IndexState.NOT_LOADED

                # Still create basic RAG backend for reranker, but don't load vector index
                self.rag_backend, self.bootstrap = await create_rag_system(
                    kb_path=self.kb_path,
                    db_path=self.db_path,
                    config=self.config,
                    load_vector_index=False,  # New parameter to control vector loading
                )

            self._reranker_load_time = (time.time() - reranker_start) * 1000

            # Initialize background indexing queue [RAG]
            if self.config.background_indexing:
                logger.info(
                    f"[RAG] Initializing background indexing with {self.config.indexing_workers} workers"
                )
                self._indexing_queue = IndexingQueue(
                    rag_backend=self.rag_backend,
                    max_queue_size=self.config.indexing_queue_size,
                    num_workers=self.config.indexing_workers,
                    batch_size=self.config.indexing_batch_size,
                    enabled=True,
                )
                await self._indexing_queue.start_workers()
            else:
                logger.info(
                    "[RAG] Background indexing disabled, using synchronous processing"
                )
                self._indexing_queue = IndexingQueue(
                    rag_backend=self.rag_backend, enabled=False
                )

            self._initialized = True

            # Log startup summary [RAG]
            logger.info(
                f"✔ Hybrid RAG search system initialized: "
                f"reranker_load_time={self._reranker_load_time:.1f}ms, "
                f"vector_index={'loaded' if self._index_state == IndexState.LOADED else 'deferred'}, "
                f"background_indexing={'enabled' if self.config.background_indexing else 'disabled'}"
            )
            return True

        except Exception as e:
            logger.error(f"[RAG] Failed to initialize RAG system: {e}")
            logger.warning("[RAG] Falling back to legacy search only")
            self._initialized = True
            return False

    async def _load_vector_index(self) -> bool:
        """
        Load the vector index with thread-safe, idempotent behavior.

        This method implements the NOT_LOADED → LOADING → LOADED state machine
        with proper locking to ensure only one load operation occurs.

        Returns:
            True if loaded successfully, False if failed
        """
        # Quick check without lock for already loaded state
        if self._index_state == IndexState.LOADED:
            return True

        # Thread-safe state transition with lock
        with self._index_load_lock:
            # Double-check pattern: verify state hasn't changed while waiting for lock
            if self._index_state == IndexState.LOADED:
                return True

            if self._index_state == IndexState.LOADING:
                # Another task is already loading. We'll wait OUTSIDE the lock.
                logger.debug(
                    "[RAG] Vector index already loading, waiting for completion..."
                )
                already_loading = True
            else:
                # We're the first to attempt loading
                logger.info("[RAG] Starting lazy vector index load...")
                self._index_state = IndexState.LOADING
                self._index_load_start_time = time.time()
                already_loading = False

        # If another task is loading, wait without holding the lock to avoid deadlock
        if "already_loading" in locals() and already_loading:
            start_wait = time.time()
            while (
                self._index_state == IndexState.LOADING
                and time.time() - start_wait < self.config.lazy_load_timeout
            ):
                await asyncio.sleep(0.1)
            return self._index_state == IndexState.LOADED

        # Perform the actual loading outside the lock
        try:
            vector_load_start = time.time()

            # Create RAG system with full vector index loading
            self.rag_backend, self.bootstrap = await create_rag_system(
                kb_path=self.kb_path, db_path=self.db_path, config=self.config
            )

            # Check collection status (for informational purposes only)
            stats = await self.rag_backend.get_collection_stats()
            total_chunks = stats.get("total_chunks", 0)
            if total_chunks == 0:
                logger.info(
                    "[RAG] ⚠️ Vector index loaded but collection is empty - no documents indexed yet"
                )
                logger.info(
                    "[RAG] 💡 To populate the knowledge base, run: !rag bootstrap"
                )
            else:
                logger.info(
                    f"[RAG] ✅ Vector index loaded successfully with {total_chunks:,} chunks"
                )

            self._vector_load_time = (time.time() - vector_load_start) * 1000

            # Update state to loaded
            with self._index_load_lock:
                self._index_state = IndexState.LOADED

            # Update metrics
            self.search_stats["first_query_lazy_loads"] += 1
            self.search_stats["lazy_load_time_ms"] = self._vector_load_time

            logger.info(
                f"[RAG] ✔ Lazy vector index load completed: "
                f"load_time={self._vector_load_time:.1f}ms, "
                f"total_chunks={stats.get('total_chunks', 0)}"
            )

            return True

        except Exception as e:
            # Mark as failed and reset state
            with self._index_load_lock:
                self._index_state = IndexState.FAILED

            logger.error(f"[RAG] ✖ Lazy vector index load failed: {e}")
            return False

    def is_index_loaded(self) -> bool:
        """
        Check if the vector index is loaded and ready for search.

        Returns:
            True if index is loaded, False otherwise
        """
        return self._index_state == IndexState.LOADED

    def get_index_state(self) -> str:
        """
        Get the current index loading state.

        Returns:
            String representation of current state
        """
        return self._index_state.value

    async def _ensure_vector_index_loaded(self) -> bool:
        """
        Ensure the vector index is loaded before performing vector search.

        This is the main entry point for lazy loading triggered by search operations.

        Returns:
            True if index is available, False if unavailable
        """
        if self._index_state == IndexState.LOADED:
            return True

        if self._index_state == IndexState.FAILED:
            logger.warning("[RAG] Vector index in failed state, skipping vector search")
            return False

        if self._index_state == IndexState.LOADING:
            # Non-blocking: another task is loading; proceed with fallback path
            logger.debug("[RAG] Vector index load in progress; not blocking reply path")
            return False

        # Non-blocking lazy load trigger for first search
        logger.info(
            "[RAG] Vector index not loaded, triggering background lazy load for first search..."
        )
        asyncio.create_task(self._load_vector_index())
        return False

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        max_results: int = 5,
        search_type: str = "hybrid",
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search across vector and keyword sources.

        This method implements lazy loading: if the vector index is not loaded,
        it will trigger a one-time load before performing vector search.

        Args:
            query: Search query
            user_id: Optional user ID for scoped search
            guild_id: Optional guild ID for scoped search
            max_results: Maximum number of results to return
            search_type: Type of search ("vector", "keyword", "hybrid")

        Returns:
            List of search results with provenance tracking
        """
        search_start = time.time()

        # Ensure RAG system is initialized
        await self.initialize()

        # Update search statistics
        self.search_stats["total_searches"] += 1

        try:
            if search_type == "hybrid":
                self.search_stats["hybrid_searches"] += 1
                results = await self._hybrid_search(
                    query, user_id, guild_id, max_results
                )
            elif search_type == "vector":
                self.search_stats["vector_searches"] += 1
                results = await self._vector_search(
                    query, user_id, guild_id, max_results
                )
            elif search_type == "keyword":
                self.search_stats["keyword_searches"] += 1
                results = await self._keyword_search(
                    query, user_id, guild_id, max_results
                )
            else:
                raise ValueError(f"Invalid search_type: {search_type}")

            search_time = (time.time() - search_start) * 1000

            # Log search completion with metrics [RAG]
            logger.debug(
                f"[RAG] Search completed: type={search_type}, "
                f"results={len(results)}, time={search_time:.1f}ms, "
                f"index_state={self._index_state.value}"
            )

            return results

        except Exception as e:
            search_time = (time.time() - search_start) * 1000
            logger.error(
                f"[RAG] Search failed: type={search_type}, "
                f"time={search_time:.1f}ms, error={e}"
            )

            # Fallback to keyword search on error
            if search_type != "keyword":
                logger.info("[RAG] Falling back to keyword search due to error")
                self.search_stats["fallback_searches"] += 1
                return await self._keyword_search(query, user_id, guild_id, max_results)

            # If keyword search also failed, return empty results
            return []

    async def _hybrid_search(
        self,
        query: str,
        user_id: Optional[str],
        guild_id: Optional[str],
        max_results: int,
    ) -> List[HybridSearchResult]:
        """Perform hybrid vector + keyword search with lazy loading."""

        # Try vector search first (with lazy loading)
        vector_results = await self._vector_search(
            query, user_id, guild_id, self.config.max_vector_results
        )
        if vector_results is None:
            vector_results = []

        # Get keyword results
        keyword_results = await self._keyword_search(
            query, user_id, guild_id, self.config.max_keyword_results
        )
        if keyword_results is None:
            keyword_results = []

        # Combine and rank results
        if self.config.combine_results:
            combined_results = self._combine_results(vector_results, keyword_results)

            # Deduplicate if enabled
            if self.config.deduplication_threshold > 0:
                combined_results = self._deduplicate_results(combined_results)

            # Limit to max results
            final_results = combined_results[:max_results]
        else:
            # Just concatenate without sophisticated combining
            all_results = vector_results + keyword_results
            final_results = all_results[:max_results]

        # Update search type for final results
        for result in final_results:
            result.search_type = "hybrid"

        if self.config.log_retrieval_paths:
            logger.info(
                f"[RAG] Hybrid search: {len(vector_results)} vector + {len(keyword_results)} keyword = {len(final_results)} final"
            )

        return final_results

    async def _vector_search(
        self,
        query: str,
        user_id: Optional[str],
        guild_id: Optional[str],
        max_results: int,
    ) -> List[HybridSearchResult]:
        """Perform vector similarity search with lazy loading."""
        # Ensure vector index is loaded before search [RAG]
        if not await self._ensure_vector_index_loaded():
            logger.warning(
                "[RAG] Vector search requested but index not available, falling back to keyword"
            )
            return await self._keyword_search(query, user_id, guild_id, max_results)

        if not self.rag_backend:
            logger.warning(
                "[RAG] Vector search requested but RAG backend not available"
            )
            return []

        try:
            # Perform vector search
            vector_results = await self.rag_backend.search(
                query=query, max_results=max_results, user_id=user_id, guild_id=guild_id
            )

            # Check for None results
            if vector_results is None:
                logger.warning(
                    "[RAG] Vector search returned None, falling back to keyword"
                )
                if self.config.fallback_to_keyword_on_failure:
                    return await self._keyword_search(
                        query, user_id, guild_id, max_results
                    )
                return []

            # Convert to hybrid results
            hybrid_results = []
            for i, result in enumerate(vector_results):
                hybrid_result = HybridSearchResult(
                    title=result.title,
                    snippet=result.snippet,
                    source=result.source,
                    score=result.similarity_score,
                    search_type="vector",
                    explanation=f"Vector similarity: {result.similarity_score:.3f}",
                    metadata=result.document.metadata,
                )
                hybrid_results.append(hybrid_result)

            if self.config.log_retrieval_paths:
                logger.info(
                    f"[RAG] Vector search returned {len(hybrid_results)} results"
                )

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
        max_results: int,
    ) -> List[HybridSearchResult]:
        """Perform legacy keyword search."""
        self.search_stats["keyword_searches"] += 1

        try:
            # Use pluggable search provider for keyword/web search
            provider = get_search_provider()

            # Resolve SafeSearch from env (default: moderate)
            safe_env = os.getenv("SEARCH_SAFE", SafeSearch.MODERATE.value).lower()
            safe_level = (
                SafeSearch(safe_env)
                if safe_env in {s.value for s in SafeSearch}
                else SafeSearch.MODERATE
            )

            # Build query params
            try:
                timeout_ms = int(os.getenv("DDG_TIMEOUT_MS", "5000"))
            except ValueError:
                timeout_ms = 5000

            params = SearchQueryParams(
                query=query,
                max_results=max_results,
                safesearch=safe_level,
                locale=os.getenv("SEARCH_LOCALE") or None,
                timeout_ms=timeout_ms,
            )

            web_results = await provider.search(params)

            # Convert provider results to hybrid format
            hybrid_results: List[HybridSearchResult] = []
            for res in web_results[:max_results]:
                hybrid_results.append(
                    HybridSearchResult(
                        title=res.title,
                        snippet=res.snippet or "",
                        source=res.url or "web",
                        score=0.5,  # Default score for keyword results
                        search_type="keyword",
                        explanation="Web keyword search match",
                        metadata={"url": res.url},
                    )
                )

            return hybrid_results

        except Exception as e:
            logger.error(f"[RAG] Keyword search failed: {e}")
            return []

    def _combine_results(
        self,
        vector_results: List[HybridSearchResult],
        keyword_results: List[HybridSearchResult],
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

    def _deduplicate_results(
        self, results: List[HybridSearchResult]
    ) -> List[HybridSearchResult]:
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
                        logger.debug(
                            f"[RAG] Deduplicating similar result: {similarity:.3f}"
                        )
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
        file_type: str = "text",
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
            # If background indexing is enabled, enqueue the task non-blockingly
            if self._indexing_queue and self._indexing_queue.enabled:
                task = IndexingTask(
                    source_id=source_id,
                    text=text,
                    metadata=metadata,
                    file_type=file_type,
                )
                enq_ok = await self._indexing_queue.enqueue_task(task)
                if enq_ok:
                    logger.info(
                        f"[RAG] Enqueued document for background indexing: {source_id}"
                    )
                    return True
                else:
                    logger.warning(
                        f"[RAG] Indexing queue full, processing document synchronously: {source_id}"
                    )
                    # Fall through to synchronous path

            # Synchronous processing (background disabled or enqueue failed)
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
            "search_stats": self.search_stats.copy(),
        }

        if self.rag_backend:
            try:
                collection_stats = await self.rag_backend.get_collection_stats()
                stats["collection_stats"] = collection_stats
            except Exception as e:
                stats["collection_error"] = str(e)

        return stats

    async def wipe_collection(self) -> bool:
        """Wipe the entire RAG collection/database.

        Returns:
            True if successful, False otherwise
        """
        rag_available = await self.initialize()

        if not rag_available:
            logger.warning("[RAG] Cannot wipe collection - RAG system not available")
            return False

        try:
            # Wipe the collection in the backend
            await self.rag_backend.wipe_collection()

            # Reset search statistics
            self.search_stats = {
                "vector_searches": 0,
                "keyword_searches": 0,
                "hybrid_searches": 0,
                "fallback_searches": 0,
                "total_searches": 0,
            }

            logger.info("[RAG] Successfully wiped RAG collection")
            return True

        except Exception as e:
            logger.error(f"[RAG] Failed to wipe collection: {e}")
            return False

    async def close(self) -> None:
        """Clean up RAG system resources during shutdown."""
        try:
            logger.debug("[RAG] Closing hybrid search system")

            # Shutdown background indexing first to stop new writes
            if self._indexing_queue:
                try:
                    await self._indexing_queue.shutdown(timeout=30.0)
                except Exception as qe:
                    logger.warning(f"[RAG] Error shutting down indexing queue: {qe}")
                finally:
                    self._indexing_queue = None

            if self.rag_backend:
                await self.rag_backend.close()
                self.rag_backend = None

            if self.bootstrap:
                # Bootstrap doesn't have a close method, just clear reference
                self.bootstrap = None

            self._initialized = False
            logger.info("[RAG] ✔ Hybrid search system closed successfully")

        except Exception as e:
            logger.warning(f"[RAG] Error closing hybrid search system: {e}")


# Global instance for the bot
_hybrid_search: Optional[HybridRAGSearch] = None
_shutdown_in_progress: bool = False


async def get_hybrid_search() -> HybridRAGSearch:
    """Get or create the global hybrid search instance."""
    global _hybrid_search, _shutdown_in_progress

    # Prevent initialization during shutdown
    if _shutdown_in_progress:
        logger.warning("[RAG] Preventing RAG initialization during shutdown")
        raise RuntimeError("RAG system initialization blocked during shutdown")

    if _hybrid_search is None:
        logger.info("[RAG] Initializing hybrid search system...")
        _hybrid_search = HybridRAGSearch()
        await _hybrid_search.initialize()

    return _hybrid_search


def set_shutdown_flag(shutdown: bool = True) -> None:
    """Set the shutdown flag to prevent RAG initialization during shutdown."""
    global _shutdown_in_progress
    _shutdown_in_progress = shutdown
    if shutdown:
        logger.info("[RAG] Shutdown flag set - blocking new RAG initialization")
    else:
        logger.info("[RAG] Shutdown flag cleared - RAG initialization allowed")


async def hybrid_search(
    query: str,
    user_id: Optional[str] = None,
    guild_id: Optional[str] = None,
    max_results: int = 5,
    search_type: str = "hybrid",
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
        search_type=search_type,
    )

    # Convert to legacy format
    legacy_results = [result.to_legacy_result() for result in hybrid_results]

    return legacy_results
