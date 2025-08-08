"""
Lazy loading methods for HybridRAGSearch class.

This module contains the lazy loading implementation that will be integrated
into the HybridRAGSearch class.
"""
import time
import asyncio
from typing import Optional

from .bootstrap import create_rag_system
from ..util.logging import get_logger

logger = get_logger(__name__)


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
            # Another thread is already loading, wait for it
            logger.debug("[RAG] Vector index already loading, waiting for completion...")
            
            # Wait for loading to complete (with timeout)
            start_wait = time.time()
            while (self._index_state == IndexState.LOADING and 
                   time.time() - start_wait < self.config.lazy_load_timeout):
                await asyncio.sleep(0.1)
            
            return self._index_state == IndexState.LOADED
        
        # We're the first to attempt loading
        logger.info("[RAG] Starting lazy vector index load...")
        self._index_state = IndexState.LOADING
        self._index_load_start_time = time.time()
    
    # Perform the actual loading outside the lock
    try:
        vector_load_start = time.time()
        
        # Create RAG system with full vector index loading
        self.rag_backend, self.bootstrap = await create_rag_system(
            kb_path=self.kb_path,
            db_path=self.db_path,
            config=self.config,
            load_vector_index=True
        )
        
        # Check collection status (for informational purposes only)
        stats = await self.rag_backend.get_collection_stats()
        total_chunks = stats.get("total_chunks", 0)
        if total_chunks == 0:
            logger.info("[RAG] ⚠️ Vector index loaded but collection is empty - no documents indexed yet")
            logger.info("[RAG] 💡 To populate the knowledge base, run: !rag bootstrap")
        else:
            logger.info(f"[RAG] ✅ Vector index loaded successfully with {total_chunks:,} chunks")
        
        self._vector_load_time = (time.time() - vector_load_start) * 1000
        
        # Update state to loaded
        with self._index_load_lock:
            self._index_state = IndexState.LOADED
        
        # Update metrics
        self.search_stats["first_query_lazy_loads"] += 1
        self.search_stats["lazy_load_time_ms"] = self._vector_load_time
        
        logger.info(f"[RAG] ✔ Lazy vector index load completed: "
                   f"load_time={self._vector_load_time:.1f}ms, "
                   f"total_chunks={stats.get('total_chunks', 0)}")
        
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
    
    # Trigger lazy load
    logger.info("[RAG] Vector index not loaded, triggering lazy load for first search...")
    return await self._load_vector_index()
