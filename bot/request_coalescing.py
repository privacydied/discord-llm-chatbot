"""
Request coalescing for duplicate expensive operations. [PA][RM]

This module provides deduplication for concurrent identical requests
to prevent redundant work (e.g., same URL processed multiple times).

Key features:
- In-flight request deduplication: concurrent requests for same key wait for single execution
- Completed result caching: results cached briefly to prevent thundering herd
- Automatic cleanup: prevents memory leaks from stale entries
- Timeout handling: prevents stuck waiters
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Generic, Optional, TypeVar

from .utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class _CoalescedEntry(Generic[T]):
    """Internal entry for coalesced request."""
    key: str
    future: asyncio.Future[T]
    result: Optional[T] = None
    error: Optional[Exception] = None
    completed: bool = False
    created_at: float = field(default_factory=time.time)


class RequestCoalescer(Generic[T]):
    """
    Coalesces duplicate concurrent requests into single execution.
    
    When multiple coroutines request the same key simultaneously,
    only one executes while others wait for the result.
    """
    
    def __init__(
        self,
        name: str = "coalescer",
        result_ttl_s: float = 5.0,  # Cache completed results briefly
        cleanup_interval_s: float = 30.0,
    ):
        self.name = name
        self.result_ttl_s = result_ttl_s
        self.cleanup_interval_s = cleanup_interval_s
        
        # In-flight requests (key -> entry)
        self._inflight: Dict[str, _CoalescedEntry[T]] = {}
        
        # Completed results (key -> (result, timestamp))
        self._completed: Dict[str, tuple[T, float]] = {}
        self._completed_errors: Dict[str, tuple[Exception, float]] = {}
        
        # Cleanup tracking
        self._last_cleanup = time.time()
        # Lock to prevent race conditions on dict mutations
        self._lock = asyncio.Lock()
        
    def _maybe_cleanup(self) -> None:
        """Remove stale entries to prevent memory growth. [RM]"""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval_s:
            return
            
        # Clean completed results older than TTL
        cutoff = now - self.result_ttl_s
        expired = [
            key for key, (_, ts) in self._completed.items()
            if ts < cutoff
        ]
        for key in expired:
            del self._completed[key]
            
        expired_errors = [
            key for key, (_, ts) in self._completed_errors.items()
            if ts < cutoff
        ]
        for key in expired_errors:
            del self._completed_errors[key]
            
        self._last_cleanup = now
        
        if expired or expired_errors:
            logger.debug(
                f"{self.name}.cleanup | removed={len(expired)} results, {len(expired_errors)} errors"
            )
    
    async def execute(
        self,
        key: str,
        coro_factory: Callable[[], Coroutine[Any, Any, T]],
        timeout: Optional[float] = None,
    ) -> T:
        """
        Execute coroutine with coalescing.
        
        If another request with the same key is in-flight, wait for its result.
        If a recent result is cached, return it immediately.
        Otherwise, execute the coroutine and share the result with waiters.
        
        Args:
            key: Unique identifier for the operation
            coro_factory: Factory that creates the coroutine (called only once)
            timeout: Maximum time to wait for result
            
        Returns:
            Result from the coroutine execution
            
        Raises:
            asyncio.TimeoutError: If operation times out
            Exception: Any exception from the underlying coroutine
        """
        async with self._lock:
            # Check for cached result first
            if key in self._completed:
                result, ts = self._completed[key]
                if time.time() - ts <= self.result_ttl_s:
                    logger.debug(f"{self.name}.cache_hit | key={key[:50]}...")
                    return result
                else:
                    del self._completed[key]
                    
            # Check for cached error (re-raise to maintain semantics)
            if key in self._completed_errors:
                error, ts = self._completed_errors[key]
                if time.time() - ts <= self.result_ttl_s:
                    logger.debug(f"{self.name}.cache_hit_error | key={key[:50]}...")
                    raise error
                else:
                    del self._completed_errors[key]
            
            # Check if request is already in-flight
            entry = self._inflight.get(key)
            if entry is not None:
                logger.debug(f"{self.name}.wait | key={key[:50]}...")
                # Wait for the in-flight request to complete
                waiter_future = entry.future
            else:
                waiter_future = None
            
        if waiter_future is not None:
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        asyncio.shield(waiter_future), timeout=timeout
                    )
                else:
                    result = await asyncio.shield(waiter_future)
                return result
            except asyncio.TimeoutError:
                logger.warning(f"{self.name}.timeout | key={key[:50]}...")
                raise
        
        # We are the leader - execute the coroutine
        async with self._lock:
            entry = _CoalescedEntry(key=key, future=asyncio.get_event_loop().create_future())
            self._inflight[key] = entry
        
        try:
            logger.debug(f"{self.name}.execute | key={key[:50]}...")
            
            # Execute with optional timeout
            if timeout:
                result = await asyncio.wait_for(coro_factory(), timeout=timeout)
            else:
                result = await coro_factory()
            
            # Store result and notify waiters
            entry.result = result
            entry.completed = True
            entry.future.set_result(result)
            
            # Cache briefly for thundering herd protection
            async with self._lock:
                self._completed[key] = (result, time.time())
            
            return result
            
        except asyncio.TimeoutError as e:
            entry.error = e
            entry.completed = True
            entry.future.set_exception(e)
            raise
            
        except Exception as e:
            entry.error = e
            entry.completed = True
            entry.future.set_exception(e)
            # Cache errors briefly too (prevents immediate retry storm)
            async with self._lock:
                self._completed_errors[key] = (e, time.time())
            raise
            
        finally:
            # Remove from in-flight
            async with self._lock:
                self._inflight.pop(key, None)
                self._maybe_cleanup()


# Global coalescers for different operations
_url_processing_coalescer: Optional[RequestCoalescer[str]] = None
_vl_image_coalescer: Optional[RequestCoalescer[str]] = None


def get_url_processing_coalescer() -> RequestCoalescer[str]:
    """Get global URL processing coalescer. [CA]"""
    global _url_processing_coalescer
    if _url_processing_coalescer is None:
        _url_processing_coalescer = RequestCoalescer[str](
            name="url_proc",
            result_ttl_s=10.0,  # URLs can be reused within 10s
            cleanup_interval_s=60.0,
        )
    return _url_processing_coalescer


def get_vl_image_coalescer() -> RequestCoalescer[str]:
    """Get global vision-language image coalescer. [CA]"""
    global _vl_image_coalescer
    if _vl_image_coalescer is None:
        _vl_image_coalescer = RequestCoalescer[str](
            name="vl_image",
            result_ttl_s=30.0,  # VL results can be reused longer
            cleanup_interval_s=60.0,
        )
    return _vl_image_coalescer
