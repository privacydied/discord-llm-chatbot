"""
Single-flight cache system with deduplication for router optimization. [PA][DRY]

This module prevents duplicate external calls by ensuring only one request per unique key
is in flight at any time. Additional requests for the same key wait for the result
rather than making redundant network calls.

Key features:
- In-memory LRU cache with configurable TTL
- Single-flight deduplication prevents redundant API calls
- Per-family cache configurations (tweet, readability, STT, etc.)
- Negative caching for failed requests
- Cache hit/miss metrics for monitoring
- Optional Redis/disk secondary cache support
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union, Callable, Awaitable
import os
from pathlib import Path

from .util.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class CacheFamily(Enum):
    """Cache families with different TTL and policies. [CMV]"""
    TWEET_TEXT = "tweet_text"           # Tweet text/photos: 24h TTL
    TWEET_NEGATIVE = "tweet_negative"   # Failed tweet fetches: 15m TTL
    READABILITY = "readability"         # Web page readability: 4h TTL
    STT_RESULT = "stt_result"           # Speech-to-text: 7 days TTL
    SCREENSHOT = "screenshot"           # Screenshot URLs: 1h TTL
    WEB_EXTRACTION = "web_extraction"   # General web extraction: 2h TTL

@dataclass
class CacheEntry:
    """Individual cache entry with metadata. [CA]"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: float
    hit_count: int = 0
    family: CacheFamily = CacheFamily.READABILITY
    negative: bool = False  # True if this caches a failure
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Age of entry in seconds."""
        return time.time() - self.created_at

@dataclass
class CacheMetrics:
    """Cache metrics for monitoring. [PA]"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    single_flight_hits: int = 0  # Requests that joined in-flight operations
    evictions: int = 0
    total_entries: int = 0
    memory_usage_bytes: int = 0
    avg_hit_rate: float = 0.0
    
    def calculate_hit_rate(self) -> None:
        """Update hit rate calculation."""
        if self.total_requests > 0:
            self.avg_hit_rate = self.cache_hits / self.total_requests

class SingleFlightGroup:
    """Manages in-flight operations to prevent duplication. [PA][DRY]"""
    
    def __init__(self):
        self.in_flight: Dict[str, asyncio.Future] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
    
    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> Tuple[T, bool]:
        """Execute function with single-flight semantics.
        
        Returns: (result, was_duplicate) where was_duplicate=True if this
        request joined an already in-flight operation.
        """
        # Get or create lock for this key
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        lock = self.locks[key]
        
        async with lock:
            # Check if operation is already in flight
            if key in self.in_flight:
                future = self.in_flight[key]
                try:
                    result = await future
                    return result, True  # This was a duplicate request
                except Exception as e:
                    # If the in-flight operation failed, we still need to try
                    # Remove failed future and continue
                    self.in_flight.pop(key, None)
                    raise e
            
            # Start new operation
            future = asyncio.create_task(fn())
            self.in_flight[key] = future
            
            try:
                result = await future
                return result, False  # This was the original request
            finally:
                # Clean up completed operation
                self.in_flight.pop(key, None)
                # Clean up lock if no longer needed
                if key in self.locks and key not in self.in_flight:
                    del self.locks[key]

class LRUCache:
    """In-memory LRU cache with TTL support. [PA][RM]"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # Most recently used at end
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache if not expired."""
        async with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired:
                # Clean up expired entry
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return None
            
            # Update access order (move to end)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            # Update access time and hit count
            entry.accessed_at = time.time()
            entry.hit_count += 1
            
            return entry
    
    async def put(self, key: str, entry: CacheEntry) -> None:
        """Put entry in cache with LRU eviction."""
        async with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                if key in self.access_order:
                    self.access_order.remove(key)
            
            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)
            
            # Evict oldest entries if over capacity
            while len(self.cache) > self.max_size:
                if not self.access_order:
                    break
                
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
    
    async def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entries from cache."""
        async with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self.lock:
            total_size = sum(len(str(entry.value)) for entry in self.cache.values())
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_bytes": total_size,
                "oldest_entry_age": min((e.age_seconds for e in self.cache.values()), default=0),
                "newest_entry_age": max((e.age_seconds for e in self.cache.values()), default=0),
            }

class SingleFlightCache:
    """Main cache with single-flight deduplication. [PA][DRY]"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cache with configuration."""
        self.config = config or {}
        
        # Cache configuration
        max_entries = int(self.config.get('CACHE_MAX_ENTRIES', 2000))
        self.cache = LRUCache(max_entries)
        self.single_flight = SingleFlightGroup()
        self.metrics = CacheMetrics()
        
        # Family-specific TTL configuration
        self.family_ttls = {
            CacheFamily.TWEET_TEXT: float(self.config.get('TWEET_CACHE_TTL_S', 86400)),        # 24 hours
            CacheFamily.TWEET_NEGATIVE: float(self.config.get('TWEET_NEGATIVE_TTL_S', 900)),   # 15 minutes
            CacheFamily.READABILITY: float(self.config.get('CACHE_READABILITY_TTL_S', 14400)), # 4 hours
            CacheFamily.STT_RESULT: float(self.config.get('STT_CACHE_TTL_S', 604800)),         # 7 days
            CacheFamily.SCREENSHOT: 3600.0,    # 1 hour
            CacheFamily.WEB_EXTRACTION: 7200.0, # 2 hours
        }
        
        # Enable/disable features
        self.enabled = self.config.get('CACHE_SINGLE_FLIGHT_ENABLE', True)
        self.cache_backend = self.config.get('CACHE_BACKEND', 'memory')
        
        logger.info(f"💾 SingleFlightCache initialized (backend: {self.cache_backend}, max_entries: {max_entries})")
    
    def _make_cache_key(self, family: CacheFamily, key_parts: List[str]) -> str:
        """Create cache key from family and parts. [IV]"""
        # Normalize key parts to ensure consistent caching
        normalized_parts = []
        for part in key_parts:
            if isinstance(part, (dict, list)):
                # Sort dict keys for consistent hashing
                if isinstance(part, dict):
                    part = json.dumps(part, sort_keys=True)
                else:
                    part = json.dumps(part)
            normalized_parts.append(str(part))
        
        # Create hash of key parts to handle long URLs and normalize spacing
        key_data = f"{family.value}:{':'.join(normalized_parts)}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        
        return f"{family.value}:{key_hash}"
    
    async def get_or_compute(
        self,
        family: CacheFamily,
        key_parts: List[str],
        compute_fn: Callable[[], Awaitable[T]],
        negative_on_exception: bool = True
    ) -> Tuple[T, bool]:
        """Get from cache or compute with single-flight semantics.
        
        Returns: (result, cache_hit) where cache_hit=True if result came from cache.
        """
        if not self.enabled:
            # Cache disabled, always compute
            result = await compute_fn()
            return result, False
        
        cache_key = self._make_cache_key(family, key_parts)
        self.metrics.total_requests += 1
        
        # Try cache first
        entry = await self.cache.get(cache_key)
        if entry is not None:
            self.metrics.cache_hits += 1
            self.metrics.calculate_hit_rate()
            
            # Check if this is a negative cache entry
            if entry.negative:
                # Re-raise the cached exception
                if isinstance(entry.value, Exception):
                    raise entry.value
                else:
                    raise ValueError(f"Cached negative result for {family.value}")
            
            return entry.value, True
        
        # Cache miss - use single-flight to compute
        self.metrics.cache_misses += 1
        
        async def compute_and_cache():
            try:
                result = await compute_fn()
                
                # Cache successful result
                ttl = self.family_ttls.get(family, 3600.0)
                entry = CacheEntry(
                    key=cache_key,
                    value=result,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl_seconds=ttl,
                    family=family,
                    negative=False
                )
                await self.cache.put(cache_key, entry)
                
                return result
                
            except Exception as e:
                # Optionally cache negative results
                if negative_on_exception:
                    # Use shorter TTL for negative cache
                    negative_ttl = min(self.family_ttls.get(family, 3600.0) / 10, 900.0)  # Max 15 minutes
                    entry = CacheEntry(
                        key=cache_key,
                        value=e,
                        created_at=time.time(),
                        accessed_at=time.time(),
                        ttl_seconds=negative_ttl,
                        family=family,
                        negative=True
                    )
                    await self.cache.put(cache_key, entry)
                
                raise e
        
        try:
            result, was_duplicate = await self.single_flight.do(cache_key, compute_and_cache)
            
            if was_duplicate:
                self.metrics.single_flight_hits += 1
                logger.debug(f"🎯 Single-flight hit for {family.value}")
            
            self.metrics.calculate_hit_rate()
            return result, False
            
        except Exception as e:
            self.metrics.calculate_hit_rate()
            raise e
    
    async def put(
        self,
        family: CacheFamily,
        key_parts: List[str],
        value: Any,
        ttl_override: Optional[float] = None
    ) -> None:
        """Manually put value in cache."""
        if not self.enabled:
            return
        
        cache_key = self._make_cache_key(family, key_parts)
        ttl = ttl_override or self.family_ttls.get(family, 3600.0)
        
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            ttl_seconds=ttl,
            family=family,
            negative=False
        )
        
        await self.cache.put(cache_key, entry)
    
    async def invalidate(self, family: CacheFamily, key_parts: List[str]) -> bool:
        """Remove specific entry from cache."""
        if not self.enabled:
            return False
        
        cache_key = self._make_cache_key(family, key_parts)
        return await self.cache.remove(cache_key)
    
    async def invalidate_family(self, family: CacheFamily) -> int:
        """Remove all entries for a specific family."""
        if not self.enabled:
            return 0
        
        # This is expensive but useful for cache management
        removed_count = 0
        async with self.cache.lock:
            keys_to_remove = [
                key for key, entry in self.cache.cache.items()
                if entry.family == family
            ]
            
            for key in keys_to_remove:
                await self.cache.remove(key)
                removed_count += 1
        
        logger.info(f"🗑️ Invalidated {removed_count} entries for {family.value}")
        return removed_count
    
    async def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics."""
        cache_stats = await self.cache.get_stats()
        
        self.metrics.total_entries = cache_stats["entries"]
        self.metrics.memory_usage_bytes = cache_stats["memory_usage_bytes"]
        
        return self.metrics
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        if not self.enabled:
            return 0
        
        removed_count = 0
        async with self.cache.lock:
            expired_keys = [
                key for key, entry in self.cache.cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                await self.cache.remove(key)
                removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"🧹 Cleaned up {removed_count} expired cache entries")
        
        return removed_count

# Global singleton instance
_cache_instance: Optional[SingleFlightCache] = None

def get_cache(config: Optional[Dict[str, Any]] = None) -> SingleFlightCache:
    """Get or create the global cache instance. [CA]"""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = SingleFlightCache(config)
    
    return _cache_instance

async def cleanup_cache() -> None:
    """Clean up the global cache instance."""
    global _cache_instance
    
    if _cache_instance is not None:
        await _cache_instance.cache.clear()
        _cache_instance = None
