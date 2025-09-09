"""
Session Cache System - TTL-based caching for user profiles and context with token budgets.
Implements PA (Performance Awareness) and CMV (Constants over Magic Values) rules.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from collections import OrderedDict

from .phase_constants import PhaseConstants as PC
from .phase_timing import get_timing_manager, PipelineTracker
from ..util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Individual cache entry with TTL and access tracking."""

    key: str
    data: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: int = PC.CONTEXT_CACHE_TTL_SECS
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        return time.time() - self.created_at > self.ttl_seconds

    def access(self) -> Any:
        """Access the cached data, updating stats."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.data

    def update_data(self, new_data: Any):
        """Update cached data, resetting TTL."""
        self.data = new_data
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.size_bytes = self._calculate_size(new_data)

    def _calculate_size(self, data: Any) -> int:
        """Estimate memory size of cached data [PA]."""
        try:
            import sys

            return sys.getsizeof(data)
        except Exception:
            # Fallback estimation
            if isinstance(data, str):
                return len(data) * 2  # Unicode chars
            if isinstance(data, dict):
                return sum(len(str(k)) + len(str(v)) for k, v in data.items()) * 2
            if isinstance(data, list):
                return sum(len(str(item)) for item in data) * 2
            return 1024  # Default estimate


@dataclass
class UserProfile:
    """User profile data structure."""

    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    context_summary: str = ""
    last_interaction: float = field(default_factory=time.time)
    total_messages: int = 0

    def add_message(self, role: str, content: str, timestamp: float = None):
        """Add message to conversation history with token budget [CMV]."""
        if timestamp is None:
            timestamp = time.time()

        message = {"role": role, "content": content, "timestamp": timestamp}

        self.conversation_history.append(message)
        self.total_messages += 1
        self.last_interaction = timestamp

        # Trim history based on token budget
        self._trim_history()

    def _trim_history(self):
        """Trim conversation history to fit token budget [PA]."""
        max_tokens = PC.HISTORY_MAX_TOKENS_DM
        current_chars = sum(
            len(msg.get("content", "")) for msg in self.conversation_history
        )

        # Simple token estimation: 4 chars ≈ 1 token
        estimated_tokens = current_chars // 4

        # Remove oldest messages if over budget
        while estimated_tokens > max_tokens and len(self.conversation_history) > 1:
            removed = self.conversation_history.pop(0)
            current_chars -= len(removed.get("content", ""))
            estimated_tokens = current_chars // 4

    def get_recent_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context [PA]."""
        return self.conversation_history[-max_messages:]


@dataclass
class ServerContext:
    """Server-specific context and settings."""

    guild_id: str
    server_notes: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    active_users: Dict[str, float] = field(default_factory=dict)  # user_id -> last_seen
    last_updated: float = field(default_factory=time.time)


class SessionCache:
    """High-performance session cache with TTL and LRU eviction."""

    def __init__(self, max_entries: int = 1000, default_ttl_seconds: int = None):
        self.max_entries = max_entries
        self.default_ttl = default_ttl_seconds or PC.CONTEXT_CACHE_TTL_SECS

        # Separate caches for different data types [PA]
        self.user_profiles: OrderedDict[str, CacheEntry] = OrderedDict()
        self.server_contexts: OrderedDict[str, CacheEntry] = OrderedDict()
        self.generic_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 60  # Clean every 60 seconds

        # Performance statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "expirations": 0,
            "total_size_bytes": 0,
            "avg_access_time_ms": 0,
        }

        logger.info(
            f"💾 SessionCache initialized (max_entries: {max_entries}, ttl: {self.default_ttl}s)"
        )

        # Start background cleanup
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task for expired entries [RM]."""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self._cleanup_interval)
                    await self._cleanup_expired_entries()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Cache cleanup error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def _cleanup_expired_entries(self):
        """Remove expired entries from all caches [PA]."""
        cleanup_start = time.time()
        removed_count = 0

        # Clean all cache types
        for cache_name, cache in [
            ("user_profiles", self.user_profiles),
            ("server_contexts", self.server_contexts),
            ("generic", self.generic_cache),
        ]:
            expired_keys = [key for key, entry in cache.items() if entry.is_expired()]

            for key in expired_keys:
                del cache[key]
                removed_count += 1
                self.stats["expirations"] += 1

        cleanup_time_ms = int((time.time() - cleanup_start) * 1000)

        if removed_count > 0:
            logger.debug(
                f"🗑️ Cleaned {removed_count} expired cache entries ({cleanup_time_ms}ms)"
            )

    def _evict_lru(self, cache: OrderedDict, max_size: int = None):
        """Evict least recently used entries when cache is full [PA]."""
        if max_size is None:
            max_size = self.max_entries // 3  # Each cache gets 1/3 of total capacity

        while len(cache) > max_size:
            # Remove least recently used (first in OrderedDict)
            key, entry = cache.popitem(last=False)
            self.stats["evictions"] += 1
            logger.debug(f"🗑️ Evicted LRU cache entry: {key}")

    async def get_user_profile(
        self, user_id: str, tracker: Optional[PipelineTracker] = None
    ) -> Optional[UserProfile]:
        """Get user profile from cache or return None if not found [PA]."""
        start_time = time.time()

        entry = self.user_profiles.get(user_id)
        if entry and not entry.is_expired():
            # Cache hit
            self.stats["cache_hits"] += 1

            # Move to end (most recently used)
            self.user_profiles.move_to_end(user_id)

            profile = entry.access()
            access_time_ms = int((time.time() - start_time) * 1000)

            # Track in pipeline if provided
            if tracker:
                timing_manager = get_timing_manager()
                async with timing_manager.track_phase(
                    tracker,
                    "PROFILE_CACHE_HIT",
                    user_id=user_id,
                    access_time_ms=access_time_ms,
                ):
                    pass

            logger.debug(f"✅ User profile cache HIT: {user_id} ({access_time_ms}ms)")
            return profile
        else:
            # Cache miss
            self.stats["cache_misses"] += 1

            if entry:  # Expired
                del self.user_profiles[user_id]
                self.stats["expirations"] += 1
                logger.debug(f"🕐 User profile expired: {user_id}")
            else:
                logger.debug(f"❌ User profile cache MISS: {user_id}")

            return None

    async def set_user_profile(
        self, user_id: str, profile: UserProfile, ttl_seconds: Optional[int] = None
    ):
        """Cache user profile with TTL [PA]."""
        ttl = ttl_seconds or self.default_ttl

        entry = CacheEntry(key=user_id, data=profile, ttl_seconds=ttl)
        entry.size_bytes = entry._calculate_size(profile)

        # Add to cache
        self.user_profiles[user_id] = entry

        # Evict LRU if needed
        self._evict_lru(self.user_profiles)

        logger.debug(f"💾 Cached user profile: {user_id} (ttl: {ttl}s)")

    async def get_server_context(
        self, guild_id: str, tracker: Optional[PipelineTracker] = None
    ) -> Optional[ServerContext]:
        """Get server context from cache [PA]."""
        start_time = time.time()

        entry = self.server_contexts.get(guild_id)
        if entry and not entry.is_expired():
            self.stats["cache_hits"] += 1
            self.server_contexts.move_to_end(guild_id)

            context = entry.access()
            access_time_ms = int((time.time() - start_time) * 1000)

            logger.debug(
                f"✅ Server context cache HIT: {guild_id} ({access_time_ms}ms)"
            )
            return context
        else:
            self.stats["cache_misses"] += 1

            if entry:
                del self.server_contexts[guild_id]
                self.stats["expirations"] += 1

            return None

    async def set_server_context(
        self, guild_id: str, context: ServerContext, ttl_seconds: Optional[int] = None
    ):
        """Cache server context with TTL [PA]."""
        ttl = ttl_seconds or self.default_ttl

        entry = CacheEntry(key=guild_id, data=context, ttl_seconds=ttl)

        self.server_contexts[guild_id] = entry
        self._evict_lru(self.server_contexts)

        logger.debug(f"💾 Cached server context: {guild_id} (ttl: {ttl}s)")

    async def get_generic(self, key: str) -> Optional[Any]:
        """Get generic cached data [PA]."""
        entry = self.generic_cache.get(key)
        if entry and not entry.is_expired():
            self.stats["cache_hits"] += 1
            self.generic_cache.move_to_end(key)
            return entry.access()
        else:
            self.stats["cache_misses"] += 1
            if entry:
                del self.generic_cache[key]
                self.stats["expirations"] += 1
            return None

    async def set_generic(self, key: str, data: Any, ttl_seconds: Optional[int] = None):
        """Cache generic data with TTL [PA]."""
        ttl = ttl_seconds or self.default_ttl

        entry = CacheEntry(key=key, data=data, ttl_seconds=ttl)
        self.generic_cache[key] = entry
        self._evict_lru(self.generic_cache)

    async def invalidate_user(self, user_id: str):
        """Invalidate all cached data for a user [RM]."""
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
            logger.debug(f"🗑️ Invalidated user cache: {user_id}")

    async def invalidate_server(self, guild_id: str):
        """Invalidate server context cache [RM]."""
        if guild_id in self.server_contexts:
            del self.server_contexts[guild_id]
            logger.debug(f"🗑️ Invalidated server cache: {guild_id}")

    async def update_user_interaction(
        self, user_id: str, message_content: str, role: str = "user"
    ):
        """Update user profile with new interaction [PA]."""
        profile = await self.get_user_profile(user_id)

        if profile is None:
            # Create new profile
            profile = UserProfile(user_id=user_id)

        # Add new message
        profile.add_message(role, message_content)

        # Re-cache updated profile
        await self.set_user_profile(user_id, profile)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache performance statistics [PA]."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (
            self.stats["cache_hits"] / total_requests if total_requests > 0 else 0
        )

        # Calculate total entries and estimated memory usage
        total_entries = (
            len(self.user_profiles)
            + len(self.server_contexts)
            + len(self.generic_cache)
        )

        estimated_memory_kb = (
            sum(
                entry.size_bytes
                for cache in [
                    self.user_profiles,
                    self.server_contexts,
                    self.generic_cache,
                ]
                for entry in cache.values()
            )
            // 1024
        )

        return {
            "total_entries": total_entries,
            "max_entries": self.max_entries,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "estimated_memory_kb": estimated_memory_kb,
            "cache_breakdown": {
                "user_profiles": len(self.user_profiles),
                "server_contexts": len(self.server_contexts),
                "generic": len(self.generic_cache),
            },
            "default_ttl_seconds": self.default_ttl,
            **self.stats,
        }

    async def cleanup(self):
        """Clean up resources and stop background tasks [RM]."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clear all caches
        self.user_profiles.clear()
        self.server_contexts.clear()
        self.generic_cache.clear()

        logger.debug("🧹 SessionCache cleaned up")


class ContextManager:
    """High-level context manager using session cache [PA]."""

    def __init__(self, session_cache: SessionCache):
        self.cache = session_cache

    async def get_user_context(
        self,
        user_id: str,
        tracker: Optional[PipelineTracker] = None,
        include_history: bool = True,
        max_history_messages: int = 10,
    ) -> Dict[str, Any]:
        """Get comprehensive user context for prompt building [PA]."""
        timing_manager = get_timing_manager()

        context = {
            "user_id": user_id,
            "preferences": {},
            "conversation_history": [],
            "context_summary": "",
            "cache_hit": False,
        }

        if tracker:
            async with timing_manager.track_phase(
                tracker,
                PC.PHASE_CONTEXT_GATHER,
                user_id=user_id,
                include_history=include_history,
            ) as phase_metric:
                profile = await self.cache.get_user_profile(user_id, tracker)

                if profile:
                    context["cache_hit"] = True
                    context["preferences"] = profile.preferences.copy()
                    context["context_summary"] = profile.context_summary

                    if include_history:
                        context["conversation_history"] = profile.get_recent_context(
                            max_history_messages
                        )

                    phase_metric.metadata["profile_messages"] = len(
                        profile.conversation_history
                    )
                    phase_metric.metadata["cache_hit"] = True
                else:
                    phase_metric.metadata["cache_hit"] = False
        else:
            # Direct call without tracking
            profile = await self.cache.get_user_profile(user_id)
            if profile:
                context["cache_hit"] = True
                context["preferences"] = profile.preferences.copy()
                context["context_summary"] = profile.context_summary
                if include_history:
                    context["conversation_history"] = profile.get_recent_context(
                        max_history_messages
                    )

        return context

    async def update_conversation(
        self,
        user_id: str,
        user_message: str,
        bot_response: str,
        update_preferences: Dict[str, Any] = None,
    ):
        """Update user conversation context [PA]."""
        profile = await self.cache.get_user_profile(user_id)

        if profile is None:
            profile = UserProfile(user_id=user_id)

        # Add both messages to conversation
        profile.add_message("user", user_message)
        profile.add_message("assistant", bot_response)

        # Update preferences if provided
        if update_preferences:
            profile.preferences.update(update_preferences)

        # Re-cache updated profile
        await self.cache.set_user_profile(user_id, profile)

    async def get_server_settings(self, guild_id: str) -> Dict[str, Any]:
        """Get server-specific settings and context [PA]."""
        context = await self.cache.get_server_context(guild_id)

        if context:
            return {
                "server_notes": context.server_notes,
                "settings": context.settings.copy(),
                "active_users_count": len(context.active_users),
                "last_updated": context.last_updated,
            }

        return {
            "server_notes": "",
            "settings": {},
            "active_users_count": 0,
            "last_updated": 0,
        }


# Global session cache instance [PA]
_session_cache_instance: Optional[SessionCache] = None
_context_manager_instance: Optional[ContextManager] = None


def get_session_cache() -> SessionCache:
    """Get global session cache instance."""
    global _session_cache_instance

    if _session_cache_instance is None:
        _session_cache_instance = SessionCache()
        logger.info("🚀 Global SessionCache created")

    return _session_cache_instance


def get_context_manager() -> ContextManager:
    """Get global context manager instance."""
    global _context_manager_instance

    if _context_manager_instance is None:
        cache = get_session_cache()
        _context_manager_instance = ContextManager(cache)
        logger.info("🚀 Global ContextManager created")

    return _context_manager_instance


async def cleanup_session_cache():
    """Clean up global session cache [RM]."""
    global _session_cache_instance, _context_manager_instance

    if _session_cache_instance:
        await _session_cache_instance.cleanup()
        _session_cache_instance = None

    _context_manager_instance = None
