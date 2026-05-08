"""Coordinator for curated long-term memory storage, ingestion, and retrieval."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import load_config
from .curator import CuratedMemoryCurator, MemoryCandidate
from .ingestion_queue import CuratedMemoryIngestionQueue
from .persistent_store import MemoryRecord, PersistentMemoryStore
from .scoring import combined_score
from .semantic_store import CuratedMemorySemanticStore

logger = logging.getLogger(__name__)

_memory_service: Optional["CuratedMemoryService"] = None
_memory_service_lock = asyncio.Lock()


class CuratedMemoryService:
    """Single curated-memory pipeline; no parallel brain."""

    def __init__(self, bot: Any | None = None):
        cfg = load_config()
        self.bot = bot
        self.enabled = bool(cfg.get("PERSISTENT_MEMORY_ENABLE", True))
        self.sqlite_path = Path(cfg.get("PERSISTENT_MEMORY_SQLITE_PATH", "./data/memory.db"))
        self.chroma_path = Path(cfg.get("PERSISTENT_MEMORY_CHROMA_PATH", "./chroma_db"))
        self.collection_name = str(cfg.get("PERSISTENT_MEMORY_CHROMA_COLLECTION", "curated_memories"))
        self.top_k = int(cfg.get("PERSISTENT_MEMORY_TOP_K", 6))
        self.max_prompt_chars = int(cfg.get("PERSISTENT_MEMORY_MAX_PROMPT_CHARS", 1200))
        self.default_ttl_days = int(cfg.get("PERSISTENT_MEMORY_DEFAULT_TTL_DAYS", 180))
        self.temp_ttl_days = int(cfg.get("PERSISTENT_MEMORY_TEMP_TTL_DAYS", 14))
        self.min_importance = float(cfg.get("PERSISTENT_MEMORY_MIN_IMPORTANCE", 0.55))
        self.queue_max = int(cfg.get("PERSISTENT_MEMORY_QUEUE_MAX", 256))
        self.queue_workers = int(cfg.get("PERSISTENT_MEMORY_WORKERS", 1))
        self.curator = CuratedMemoryCurator(
            default_ttl_days=self.default_ttl_days,
            temp_ttl_days=self.temp_ttl_days,
            min_importance=self.min_importance,
        )
        self.store = PersistentMemoryStore(self.sqlite_path)
        self.semantic_store = CuratedMemorySemanticStore(
            db_path=self.chroma_path,
            collection_name=self.collection_name,
        )
        self.queue = CuratedMemoryIngestionQueue(
            persist_callback=self._persist_batch,
            max_size=self.queue_max,
            workers=self.queue_workers,
            batch_size=8,
            enabled=self.enabled,
        )
        self._started = False
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        if not self.enabled:
            return
        async with self._start_lock:
            if self._started:
                return
            await self.store.initialize()
            await self.queue.start()
            self._started = True
            logger.info(
                "Curated memory service started",
                extra={"subsys": "memory", "event": "memory_service_started"},
            )

    async def stop(self) -> None:
        if not self.enabled or not self._started:
            return
        await self.queue.stop(timeout=3.0)
        self._started = False
        logger.info(
            "Curated memory service stopped",
            extra={"subsys": "memory", "event": "memory_service_stopped"},
        )

    async def add_explicit_memory(
        self,
        *,
        user_id: str,
        text: str,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        context_type: str = "user_preference",
        source: str = "explicit_memory_command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        if not self.enabled:
            raise RuntimeError("Persistent memory is disabled")

        candidate = self.curator.build_explicit_candidate(
            user_id=user_id,
            text=text,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            source_message_id=source_message_id,
            context_type=context_type,
            source=source,
            metadata=metadata,
        )
        if candidate is None:
            raise ValueError("Memory content was rejected by the curator")

        await self._persist_candidate(candidate)
        persisted = await self.store.get_memory(candidate.memory_id)
        return persisted or candidate.to_record()

    async def enqueue_inferred_memory(
        self,
        *,
        user_id: str,
        text: str,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enabled:
            return False
        candidate = self.curator.curate_inferred_candidate(
            user_id=user_id,
            text=text,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            source_message_id=source_message_id,
            metadata=metadata,
        )
        if candidate is None:
            return False
        return await self.queue.enqueue(candidate)

    async def delete_memory(self, memory_id: str) -> bool:
        if not self.enabled:
            return False
        record = await self.store.get_memory(memory_id)
        if record is None:
            return False
        await self.store.soft_delete_memory(memory_id)
        try:
            await self.semantic_store.delete(memory_id)
        except Exception:
            logger.warning("Chroma delete failed for memory %s", memory_id, exc_info=True)
        return True

    async def wipe_user_memories(self, user_id: str) -> int:
        if not self.enabled:
            return 0
        ids = await self.store.wipe_user_memories(user_id)
        if ids:
            try:
                await self.semantic_store.delete_many(ids)
            except Exception:
                logger.warning("Chroma wipe failed for user %s", user_id, exc_info=True)
        return len(ids)

    async def list_user_memories(self, user_id: str, limit: int = 20) -> List[MemoryRecord]:
        if not self.enabled:
            return []
        return await self.store.list_memories(user_id=user_id, limit=limit)

    async def search_user_memories(
        self,
        user_id: str,
        query: str,
        *,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 8,
    ) -> List[MemoryRecord]:
        if not self.enabled:
            return []
        query = (query or "").strip()
        if not query:
            return []
        results = await self.semantic_search(
            query,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            top_k=limit,
        )
        ids = [result["memory_id"] for result in results]
        if not ids:
            return []
        records = await self.store.fetch_active_by_ids(ids)
        by_id = {record.memory_id: record for record in records}
        return [by_id[mid] for mid in ids if mid in by_id]

    async def semantic_search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        top_k = max(3, min(8, int(top_k or self.top_k)))
        scope_filters = self._scope_filters(
            user_id=user_id, guild_id=guild_id, channel_id=channel_id, thread_id=thread_id
        )
        if not scope_filters:
            return []

        combined: dict[str, Dict[str, Any]] = {}
        for scope_name, where, boost in scope_filters:
            try:
                results = await self.semantic_store.query(query, top_k=top_k, where=where)
            except Exception:
                logger.debug("Semantic query failed for scope %s", scope_name, exc_info=True)
                continue
            for item in results:
                metadata = item.get("metadata") or {}
                if not self._scope_allows(
                    metadata,
                    scope_name,
                    user_id=user_id,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    thread_id=thread_id,
                ):
                    continue
                memory_id = item.get("memory_id")
                if not memory_id:
                    continue
                record = combined.get(memory_id)
                semantic_score = float(item.get("semantic_score", 0.0))
                importance = float(metadata.get("importance", 0.5) or 0.5)
                created_at = metadata.get("created_at")
                last_accessed_at = metadata.get("last_accessed_at")
                expires_at = metadata.get("expires_at")
                score = combined_score(
                    semantic_score,
                    importance,
                    created_at,
                    last_accessed_at=last_accessed_at,
                    expires_at=expires_at,
                    scope_boost=boost,
                )
                if record is None or score > record["score"]:
                    combined[memory_id] = {
                        "memory_id": memory_id,
                        "document": item.get("document", ""),
                        "metadata": metadata,
                        "score": score,
                        "semantic_score": semantic_score,
                        "scope": scope_name,
                    }

        records = await self.store.fetch_active_by_ids(list(combined.keys()))
        record_map = {record.memory_id: record for record in records}
        final: list[Dict[str, Any]] = []
        for memory_id, payload in combined.items():
            record = record_map.get(memory_id)
            if not record:
                continue
            if record.confidence < 0.45:
                continue
            final.append(
                {
                    **payload,
                    "record": record.to_dict(),
                    "score": payload["score"],
                }
            )

        final.sort(key=lambda item: item["score"], reverse=True)
        return final[:top_k]

    async def build_prompt_block(
        self,
        *,
        user_id: Optional[str],
        guild_id: Optional[str],
        channel_id: Optional[str],
        thread_id: Optional[str],
        query: str,
        max_chars: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> str:
        if not self.enabled or not user_id:
            return ""

        results = await self.semantic_search(
            query,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            top_k=top_k,
        )
        if not results:
            return ""

        max_chars = int(max_chars or self.max_prompt_chars)
        lines: list[str] = ["Relevant long-term memory:"]
        used = len(lines[0]) + 1
        for item in results:
            record = item["record"]
            summary = str(record.get("summary") or item.get("document") or "").strip()
            if not summary:
                continue
            line = f"- {summary}"
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines) if len(lines) > 1 else ""

    async def _persist_batch(self, candidates: List[MemoryCandidate]) -> None:
        if not candidates:
            return
        await self.store.initialize()
        await self.semantic_store.initialize()
        for candidate in candidates:
            record = candidate.to_record()
            try:
                chroma_id = await self.semantic_store.upsert(
                    candidate.memory_id,
                    candidate.summary,
                    {
                        "memory_id": candidate.memory_id,
                        "user_id": candidate.user_id,
                        "guild_id": candidate.guild_id,
                        "channel_id": candidate.channel_id,
                        "thread_id": candidate.thread_id,
                        "context_type": candidate.context_type,
                        "created_at": candidate.created_at,
                        "expires_at": candidate.expires_at,
                        "importance": candidate.importance,
                        "confidence": candidate.confidence,
                        "source": candidate.source,
                    },
                )
                record.chroma_id = chroma_id
                await self.store.upsert_memory(record)
            except Exception:
                logger.exception(
                    "Failed to persist curated memory to Chroma",
                    extra={"memory_id": candidate.memory_id},
                )
                raise

    async def _persist_candidate(self, candidate: MemoryCandidate) -> None:
        await self._persist_batch([candidate])

    def _scope_allows(
        self,
        metadata: Dict[str, Any],
        scope_name: str,
        *,
        user_id: Optional[str],
        guild_id: Optional[str],
        channel_id: Optional[str],
        thread_id: Optional[str],
    ) -> bool:
        if scope_name == "user":
            return user_id is None or str(metadata.get("user_id") or "") == str(user_id)
        if scope_name == "guild":
            if guild_id is None or str(metadata.get("guild_id") or "") != str(guild_id):
                return False
            meta_channel = metadata.get("channel_id")
            meta_thread = metadata.get("thread_id")
            if meta_thread not in (None, "", thread_id):
                return False
            if meta_channel not in (None, "", channel_id):
                return False
            return True
        if scope_name == "channel":
            return channel_id is not None and str(metadata.get("channel_id") or "") == str(channel_id)
        if scope_name == "thread":
            return thread_id is not None and str(metadata.get("thread_id") or "") == str(thread_id)
        return True

    def _scope_filters(
        self,
        *,
        user_id: Optional[str],
        guild_id: Optional[str],
        channel_id: Optional[str],
        thread_id: Optional[str],
    ) -> list[tuple[str, Dict[str, Any], float]]:
        filters: list[tuple[str, Dict[str, Any], float]] = []
        if user_id is not None:
            filters.append(("user", {"user_id": str(user_id)}, 0.04))
        if guild_id is not None:
            filters.append(("guild", {"guild_id": str(guild_id)}, 0.06))
        if channel_id is not None:
            filters.append(("channel", {"channel_id": str(channel_id)}, 0.08))
        if thread_id is not None:
            filters.append(("thread", {"thread_id": str(thread_id)}, 0.1))
        return filters


async def get_memory_service(bot: Any | None = None) -> CuratedMemoryService:
    global _memory_service

    if _memory_service is not None:
        if bot is not None:
            _memory_service.bot = bot
        return _memory_service

    async with _memory_service_lock:
        if _memory_service is None:
            _memory_service = CuratedMemoryService(bot=bot)
            if _memory_service.enabled:
                await _memory_service.start()
    if bot is not None:
        _memory_service.bot = bot
    return _memory_service


async def start_memory_service(bot: Any | None = None) -> CuratedMemoryService:
    service = await get_memory_service(bot)
    await service.start()
    return service


async def stop_memory_service() -> None:
    global _memory_service
    if _memory_service is None:
        return
    await _memory_service.stop()


async def enqueue_inferred_memory(**kwargs: Any) -> bool:
    service = await get_memory_service()
    return await service.enqueue_inferred_memory(**kwargs)


async def add_explicit_memory(**kwargs: Any) -> MemoryRecord:
    service = await get_memory_service()
    return await service.add_explicit_memory(**kwargs)


async def build_memory_prompt_block(**kwargs: Any) -> str:
    service = await get_memory_service()
    return await service.build_prompt_block(**kwargs)


async def list_user_memories(user_id: str, limit: int = 20) -> List[MemoryRecord]:
    service = await get_memory_service()
    return await service.list_user_memories(user_id, limit=limit)


async def delete_memory(memory_id: str) -> bool:
    service = await get_memory_service()
    return await service.delete_memory(memory_id)


async def wipe_user_memories(user_id: str) -> int:
    service = await get_memory_service()
    return await service.wipe_user_memories(user_id)


async def search_user_memories(
    user_id: str,
    query: str,
    *,
    guild_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    limit: int = 8,
) -> List[MemoryRecord]:
    service = await get_memory_service()
    return await service.search_user_memories(
        user_id,
        query,
        guild_id=guild_id,
        channel_id=channel_id,
        thread_id=thread_id,
        limit=limit,
    )
