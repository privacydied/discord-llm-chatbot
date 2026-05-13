"""Async SQLite persistence for curated long-term memory."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


@dataclass(slots=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    guild_id: Optional[str]
    channel_id: Optional[str]
    thread_id: Optional[str]
    source_message_id: Optional[str]
    context_type: str
    text: str
    summary: str
    importance: float
    confidence: float
    created_at: str
    updated_at: str
    last_accessed_at: Optional[str]
    expires_at: Optional[str]
    source: str
    deleted_at: Optional[str]
    chroma_id: Optional[str]
    metadata_json: str

    @classmethod
    def from_row(cls, row: sqlite3.Row | Dict[str, Any]) -> "MemoryRecord":
        payload = dict(row)
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersistentMemoryStore:
    """SQLite control-plane store for curated memories."""

    def __init__(self, sqlite_path: str | Path):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await asyncio.to_thread(self._bootstrap_sync)
        self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _bootstrap_sync(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
                if version < 1:
                    self._create_schema(conn)
                    conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
                    conn.commit()
                    logger.info(
                        "Initialized curated memory schema",
                        extra={"subsys": "memory", "event": "memory_schema_bootstrap"},
                    )
                elif version > _SCHEMA_VERSION:
                    logger.warning(
                        "Curated memory schema is newer than this code understands",
                        extra={
                            "subsys": "memory",
                            "event": "memory_schema_version_ahead",
                            "detail": {"version": version},
                        },
                    )
            finally:
                conn.close()

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS curated_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                guild_id TEXT,
                channel_id TEXT,
                thread_id TEXT,
                source_message_id TEXT,
                context_type TEXT NOT NULL,
                text TEXT NOT NULL,
                summary TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                expires_at TEXT,
                source TEXT NOT NULL,
                deleted_at TEXT,
                chroma_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_curated_memories_user_active
                ON curated_memories(user_id, deleted_at, expires_at, updated_at);
            CREATE INDEX IF NOT EXISTS idx_curated_memories_guild_active
                ON curated_memories(guild_id, channel_id, thread_id, deleted_at, expires_at);
            CREATE INDEX IF NOT EXISTS idx_curated_memories_context_type
                ON curated_memories(context_type, deleted_at, expires_at);
            CREATE INDEX IF NOT EXISTS idx_curated_memories_last_accessed
                ON curated_memories(last_accessed_at);
            """
        )

    async def upsert_memory(self, record: MemoryRecord) -> MemoryRecord:
        await self.initialize()
        await asyncio.to_thread(self._upsert_sync, record)
        return record

    def _upsert_sync(self, record: MemoryRecord) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO curated_memories (
                        memory_id, user_id, guild_id, channel_id, thread_id,
                        source_message_id, context_type, text, summary, importance,
                        confidence, created_at, updated_at, last_accessed_at,
                        expires_at, source, deleted_at, chroma_id, metadata_json
                    ) VALUES (
                        :memory_id, :user_id, :guild_id, :channel_id, :thread_id,
                        :source_message_id, :context_type, :text, :summary, :importance,
                        :confidence, :created_at, :updated_at, :last_accessed_at,
                        :expires_at, :source, :deleted_at, :chroma_id, :metadata_json
                    )
                    ON CONFLICT(memory_id) DO UPDATE SET
                        user_id=excluded.user_id,
                        guild_id=excluded.guild_id,
                        channel_id=excluded.channel_id,
                        thread_id=excluded.thread_id,
                        source_message_id=excluded.source_message_id,
                        context_type=excluded.context_type,
                        text=excluded.text,
                        summary=excluded.summary,
                        importance=excluded.importance,
                        confidence=excluded.confidence,
                        created_at=excluded.created_at,
                        updated_at=excluded.updated_at,
                        last_accessed_at=excluded.last_accessed_at,
                        expires_at=excluded.expires_at,
                        source=excluded.source,
                        deleted_at=excluded.deleted_at,
                        chroma_id=excluded.chroma_id,
                        metadata_json=excluded.metadata_json
                    """,
                    record.to_dict(),
                )
                conn.commit()
            finally:
                conn.close()

    async def upsert_many(self, records: Sequence[MemoryRecord]) -> None:
        if not records:
            return
        await self.initialize()
        await asyncio.to_thread(self._upsert_many_sync, list(records))

    def _upsert_many_sync(self, records: Sequence[MemoryRecord]) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    """
                    INSERT INTO curated_memories (
                        memory_id, user_id, guild_id, channel_id, thread_id,
                        source_message_id, context_type, text, summary, importance,
                        confidence, created_at, updated_at, last_accessed_at,
                        expires_at, source, deleted_at, chroma_id, metadata_json
                    ) VALUES (
                        :memory_id, :user_id, :guild_id, :channel_id, :thread_id,
                        :source_message_id, :context_type, :text, :summary, :importance,
                        :confidence, :created_at, :updated_at, :last_accessed_at,
                        :expires_at, :source, :deleted_at, :chroma_id, :metadata_json
                    )
                    ON CONFLICT(memory_id) DO UPDATE SET
                        user_id=excluded.user_id,
                        guild_id=excluded.guild_id,
                        channel_id=excluded.channel_id,
                        thread_id=excluded.thread_id,
                        source_message_id=excluded.source_message_id,
                        context_type=excluded.context_type,
                        text=excluded.text,
                        summary=excluded.summary,
                        importance=excluded.importance,
                        confidence=excluded.confidence,
                        created_at=excluded.created_at,
                        updated_at=excluded.updated_at,
                        last_accessed_at=excluded.last_accessed_at,
                        expires_at=excluded.expires_at,
                        source=excluded.source,
                        deleted_at=excluded.deleted_at,
                        chroma_id=excluded.chroma_id,
                        metadata_json=excluded.metadata_json
                    """,
                    [record.to_dict() for record in records],
                )
                conn.commit()
            finally:
                conn.close()

    async def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        await self.initialize()
        return await asyncio.to_thread(self._get_memory_sync, memory_id)

    def _get_memory_sync(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM curated_memories WHERE memory_id = ?", (memory_id,)).fetchone()
                return MemoryRecord.from_row(row) if row else None
            finally:
                conn.close()

    async def list_memories(
        self,
        *,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        include_deleted: bool = False,
        limit: int = 20,
    ) -> List[MemoryRecord]:
        await self.initialize()
        return await asyncio.to_thread(
            self._list_memories_sync,
            user_id,
            guild_id,
            channel_id,
            thread_id,
            include_deleted,
            limit,
        )

    def _list_memories_sync(
        self,
        user_id: Optional[str],
        guild_id: Optional[str],
        channel_id: Optional[str],
        thread_id: Optional[str],
        include_deleted: bool,
        limit: int,
    ) -> List[MemoryRecord]:
        clauses = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(str(user_id))
        if guild_id is not None:
            clauses.append("guild_id = ?")
            params.append(str(guild_id))
        if channel_id is not None:
            clauses.append("channel_id = ?")
            params.append(str(channel_id))
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(str(thread_id))
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(self._now_iso())

        sql = "SELECT * FROM curated_memories"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY datetime(updated_at) DESC, datetime(created_at) DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [MemoryRecord.from_row(row) for row in rows]
            finally:
                conn.close()

    async def search_memories(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[MemoryRecord]:
        await self.initialize()
        return await asyncio.to_thread(
            self._search_memories_sync,
            query,
            user_id,
            guild_id,
            channel_id,
            thread_id,
            limit,
        )

    def _search_memories_sync(
        self,
        query: str,
        user_id: Optional[str],
        guild_id: Optional[str],
        channel_id: Optional[str],
        thread_id: Optional[str],
        limit: int,
    ) -> List[MemoryRecord]:
        like = f"%{query.strip()}%"
        clauses = ["deleted_at IS NULL", "(expires_at IS NULL OR expires_at > ?)"]
        params: list[Any] = [self._now_iso()]
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(str(user_id))
        if guild_id is not None:
            clauses.append("guild_id = ?")
            params.append(str(guild_id))
        if channel_id is not None:
            clauses.append("channel_id = ?")
            params.append(str(channel_id))
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(str(thread_id))
        clauses.append("(summary LIKE ? OR text LIKE ? OR metadata_json LIKE ?)")
        params.extend([like, like, like])
        params.append(int(limit))

        sql = "SELECT * FROM curated_memories WHERE " + " AND ".join(clauses) + " ORDER BY datetime(updated_at) DESC, datetime(created_at) DESC LIMIT ?"  # nosec B608
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [MemoryRecord.from_row(row) for row in rows]
            finally:
                conn.close()

    async def mark_accessed(self, memory_id: str) -> None:
        await self.initialize()
        await asyncio.to_thread(self._mark_accessed_sync, memory_id)

    def _mark_accessed_sync(self, memory_id: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE curated_memories
                    SET last_accessed_at = ?, updated_at = ?
                    WHERE memory_id = ? AND deleted_at IS NULL
                    """,
                    (self._now_iso(), self._now_iso(), memory_id),
                )
                conn.commit()
            finally:
                conn.close()

    async def soft_delete_memory(self, memory_id: str) -> bool:
        await self.initialize()
        return await asyncio.to_thread(self._soft_delete_memory_sync, memory_id)

    def _soft_delete_memory_sync(self, memory_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE curated_memories
                    SET deleted_at = ?, updated_at = ?
                    WHERE memory_id = ? AND deleted_at IS NULL
                    """,
                    (self._now_iso(), self._now_iso(), memory_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    async def wipe_user_memories(self, user_id: str) -> List[str]:
        await self.initialize()
        return await asyncio.to_thread(self._wipe_user_memories_sync, str(user_id))

    def _wipe_user_memories_sync(self, user_id: str) -> List[str]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT memory_id FROM curated_memories WHERE user_id = ? AND deleted_at IS NULL",
                    (str(user_id),),
                ).fetchall()
                ids = [str(r[0]) for r in rows]
                if ids:
                    placeholders = ",".join(["?"] * len(ids))
                    conn.execute(
                        f"""
                        UPDATE curated_memories
                        SET deleted_at = ?, updated_at = ?
                        WHERE memory_id IN ({placeholders})
                        """,  # nosec B608
                        [self._now_iso(), self._now_iso(), *ids],
                    )
                    conn.commit()
                return ids
            finally:
                conn.close()

    async def delete_by_ids(self, memory_ids: Sequence[str]) -> int:
        if not memory_ids:
            return 0
        await self.initialize()
        return await asyncio.to_thread(self._delete_by_ids_sync, list(memory_ids))

    def _delete_by_ids_sync(self, memory_ids: Sequence[str]) -> int:
        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join(["?"] * len(memory_ids))
                cur = conn.execute(
                    f"""\n                    UPDATE curated_memories\n                    SET deleted_at = ?, updated_at = ?\n                    WHERE memory_id IN ({placeholders}) AND deleted_at IS NULL\n                    """,  # nosec B608
                    [self._now_iso(), self._now_iso(), *memory_ids],
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    async def active_ids_for_user(self, user_id: str) -> List[str]:
        await self.initialize()
        return await asyncio.to_thread(self._active_ids_for_user_sync, str(user_id))

    def _active_ids_for_user_sync(self, user_id: str) -> List[str]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT memory_id FROM curated_memories
                    WHERE user_id = ? AND deleted_at IS NULL
                      AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY datetime(updated_at) DESC
                    """,
                    (str(user_id), self._now_iso()),
                ).fetchall()
                return [str(row[0]) for row in rows]
            finally:
                conn.close()

    async def active_ids_for_scope(
        self,
        *,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[str]:
        records = await self.list_memories(
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            include_deleted=False,
            limit=500,
        )
        return [record.memory_id for record in records]

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def fetch_active_by_ids(self, memory_ids: Sequence[str]) -> List[MemoryRecord]:
        if not memory_ids:
            return []
        await self.initialize()
        return await asyncio.to_thread(self._fetch_active_by_ids_sync, list(memory_ids))

    def _fetch_active_by_ids_sync(self, memory_ids: Sequence[str]) -> List[MemoryRecord]:
        placeholders = ",".join(["?"] * len(memory_ids))
        sql = f"SELECT * FROM curated_memories WHERE memory_id IN ({placeholders}) AND deleted_at IS NULL AND (expires_at IS NULL OR expires_at > ?)"  # nosec B608  # nosec B608
        params: list[Any] = [*memory_ids, self._now_iso()]
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [MemoryRecord.from_row(row) for row in rows]
            finally:
                conn.close()

    async def find_by_normalized_text(
        self,
        *,
        user_id: str,
        guild_id: Optional[str],
        normalized_text: str,
    ) -> Optional[MemoryRecord]:
        """
        Find an active memory with the same (user+guild) and identical
        normalized summary text for deduplication.
        """
        await self.initialize()
        return await asyncio.to_thread(
            self._find_by_normalized_text_sync,
            str(user_id),
            str(guild_id) if guild_id is not None else None,
            normalized_text,
        )

    def _find_by_normalized_text_sync(
        self,
        user_id: str,
        guild_id: Optional[str],
        normalized_text: str,
    ) -> Optional[MemoryRecord]:
        # Use a LIKE match on summary with normalized input for robustness.
        like = f"%{normalized_text}%"
        clauses: list[str] = [
            "deleted_at IS NULL",
            "(expires_at IS NULL OR expires_at > ?)",
            "user_id = ?",
        ]
        params: list[Any] = [self._now_iso(), user_id]

        if guild_id is not None:
            clauses.append("guild_id = ?")
            params.append(guild_id)

        # Match summary that fully contains the normalized text
        clauses.append("LOWER(summary) LIKE LOWER(?)")
        params.append(like)

        sql = "SELECT * FROM curated_memories WHERE " + " AND ".join(clauses) + " ORDER BY datetime(updated_at) DESC LIMIT 1"  # nosec B608

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(sql, params).fetchone()
                return MemoryRecord.from_row(row) if row else None
            finally:
                conn.close()
