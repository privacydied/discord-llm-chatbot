"""Unified SQLite store for guild + DM messages.

Provides a single schema for all bot-visible messages (guild text channels,
threads, and DM conversations). Thread-safe with WAL mode, all I/O dispatched
through asyncio.to_thread so the event loop is never blocked.

Schema versioning follows the same pattern as AuditStore and DMStore.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from bot.utils.logging import get_logger, redact_sensitive_values

logger = get_logger(__name__)

# Max chars for content_redacted (preview) column
_CONTENT_REDACTED_CHARS = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _make_content_redacted(text: str, max_chars: int = _CONTENT_REDACTED_CHARS) -> str:
    """Redact sensitive values from content and truncate for storage."""
    cleaned = redact_sensitive_values(text or "")
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "..."
    return cleaned


def _to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _from_json(text: Optional[str]) -> Any:
    if not text:
        return {} if text is not None else None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


class MessageStore:
    """Thread-safe SQLite store for bot-visible messages (guild + DM).

    Stores every message the bot sees — inbound and outbound — in a single
    ``messages`` table and tracks DM thread metadata in a ``dm_threads`` table.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, db_path: str, retention_days: int = 90) -> None:
        self._db_path = db_path
        self._retention_days = retention_days
        self._lock = threading.RLock()
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        await asyncio.to_thread(self._bootstrap_sync)
        self._initialized = True

    def _bootstrap_sync(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            if version < 1:
                self._create_schema_v1(conn)
            conn.execute(f"PRAGMA user_version={self._SCHEMA_VERSION}")
            conn.commit()
        finally:
            conn.close()

    def _create_schema_v1(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                discord_message_id TEXT NOT NULL,
                guild_id TEXT,
                channel_id TEXT NOT NULL,
                channel_name TEXT,
                channel_type TEXT,
                author_id TEXT NOT NULL,
                author_username TEXT,
                author_display_name TEXT,
                author_avatar_url TEXT,
                author_is_bot INTEGER NOT NULL DEFAULT 0,
                is_own_bot INTEGER NOT NULL DEFAULT 0,
                direction TEXT NOT NULL DEFAULT 'inbound',
                content TEXT NOT NULL DEFAULT '',
                content_redacted TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                edited_at TEXT,
                deleted_at TEXT,
                reply_to_message_id TEXT,
                attachments_json TEXT NOT NULL DEFAULT '[]',
                embeds_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                inserted_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_threads (
                dm_channel_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                username TEXT,
                display_name TEXT,
                avatar_url TEXT,
                last_message_id TEXT,
                last_message_at TEXT,
                message_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        # Indexes for messages
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_channel_time ON messages(channel_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_guild_channel_time ON messages(guild_id, channel_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_author_time ON messages(author_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_discord_id ON messages(discord_message_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_created_at ON messages(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_direction ON messages(direction, created_at)")

        # Index for dm_threads
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_thread_last_at ON dm_threads(last_message_at)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Insert / Upsert
    # ------------------------------------------------------------------

    async def insert_message(
        self,
        discord_message_id: int,
        channel_id: int,
        content: str = "",
        guild_id: Optional[int] = None,
        channel_name: Optional[str] = None,
        channel_type: Optional[str] = None,
        author_id: Optional[int] = None,
        author_username: Optional[str] = None,
        author_display_name: Optional[str] = None,
        author_avatar_url: Optional[str] = None,
        author_is_bot: bool = False,
        is_own_bot: bool = False,
        direction: str = "inbound",
        created_at: Optional[str] = None,
        edited_at: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
        embeds: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Insert a message, returning True if inserted, False if duplicate."""
        await self.initialize()
        return await asyncio.to_thread(
            self._insert_message_sync,
            discord_message_id=str(discord_message_id),
            channel_id=str(channel_id),
            content=content,
            guild_id=str(guild_id) if guild_id else None,
            channel_name=channel_name,
            channel_type=channel_type,
            author_id=str(author_id) if author_id else None,
            author_username=author_username,
            author_display_name=author_display_name,
            author_avatar_url=author_avatar_url,
            author_is_bot=int(author_is_bot),
            is_own_bot=int(is_own_bot),
            direction=direction,
            created_at=created_at or _now_iso(),
            edited_at=edited_at,
            reply_to_message_id=str(reply_to_message_id) if reply_to_message_id else None,
            attachments_json=_to_json(attachments or []),
            embeds_json=_to_json(embeds or []),
            metadata_json=_to_json(metadata or {}),
        )

    def _insert_message_sync(
        self,
        discord_message_id: str,
        channel_id: str,
        content: str,
        guild_id: Optional[str],
        channel_name: Optional[str],
        channel_type: Optional[str],
        author_id: Optional[str],
        author_username: Optional[str],
        author_display_name: Optional[str],
        author_avatar_url: Optional[str],
        author_is_bot: int,
        is_own_bot: int,
        direction: str,
        created_at: str,
        edited_at: Optional[str],
        reply_to_message_id: Optional[str],
        attachments_json: str,
        embeds_json: str,
        metadata_json: str,
    ) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                content_redacted = _make_content_redacted(content)
                now = _now_iso()

                cursor = conn.execute(
                    """INSERT OR IGNORE INTO messages (
                        discord_message_id, guild_id, channel_id, channel_name,
                        channel_type, author_id, author_username, author_display_name,
                        author_avatar_url, author_is_bot, is_own_bot, direction,
                        content, content_redacted, created_at, edited_at,
                        reply_to_message_id, attachments_json, embeds_json,
                        metadata_json, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        discord_message_id,
                        guild_id,
                        channel_id,
                        channel_name,
                        channel_type,
                        author_id,
                        author_username,
                        author_display_name,
                        author_avatar_url,
                        author_is_bot,
                        is_own_bot,
                        direction,
                        content,
                        content_redacted,
                        created_at,
                        edited_at,
                        reply_to_message_id,
                        attachments_json,
                        embeds_json,
                        metadata_json,
                        now,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def get_message_by_discord_id(self, discord_message_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a single message by its Discord message ID."""
        await self.initialize()
        return await asyncio.to_thread(self._get_message_by_discord_id_sync, str(discord_message_id))

    def _get_message_by_discord_id_sync(self, discord_message_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM messages WHERE discord_message_id = ? LIMIT 1",
                    (discord_message_id,),
                ).fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_channel_messages(
        self,
        channel_id: int,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
        before_id: Optional[int] = None,
        after_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get messages for a channel, newest first, with cursor-style pagination."""
        await self.initialize()
        page = max(1, page)
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size

        rows, total = await asyncio.to_thread(
            self._get_channel_messages_sync,
            channel_id=str(channel_id),
            page_size=page_size,
            offset=offset,
            before_id=str(before_id) if before_id else None,
            after_id=str(after_id) if after_id else None,
        )

        messages = [self._row_to_dict(r) for r in rows]
        total_pages = max(1, (total + page_size - 1) // page_size)

        return {
            "messages": messages,
            "channel_id": str(channel_id),
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        }

    def _get_channel_messages_sync(
        self,
        channel_id: str,
        page_size: int,
        offset: int,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
    ) -> tuple[list, int]:
        with self._lock:
            conn = self._connect()
            try:
                where_parts = ["channel_id = ? AND deleted_at IS NULL"]
                params: list = [channel_id]

                if before_id:
                    where_parts.append("row_id < (SELECT row_id FROM messages WHERE discord_message_id = ?)")
                    params.append(before_id)
                if after_id:
                    where_parts.append("row_id > (SELECT row_id FROM messages WHERE discord_message_id = ?)")
                    params.append(after_id)

                where_sql = " AND ".join(where_parts)
                count = conn.execute(
                    f"SELECT COUNT(*) FROM messages WHERE {where_sql}",  # nosec B608
                    params,
                ).fetchone()[0]

                rows = conn.execute(
                    f"SELECT * FROM messages WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # nosec B608
                    params + [page_size, offset],
                ).fetchall()
                return rows, count
            finally:
                conn.close()

    async def search_messages(
        self,
        query: str,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
        guild_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        author_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Simple LIKE-based search across message content."""
        await self.initialize()
        page = max(1, page)
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size

        rows, total = await asyncio.to_thread(
            self._search_messages_sync,
            query=query,
            page_size=page_size,
            offset=offset,
            guild_id=str(guild_id) if guild_id else None,
            channel_id=str(channel_id) if channel_id else None,
            author_id=str(author_id) if author_id else None,
        )

        messages = [self._row_to_dict(r) for r in rows]
        total_pages = max(1, (total + page_size - 1) // page_size)

        return {
            "messages": messages,
            "query": query,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        }

    def _search_messages_sync(
        self,
        query: str,
        page_size: int,
        offset: int,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        author_id: Optional[str] = None,
    ) -> tuple[list, int]:
        with self._lock:
            conn = self._connect()
            try:
                like_pattern = f"%{query}%"
                where_parts = ["(content LIKE ? OR content_redacted LIKE ?) AND deleted_at IS NULL"]
                params: list = [like_pattern, like_pattern]

                if guild_id:
                    where_parts.append("guild_id = ?")
                    params.append(guild_id)
                if channel_id:
                    where_parts.append("channel_id = ?")
                    params.append(channel_id)
                if author_id:
                    where_parts.append("author_id = ?")
                    params.append(author_id)

                where_sql = " AND ".join(where_parts)
                count = conn.execute(
                    f"SELECT COUNT(*) FROM messages WHERE {where_sql}",  # nosec B608
                    params,
                ).fetchone()[0]

                rows = conn.execute(
                    f"SELECT * FROM messages WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # nosec B608
                    params + [page_size, offset],
                ).fetchall()
                return rows, count
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # DM Threads
    # ------------------------------------------------------------------

    async def upsert_dm_thread(
        self,
        dm_channel_id: int,
        user_id: int,
        username: Optional[str] = None,
        display_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        last_message_id: Optional[int] = None,
        last_message_at: Optional[str] = None,
        increment_count: bool = True,
    ) -> None:
        """Upsert a DM thread entry, optionally incrementing message_count."""
        await self.initialize()
        await asyncio.to_thread(
            self._upsert_dm_thread_sync,
            dm_channel_id=str(dm_channel_id),
            user_id=str(user_id),
            username=username,
            display_name=display_name,
            avatar_url=avatar_url,
            last_message_id=str(last_message_id) if last_message_id else None,
            last_message_at=last_message_at or _now_iso(),
            increment_count=increment_count,
        )

    def _upsert_dm_thread_sync(
        self,
        dm_channel_id: str,
        user_id: str,
        username: Optional[str],
        display_name: Optional[str],
        avatar_url: Optional[str],
        last_message_id: Optional[str],
        last_message_at: str,
        increment_count: bool,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                count_expr = "message_count + 1" if increment_count else "excluded.message_count"
                conn.execute(  # nosec B608 - parameterized query with safe values
                    """INSERT INTO dm_threads (
                        dm_channel_id, user_id, username, display_name, avatar_url,
                        last_message_id, last_message_at, message_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(dm_channel_id) DO UPDATE SET
                        user_id=excluded.user_id,
                        username=COALESCE(excluded.username, dm_threads.username),
                        display_name=COALESCE(excluded.display_name, dm_threads.display_name),
                        avatar_url=COALESCE(excluded.avatar_url, dm_threads.avatar_url),
                        last_message_id=COALESCE(excluded.last_message_id, dm_threads.last_message_id),
                        last_message_at=excluded.last_message_at,
                        message_count="""  # nosec B608
                    + count_expr
                    + """ """,
                    (
                        dm_channel_id,
                        user_id,
                        username,
                        display_name,
                        avatar_url,
                        last_message_id,
                        last_message_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def get_dm_threads(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
    ) -> dict[str, Any]:
        """List DM threads, ordered by most recent activity."""
        await self.initialize()
        page = max(1, page)
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size

        rows, total = await asyncio.to_thread(self._get_dm_threads_sync, page_size=page_size, offset=offset)

        threads = []
        for row in rows:
            threads.append(
                {
                    "dm_channel_id": row["dm_channel_id"],
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "display_name": row["display_name"],
                    "avatar_url": row["avatar_url"],
                    "last_message_id": row["last_message_id"],
                    "last_message_at": row["last_message_at"],
                    "message_count": row["message_count"],
                }
            )

        total_pages = max(1, (total + page_size - 1) // page_size)
        return {
            "threads": threads,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        }

    def _get_dm_threads_sync(self, page_size: int, offset: int) -> tuple[list, int]:
        with self._lock:
            conn = self._connect()
            try:
                total = conn.execute("SELECT COUNT(*) FROM dm_threads").fetchone()[0]
                rows = conn.execute(
                    """SELECT * FROM dm_threads
                    ORDER BY last_message_at DESC
                    LIMIT ? OFFSET ?""",
                    (page_size, offset),
                ).fetchall()
                return rows, total
            finally:
                conn.close()

    async def get_dm_thread_messages(
        self,
        dm_channel_id: int,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
    ) -> dict[str, Any]:
        """Get messages for a specific DM thread, newest first."""
        await self.initialize()
        page = max(1, page)
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size

        rows, total = await asyncio.to_thread(
            self._get_dm_thread_messages_sync,
            dm_channel_id=str(dm_channel_id),
            page_size=page_size,
            offset=offset,
        )

        messages = [self._row_to_dict(r) for r in rows]
        total_pages = max(1, (total + page_size - 1) // page_size)

        return {
            "messages": messages,
            "dm_channel_id": str(dm_channel_id),
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        }

    def _get_dm_thread_messages_sync(self, dm_channel_id: str, page_size: int, offset: int) -> tuple[list, int]:
        with self._lock:
            conn = self._connect()
            try:
                total = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE channel_id = ? AND deleted_at IS NULL",
                    (dm_channel_id,),
                ).fetchone()[0]

                rows = conn.execute(
                    """SELECT * FROM messages
                    WHERE channel_id = ? AND deleted_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?""",
                    (dm_channel_id, page_size, offset),
                ).fetchall()
                return rows, total
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    async def cleanup_retention(self) -> int:
        """Soft-delete messages older than retention period. Returns count."""
        await self.initialize()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return await asyncio.to_thread(self._cleanup_sync, cutoff)

    def _cleanup_sync(self, cutoff: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                now = _now_iso()
                cur = conn.execute(
                    "UPDATE messages SET deleted_at = ? WHERE created_at < ? AND deleted_at IS NULL",
                    (now, cutoff),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict, deserializing JSON columns."""
        d = dict(row)
        # Deserialize JSON columns for convenience
        for json_col in ("attachments_json", "embeds_json", "metadata_json"):
            d[json_col] = _from_json(d.get(json_col))
        # Normalize boolean fields
        for bool_col in ("author_is_bot", "is_own_bot"):
            if bool_col in d:
                d[bool_col] = bool(d[bool_col])
        return d
