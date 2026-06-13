"""DM archive for bot-visible DM conversations only.

Stores only messages where the bot is a direct participant (DM channel).
Does NOT scrape private user data or read DMs outside bot visibility.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bot.utils.logging import get_logger, redact_sensitive_values

logger = get_logger(__name__)

_MAX_PREVIEW_CHARS = 150


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _make_preview(content: str, max_chars: int = _MAX_PREVIEW_CHARS) -> str:
    text = redact_sensitive_values(content)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


class DMStore:
    """Thread-safe SQLite store for bot-visible DM conversations."""

    _SCHEMA_VERSION = 1

    def __init__(self, db_path: str, retention_days: int = 90) -> None:
        self._db_path = db_path
        self._retention_days = retention_days
        self._lock = threading.RLock()
        self._initialized = False

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
            CREATE TABLE IF NOT EXISTS dm_users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                global_name TEXT,
                display_name TEXT,
                bot INTEGER NOT NULL DEFAULT 0,
                last_seen_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_messages (
                message_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                clean_content TEXT NOT NULL DEFAULT '',
                content_preview TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                edited_at TEXT,
                deleted_at TEXT,
                jump_url TEXT,
                reply_to_message_id TEXT,
                has_attachments INTEGER NOT NULL DEFAULT 0,
                has_embeds INTEGER NOT NULL DEFAULT 0,
                direction TEXT NOT NULL DEFAULT 'inbound',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_channel_time ON dm_messages(channel_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_author_time ON dm_messages(author_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_active ON dm_messages(channel_id, deleted_at, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_created ON dm_messages(created_at)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    async def upsert_user(
        self,
        user_id: int,
        username: str | None = None,
        global_name: str | None = None,
        display_name: str | None = None,
        is_bot: bool = False,
    ) -> None:
        await self.initialize()
        await asyncio.to_thread(
            self._upsert_user_sync,
            user_id=str(user_id),
            username=username,
            global_name=global_name,
            display_name=display_name,
            is_bot=is_bot,
            last_seen=_now_iso(),
        )

    def _upsert_user_sync(
        self,
        user_id: str,
        username: str | None,
        global_name: str | None,
        display_name: str | None,
        is_bot: bool,
        last_seen: str,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO dm_users (user_id, username, global_name, display_name, bot, last_seen_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        username=COALESCE(excluded.username, dm_users.username),
                        global_name=COALESCE(excluded.global_name, dm_users.global_name),
                        display_name=COALESCE(excluded.display_name, dm_users.display_name),
                        bot=excluded.bot,
                        last_seen_at=excluded.last_seen_at
                    """,
                    (user_id, username, global_name, display_name, int(is_bot), last_seen),
                )
                conn.commit()
            finally:
                conn.close()

    async def add_message(
        self,
        message_id: int,
        channel_id: int,
        author_id: int,
        content: str = "",
        clean_content: str = "",
        created_at: str | None = None,
        is_bot_author: bool = False,
        reply_to_message_id: int | None = None,
        has_attachments: bool = False,
        has_embeds: bool = False,
        jump_url: str | None = None,
    ) -> None:
        await self.initialize()
        direction = "outbound" if is_bot_author else "inbound"
        content_preview = _make_preview(clean_content or content)

        await asyncio.to_thread(
            self._insert_message_sync,
            message_id=str(message_id),
            channel_id=str(channel_id),
            author_id=str(author_id),
            content=content,
            clean_content=clean_content,
            content_preview=content_preview,
            created_at=created_at or _now_iso(),
            direction=direction,
            reply_to_message_id=str(reply_to_message_id) if reply_to_message_id else None,
            has_attachments=int(has_attachments),
            has_embeds=int(has_embeds),
            jump_url=jump_url,
        )

    def _insert_message_sync(
        self,
        message_id: str,
        channel_id: str,
        author_id: str,
        content: str,
        clean_content: str,
        content_preview: str,
        created_at: str,
        direction: str,
        reply_to_message_id: str | None,
        has_attachments: int,
        has_embeds: int,
        jump_url: str | None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO dm_messages (
                        message_id, channel_id, author_id, content, clean_content,
                        content_preview, created_at, direction, reply_to_message_id,
                        has_attachments, has_embeds, jump_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        channel_id,
                        author_id,
                        content,
                        clean_content,
                        content_preview,
                        created_at,
                        direction,
                        reply_to_message_id,
                        has_attachments,
                        has_embeds,
                        jump_url,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def list_threads(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
    ) -> dict[str, Any]:
        """List DM threads (unique channel_ids) with latest message info."""
        await self.initialize()
        page = max(1, page)
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size

        rows, total = await asyncio.to_thread(self._list_threads_sync, page_size=page_size, offset=offset)

        threads = []
        for row in rows:
            threads.append(
                {
                    "channel_id": row["channel_id"],
                    "other_user_id": row["other_user_id"],
                    "username": row["username"],
                    "global_name": row["global_name"],
                    "display_name": row["display_name"],
                    "message_count": row["message_count"],
                    "last_message_at": row["last_message_at"],
                    "last_preview": row["last_preview"],
                    "last_direction": row["last_direction"],
                },
            )

        return {
            "threads": threads,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    def _list_threads_sync(self, page_size: int, offset: int) -> tuple[list, int]:
        with self._lock:
            conn = self._connect()
            try:
                total = conn.execute("SELECT COUNT(DISTINCT channel_id) FROM dm_messages WHERE deleted_at IS NULL").fetchone()[0]

                # channel_id in a DM IS the other user's Discord ID (recipient).
                # We just need the latest message info per channel.
                rows = conn.execute(
                    """
                    SELECT
                        dm.channel_id,
                        dm.channel_id AS other_user_id,
                        du.username,
                        du.global_name,
                        du.display_name,
                        COUNT(*) AS message_count,
                        MAX(dm.created_at) AS last_message_at,
                        (SELECT content_preview FROM dm_messages m3
                         WHERE m3.channel_id = dm.channel_id
                         ORDER BY m3.created_at DESC LIMIT 1) AS last_preview,
                        (SELECT direction FROM dm_messages m4
                         WHERE m4.channel_id = dm.channel_id
                         ORDER BY m4.created_at DESC LIMIT 1) AS last_direction
                    FROM dm_messages dm
                    LEFT JOIN dm_users du ON du.user_id = dm.channel_id
                    WHERE dm.deleted_at IS NULL
                    GROUP BY dm.channel_id
                    ORDER BY last_message_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (page_size, offset),
                ).fetchall()
                return rows, total
            finally:
                conn.close()

    async def get_thread_messages(
        self,
        channel_id: int,
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
            self._get_thread_messages_sync,
            channel_id=str(channel_id),
            page_size=page_size,
            offset=offset,
        )

        messages = []
        for row in rows:
            messages.append(
                {
                    "message_id": row["message_id"],
                    "author_id": row["author_id"],
                    "content_preview": row["content_preview"],
                    "created_at": row["created_at"],
                    "direction": row["direction"],
                    "has_attachments": bool(row["has_attachments"]),
                    "has_embeds": bool(row["has_embeds"]),
                    "jump_url": row["jump_url"],
                },
            )

        return {
            "messages": messages,
            "channel_id": str(channel_id),
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    def _get_thread_messages_sync(self, channel_id: str, page_size: int, offset: int) -> tuple[list, int]:
        with self._lock:
            conn = self._connect()
            try:
                total = conn.execute(
                    "SELECT COUNT(*) FROM dm_messages WHERE channel_id = ? AND deleted_at IS NULL",
                    (channel_id,),
                ).fetchone()[0]

                rows = conn.execute(
                    """
                    SELECT * FROM dm_messages
                    WHERE channel_id = ? AND deleted_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (channel_id, page_size, offset),
                ).fetchall()
                return rows, total
            finally:
                conn.close()

    async def cleanup_retention(self) -> int:
        """Soft-delete messages older than retention period."""
        await self.initialize()
        cutoff = (datetime.now(UTC) - timedelta(days=self._retention_days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return await asyncio.to_thread(self._cleanup_sync, cutoff)

    def _cleanup_sync(self, cutoff: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE dm_messages SET deleted_at = ? WHERE created_at < ? AND deleted_at IS NULL",
                    (_now_iso(), cutoff),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()
