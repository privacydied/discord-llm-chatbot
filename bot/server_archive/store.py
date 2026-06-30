"""SQLite source of truth for the raw server archive."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import (
    ArchiveChannel,
    ArchiveGuild,
    ArchiveMessageBundle,
    ArchiveSearchResult,
    ArchiveSyncState,
    ArchiveThread,
    ArchiveUser,
    utc_now_iso,
)
from .search import normalize_query, sanitize_snippet

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
_SCHEMA_VERSION = 2


async def _to_thread(func, *args):
    return await asyncio.to_thread(func, *args)


class ServerArchiveStore:
    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialized = False
        self._fts_enabled = True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    async def initialize(self) -> None:
        if self._initialized:
            return
        await _to_thread(self._bootstrap_sync)
        self._initialized = True

    async def ensure_distiller_schema(self) -> None:
        await self.initialize()
        await _to_thread(self._ensure_distiller_schema_sync)

    def _ensure_distiller_schema_sync(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memory_distiller_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        guild_id TEXT NOT NULL,
                        channel_id TEXT,
                        thread_id TEXT,
                        author_id TEXT,
                        last_processed_message_id TEXT,
                        last_processed_created_at TEXT,
                        updated_at TEXT NOT NULL,
                        error TEXT
                    );

                    CREATE TABLE IF NOT EXISTS memory_distiller_runs (
                        run_id TEXT PRIMARY KEY,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        scanned_count INTEGER NOT NULL DEFAULT 0,
                        candidate_count INTEGER NOT NULL DEFAULT 0,
                        accepted_count INTEGER NOT NULL DEFAULT 0,
                        rejected_count INTEGER NOT NULL DEFAULT 0,
                        merged_count INTEGER NOT NULL DEFAULT 0,
                        error TEXT
                    );
                    """,
                )
                conn.commit()
            finally:
                conn.close()

    def _bootstrap_sync(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
                if version < _SCHEMA_VERSION:
                    conn.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS archive_guilds (
                            guild_id TEXT PRIMARY KEY,
                            name TEXT,
                            icon_url TEXT,
                            first_seen_at TEXT NOT NULL,
                            last_seen_at TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS archive_channels (
                            channel_id TEXT PRIMARY KEY,
                            guild_id TEXT NOT NULL,
                            parent_id TEXT,
                            name TEXT,
                            type TEXT,
                            archived_at TEXT NOT NULL,
                            last_synced_message_id TEXT,
                            last_synced_at TEXT,
                            FOREIGN KEY (guild_id) REFERENCES archive_guilds(guild_id) ON DELETE CASCADE
                        );

                        CREATE TABLE IF NOT EXISTS archive_threads (
                            thread_id TEXT PRIMARY KEY,
                            guild_id TEXT NOT NULL,
                            parent_channel_id TEXT NOT NULL,
                            name TEXT,
                            archived_at TEXT NOT NULL,
                            last_synced_message_id TEXT,
                            last_synced_at TEXT,
                            FOREIGN KEY (guild_id) REFERENCES archive_guilds(guild_id) ON DELETE CASCADE,
                            FOREIGN KEY (parent_channel_id) REFERENCES archive_channels(channel_id) ON DELETE CASCADE
                        );

                        CREATE TABLE IF NOT EXISTS archive_users (
                            user_id TEXT PRIMARY KEY,
                            username TEXT,
                            global_name TEXT,
                            display_name TEXT,
                            bot INTEGER NOT NULL DEFAULT 0,
                            last_seen_at TEXT NOT NULL,
                            avatar TEXT
                        );

                        CREATE TABLE IF NOT EXISTS archive_messages (
                            message_id TEXT PRIMARY KEY,
                            guild_id TEXT NOT NULL,
                            channel_id TEXT NOT NULL,
                            thread_id TEXT,
                            author_id TEXT NOT NULL,
                            content TEXT NOT NULL DEFAULT '',
                            clean_content TEXT NOT NULL DEFAULT '',
                            created_at TEXT NOT NULL,
                            edited_at TEXT,
                            deleted_at TEXT,
                            jump_url TEXT,
                            reply_to_message_id TEXT,
                            has_attachments INTEGER NOT NULL DEFAULT 0,
                            has_embeds INTEGER NOT NULL DEFAULT 0,
                            metadata_json TEXT NOT NULL DEFAULT '{}',
                            FOREIGN KEY (guild_id) REFERENCES archive_guilds(guild_id) ON DELETE CASCADE,
                            FOREIGN KEY (channel_id) REFERENCES archive_channels(channel_id) ON DELETE CASCADE,
                            FOREIGN KEY (thread_id) REFERENCES archive_threads(thread_id) ON DELETE CASCADE,
                            FOREIGN KEY (author_id) REFERENCES archive_users(user_id) ON DELETE CASCADE
                        );

                        CREATE TABLE IF NOT EXISTS archive_attachments (
                            attachment_id TEXT PRIMARY KEY,
                            message_id TEXT NOT NULL,
                            filename TEXT,
                            content_type TEXT,
                            size INTEGER,
                            url TEXT,
                            proxy_url TEXT,
                            FOREIGN KEY (message_id) REFERENCES archive_messages(message_id) ON DELETE CASCADE
                        );

                        CREATE TABLE IF NOT EXISTS archive_mentions (
                            message_id TEXT NOT NULL,
                            mentioned_user_id TEXT NOT NULL,
                            PRIMARY KEY(message_id, mentioned_user_id),
                            FOREIGN KEY (message_id) REFERENCES archive_messages(message_id) ON DELETE CASCADE
                        );

                        CREATE TABLE IF NOT EXISTS archive_sync_state (
                            scope_key TEXT PRIMARY KEY,
                            guild_id TEXT NOT NULL,
                            channel_id TEXT,
                            thread_id TEXT,
                            last_message_id TEXT,
                            last_synced_at TEXT,
                            status TEXT NOT NULL DEFAULT 'idle',
                            error TEXT
                        );

                        CREATE TABLE IF NOT EXISTS memory_distiller_state (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            guild_id TEXT NOT NULL,
                            channel_id TEXT,
                            thread_id TEXT,
                            author_id TEXT,
                            last_processed_message_id TEXT,
                            last_processed_created_at TEXT,
                            updated_at TEXT NOT NULL,
                            error TEXT
                        );

                        CREATE TABLE IF NOT EXISTS memory_distiller_runs (
                            run_id TEXT PRIMARY KEY,
                            started_at TEXT NOT NULL,
                            finished_at TEXT,
                            scanned_count INTEGER NOT NULL DEFAULT 0,
                            candidate_count INTEGER NOT NULL DEFAULT 0,
                            accepted_count INTEGER NOT NULL DEFAULT 0,
                            rejected_count INTEGER NOT NULL DEFAULT 0,
                            merged_count INTEGER NOT NULL DEFAULT 0,
                            error TEXT
                        );
                        """,
                    )
                    try:
                        conn.execute(
                            """
                            CREATE VIRTUAL TABLE IF NOT EXISTS archive_messages_fts USING fts5(
                                message_id UNINDEXED,
                                guild_id UNINDEXED,
                                channel_id UNINDEXED,
                                thread_id UNINDEXED,
                                author_id UNINDEXED,
                                content,
                                clean_content,
                                author_display,
                                created_at UNINDEXED,
                                tokenize='unicode61'
                            )
                            """,
                        )
                        self._fts_enabled = True
                    except sqlite3.OperationalError as exc:
                        if "fts5" not in str(exc).lower():
                            raise
                        self._fts_enabled = False
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS archive_messages_fts (
                                message_id TEXT PRIMARY KEY,
                                guild_id TEXT NOT NULL,
                                channel_id TEXT NOT NULL,
                                thread_id TEXT,
                                author_id TEXT NOT NULL,
                                content TEXT NOT NULL DEFAULT '',
                                clean_content TEXT NOT NULL DEFAULT '',
                                author_display TEXT NOT NULL DEFAULT '',
                                created_at TEXT NOT NULL
                            )
                            """,
                        )
                    # Idempotently add avatar column in case pre-existing table lacks it
                    try:
                        conn.execute("ALTER TABLE archive_users ADD COLUMN avatar TEXT")
                        conn.commit()
                    except sqlite3.OperationalError:
                        pass  # column already exists
                    conn.execute("PRAGMA user_version=2")
                    conn.commit()
                    logger.info(
                        "Server archive schema bootstrapped",
                        extra={
                            "subsys": "server_archive",
                            "event": "archive_schema_bootstrap",
                        },
                    )
                elif version == 1:
                    # v1→v2: add avatar column to archive_users
                    conn.execute("ALTER TABLE archive_users ADD COLUMN avatar TEXT")
                    conn.execute("PRAGMA user_version=2")
                    conn.commit()
                    logger.info(
                        "Server archive schema migrated v1→v2 (avatar column)",
                        extra={"subsys": "server_archive", "event": "archive_schema_migrate"},
                    )
                elif version > _SCHEMA_VERSION:
                    logger.warning(
                        "Server archive database schema is newer than this code supports",
                        extra={
                            "subsys": "server_archive",
                            "event": "archive_schema_version_ahead",
                            "detail": {"version": version, "code": _SCHEMA_VERSION},
                        },
                    )
            finally:
                conn.close()

    async def counts(self, *, guild_id: str | None = None) -> dict[str, int]:
        await self.initialize()
        return await _to_thread(self._counts_sync, guild_id)

    def _counts_sync(self, guild_id: str | None) -> dict[str, int]:
        with self._lock:
            conn = self._connect()
            try:
                where = " WHERE guild_id = ?" if guild_id else ""
                params: tuple[Any, ...] = (guild_id,) if guild_id else ()
                indexed_where = " WHERE guild_id = ?" if guild_id else ""
                indexed_params: tuple[Any, ...] = (guild_id,) if guild_id else ()
                indexed_messages = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM archive_messages_fts{indexed_where}",  # nosec B608
                        indexed_params,
                    ).fetchone()[0],
                )
                return {
                    "guilds": int(conn.execute(f"SELECT COUNT(*) FROM archive_guilds{where}", params).fetchone()[0]),  # nosec B608,
                    "channels": int(conn.execute(f"SELECT COUNT(*) FROM archive_channels{where}", params).fetchone()[0]),  # nosec B608,
                    "threads": int(conn.execute(f"SELECT COUNT(*) FROM archive_threads{where}", params).fetchone()[0]),  # nosec B608,
                    "users": int(conn.execute("SELECT COUNT(*) FROM archive_users").fetchone()[0]),
                    "messages": int(conn.execute(f"SELECT COUNT(*) FROM archive_messages{where}", params).fetchone()[0]),  # nosec B608,
                    "indexed_messages": indexed_messages,
                    "attachments": int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM archive_attachments a JOIN archive_messages m ON m.message_id = a.message_id{(' WHERE m.guild_id = ?' if guild_id else '')}",  # nosec B608,
                            params,
                        ).fetchone()[0],
                    ),
                    "mentions": int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM archive_mentions a JOIN archive_messages m ON m.message_id = a.message_id{(' WHERE m.guild_id = ?' if guild_id else '')}",  # nosec B608,
                            params,
                        ).fetchone()[0],
                    ),
                    "sync_states": int(conn.execute(f"SELECT COUNT(*) FROM archive_sync_state{where}", params).fetchone()[0]),  # nosec B608,
                }
            finally:
                conn.close()

    async def upsert_bundles(self, bundles: Sequence[ArchiveMessageBundle]) -> int:
        if not bundles:
            return 0
        await self.initialize()
        return await _to_thread(self._upsert_bundles_sync, list(bundles))

    async def upsert_bundle(self, bundle: ArchiveMessageBundle) -> int:
        return await self.upsert_bundles([bundle])

    def _upsert_bundles_sync(self, bundles: Sequence[ArchiveMessageBundle]) -> int:
        with self._lock:
            conn = self._connect()
            try:
                for bundle in bundles:
                    self._upsert_guild(conn, bundle.guild)
                    self._upsert_channel(conn, bundle.channel)
                    if bundle.thread is not None:
                        self._upsert_thread(conn, bundle.thread)
                    self._upsert_user(conn, bundle.author)
                    self._upsert_message(conn, bundle)
                conn.commit()
                return len(bundles)
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _upsert_guild(self, conn: sqlite3.Connection, guild: ArchiveGuild) -> None:
        conn.execute(
            """
            INSERT INTO archive_guilds(guild_id, name, icon_url, first_seen_at, last_seen_at)
            VALUES(:guild_id, :name, :icon_url, :first_seen_at, :last_seen_at)
            ON CONFLICT(guild_id) DO UPDATE SET
                name=excluded.name,
                icon_url=excluded.icon_url,
                last_seen_at=excluded.last_seen_at
            """,
            guild.to_row(),
        )

    def _upsert_channel(self, conn: sqlite3.Connection, channel: ArchiveChannel) -> None:
        conn.execute(
            """
            INSERT INTO archive_channels(channel_id, guild_id, parent_id, name, type, archived_at, last_synced_message_id, last_synced_at)
            VALUES(:channel_id, :guild_id, :parent_id, :name, :type, :archived_at, :last_synced_message_id, :last_synced_at)
            ON CONFLICT(channel_id) DO UPDATE SET
                guild_id=excluded.guild_id,
                parent_id=excluded.parent_id,
                name=excluded.name,
                type=excluded.type,
                archived_at=excluded.archived_at,
                last_synced_message_id=excluded.last_synced_message_id,
                last_synced_at=excluded.last_synced_at
            """,
            channel.to_row(),
        )

    def _upsert_thread(self, conn: sqlite3.Connection, thread: ArchiveThread) -> None:
        conn.execute(
            """
            INSERT INTO archive_threads(thread_id, guild_id, parent_channel_id, name, archived_at, last_synced_message_id, last_synced_at)
            VALUES(:thread_id, :guild_id, :parent_channel_id, :name, :archived_at, :last_synced_message_id, :last_synced_at)
            ON CONFLICT(thread_id) DO UPDATE SET
                guild_id=excluded.guild_id,
                parent_channel_id=excluded.parent_channel_id,
                name=excluded.name,
                archived_at=excluded.archived_at,
                last_synced_message_id=excluded.last_synced_message_id,
                last_synced_at=excluded.last_synced_at
            """,
            thread.to_row(),
        )

    def _upsert_user(self, conn: sqlite3.Connection, user: ArchiveUser) -> None:
        conn.execute(
            """
            INSERT INTO archive_users(user_id, username, global_name, display_name, bot, last_seen_at, avatar)
            VALUES(:user_id, :username, :global_name, :display_name, :bot, :last_seen_at, :avatar)
            ON CONFLICT(user_id) DO UPDATE SET
                username=excluded.username,
                global_name=excluded.global_name,
                display_name=excluded.display_name,
                bot=excluded.bot,
                last_seen_at=excluded.last_seen_at,
                avatar=COALESCE(excluded.avatar, avatar)
            """,
            user.to_row(),
        )

    def _upsert_message(self, conn: sqlite3.Connection, bundle: ArchiveMessageBundle) -> None:
        message = bundle.message
        conn.execute(
            """
            INSERT INTO archive_messages(
                message_id, guild_id, channel_id, thread_id, author_id,
                content, clean_content, created_at, edited_at, deleted_at,
                jump_url, reply_to_message_id, has_attachments, has_embeds, metadata_json
            ) VALUES (
                :message_id, :guild_id, :channel_id, :thread_id, :author_id,
                :content, :clean_content, :created_at, :edited_at, :deleted_at,
                :jump_url, :reply_to_message_id, :has_attachments, :has_embeds, :metadata_json
            )
            ON CONFLICT(message_id) DO UPDATE SET
                guild_id=excluded.guild_id,
                channel_id=excluded.channel_id,
                thread_id=excluded.thread_id,
                author_id=excluded.author_id,
                content=excluded.content,
                clean_content=excluded.clean_content,
                created_at=excluded.created_at,
                edited_at=excluded.edited_at,
                deleted_at=excluded.deleted_at,
                jump_url=excluded.jump_url,
                reply_to_message_id=excluded.reply_to_message_id,
                has_attachments=excluded.has_attachments,
                has_embeds=excluded.has_embeds,
                metadata_json=excluded.metadata_json
            """,
            message.to_row(),
        )
        conn.execute(
            "DELETE FROM archive_attachments WHERE message_id = ?",
            (message.message_id,),
        )
        conn.execute("DELETE FROM archive_mentions WHERE message_id = ?", (message.message_id,))
        if self._fts_enabled:
            conn.execute(
                "DELETE FROM archive_messages_fts WHERE message_id = ?",
                (message.message_id,),
            )

        for attachment in bundle.attachments:
            conn.execute(
                """
                INSERT INTO archive_attachments(attachment_id, message_id, filename, content_type, size, url, proxy_url)
                VALUES(:attachment_id, :message_id, :filename, :content_type, :size, :url, :proxy_url)
                ON CONFLICT(attachment_id) DO UPDATE SET
                    message_id=excluded.message_id,
                    filename=excluded.filename,
                    content_type=excluded.content_type,
                    size=excluded.size,
                    url=excluded.url,
                    proxy_url=excluded.proxy_url
                """,
                attachment.to_row(),
            )
        for mention in bundle.mentions:
            conn.execute(
                """
                INSERT OR REPLACE INTO archive_mentions(message_id, mentioned_user_id)
                VALUES(:message_id, :mentioned_user_id)
                """,
                mention.to_row(),
            )

        if self._fts_enabled:
            author_display = bundle.author.display_name or bundle.author.global_name or bundle.author.username or ""
            conn.execute(
                """
                INSERT INTO archive_messages_fts(
                    message_id, guild_id, channel_id, thread_id, author_id,
                    content, clean_content, author_display, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.guild_id,
                    message.channel_id,
                    message.thread_id,
                    message.author_id,
                    message.content or "",
                    message.clean_content or "",
                    author_display,
                    message.created_at,
                ),
            )
        else:
            conn.execute(
                """
                INSERT OR REPLACE INTO archive_messages_fts(
                    message_id, guild_id, channel_id, thread_id, author_id,
                    content, clean_content, author_display, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.guild_id,
                    message.channel_id,
                    message.thread_id,
                    message.author_id,
                    message.content or "",
                    message.clean_content or "",
                    bundle.author.display_name or bundle.author.global_name or bundle.author.username or "",
                    message.created_at,
                ),
            )

    async def soft_delete_message(self, message_id: str, deleted_at: str | None = None) -> bool:
        await self.initialize()
        return await _to_thread(self._soft_delete_message_sync, message_id, deleted_at)

    def _soft_delete_message_sync(self, message_id: str, deleted_at: str | None) -> bool:
        deleted_at = deleted_at or utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "UPDATE archive_messages SET deleted_at = ? WHERE message_id = ?",
                    (deleted_at, message_id),
                )
                if self._fts_enabled:
                    conn.execute(
                        "DELETE FROM archive_messages_fts WHERE message_id = ?",
                        (message_id,),
                    )
                conn.commit()
                return row.rowcount > 0
            finally:
                conn.close()

    async def delete_message(self, message_id: str) -> bool:
        await self.initialize()
        return await _to_thread(self._delete_message_sync, message_id)

    def _delete_message_sync(self, message_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("DELETE FROM archive_messages WHERE message_id = ?", (message_id,))
                conn.execute(
                    "DELETE FROM archive_attachments WHERE message_id = ?",
                    (message_id,),
                )
                conn.execute("DELETE FROM archive_mentions WHERE message_id = ?", (message_id,))
                if self._fts_enabled:
                    conn.execute(
                        "DELETE FROM archive_messages_fts WHERE message_id = ?",
                        (message_id,),
                    )
                conn.commit()
                return row.rowcount > 0
            finally:
                conn.close()

    def _sync_state_key(self, guild_id: str, channel_id: str | None = None, thread_id: str | None = None) -> str:
        return ":".join([guild_id, channel_id or "", thread_id or ""])

    async def get_sync_state(
        self,
        *,
        guild_id: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
    ) -> ArchiveSyncState | None:
        await self.initialize()
        return await _to_thread(self._get_sync_state_sync, guild_id, channel_id, thread_id)

    def _get_sync_state_sync(
        self,
        guild_id: str,
        channel_id: str | None,
        thread_id: str | None,
    ) -> ArchiveSyncState | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM archive_sync_state WHERE scope_key = ?",
                    (self._sync_state_key(guild_id, channel_id, thread_id),),
                ).fetchone()
                return ArchiveSyncState.from_row(row) if row else None
            finally:
                conn.close()

    async def set_sync_state(self, state: ArchiveSyncState) -> None:
        await self.initialize()
        await _to_thread(self._set_sync_state_sync, state)

    def _set_sync_state_sync(self, state: ArchiveSyncState) -> None:
        with self._lock:
            conn = self._connect()
            try:
                payload = state.to_row()
                conn.execute(
                    """
                    INSERT INTO archive_sync_state(
                        scope_key, guild_id, channel_id, thread_id, last_message_id,
                        last_synced_at, status, error
                    ) VALUES(:scope_key, :guild_id, :channel_id, :thread_id, :last_message_id, :last_synced_at, :status, :error)
                    ON CONFLICT(scope_key) DO UPDATE SET
                        last_message_id=excluded.last_message_id,
                        last_synced_at=excluded.last_synced_at,
                        status=excluded.status,
                        error=excluded.error
                    """,
                    payload,
                )
                conn.commit()
            finally:
                conn.close()

    async def list_sync_states(self, *, guild_id: str | None = None) -> list[ArchiveSyncState]:
        await self.initialize()
        return await _to_thread(self._list_sync_states_sync, guild_id)

    def _list_sync_states_sync(self, guild_id: str | None) -> list[ArchiveSyncState]:
        with self._lock:
            conn = self._connect()
            try:
                if guild_id is None:
                    rows = conn.execute("SELECT * FROM archive_sync_state ORDER BY last_synced_at DESC").fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM archive_sync_state WHERE guild_id = ? ORDER BY last_synced_at DESC",
                        (guild_id,),
                    ).fetchall()
                return [ArchiveSyncState.from_row(row) for row in rows]
            finally:
                conn.close()

    def _distiller_scope_key(
        self,
        guild_id: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
        author_id: str | None = None,
    ) -> tuple[Any, ...]:
        return (guild_id, channel_id, thread_id, author_id)

    async def list_distiller_scopes(self, *, limit: int = 200, guild_id: str | None = None) -> list[dict[str, Any]]:
        await self.ensure_distiller_schema()
        return await _to_thread(self._list_distiller_scopes_sync, limit, guild_id)

    def _list_distiller_scopes_sync(self, limit: int, guild_id: str | None) -> list[dict[str, Any]]:
        limit = max(1, min(1000, int(limit)))
        with self._lock:
            conn = self._connect()
            try:
                sql = "SELECT guild_id, channel_id, thread_id, author_id, MIN(created_at) AS first_seen_at FROM archive_messages WHERE deleted_at IS NULL"
                params: list[Any] = []
                if guild_id is not None:
                    sql += " AND guild_id = ?"
                    params.append(guild_id)
                sql += " GROUP BY guild_id, channel_id, thread_id, author_id ORDER BY first_seen_at ASC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    async def fetch_distiller_messages(
        self,
        *,
        guild_id: str,
        channel_id: str | None,
        thread_id: str | None,
        author_id: str,
        after_created_at: str | None = None,
        after_message_id: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        await self.ensure_distiller_schema()
        return await _to_thread(
            self._fetch_distiller_messages_sync,
            guild_id,
            channel_id,
            thread_id,
            author_id,
            after_created_at,
            after_message_id,
            limit,
        )

    def _fetch_distiller_messages_sync(
        self,
        guild_id: str,
        channel_id: str | None,
        thread_id: str | None,
        author_id: str,
        after_created_at: str | None,
        after_message_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(500, int(limit)))
        with self._lock:
            conn = self._connect()
            try:
                sql = (
                    "SELECT m.message_id, m.guild_id, m.channel_id, m.thread_id, m.author_id, u.bot AS author_bot, "
                    "m.content, m.clean_content, m.created_at, m.edited_at, m.deleted_at, m.jump_url, "
                    "m.reply_to_message_id, m.has_attachments, m.has_embeds, m.metadata_json "
                    "FROM archive_messages AS m "
                    "LEFT JOIN archive_users AS u ON u.user_id = m.author_id "
                    "WHERE m.deleted_at IS NULL AND m.guild_id = ? AND m.author_id = ?"
                )
                params: list[Any] = [guild_id, author_id]
                if channel_id is None:
                    sql += " AND channel_id IS NULL"
                else:
                    sql += " AND channel_id = ?"
                    params.append(channel_id)
                if thread_id is None:
                    sql += " AND thread_id IS NULL"
                else:
                    sql += " AND thread_id = ?"
                    params.append(thread_id)
                if after_created_at is not None and after_message_id is not None:
                    sql += " AND (created_at > ? OR (created_at = ? AND message_id > ?))"
                    params.extend([after_created_at, after_created_at, after_message_id])
                sql += " ORDER BY created_at ASC, message_id ASC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    async def get_distiller_state(
        self,
        *,
        guild_id: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
        author_id: str | None = None,
    ) -> dict[str, Any] | None:
        await self.ensure_distiller_schema()
        return await _to_thread(
            self._get_distiller_state_sync,
            guild_id,
            channel_id,
            thread_id,
            author_id,
        )

    def _get_distiller_state_sync(
        self,
        guild_id: str,
        channel_id: str | None,
        thread_id: str | None,
        author_id: str | None,
    ) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT * FROM memory_distiller_state
                    WHERE guild_id = ?
                      AND channel_id IS ?
                      AND thread_id IS ?
                      AND author_id IS ?
                    """,
                    (guild_id, channel_id, thread_id, author_id),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    async def upsert_distiller_state(
        self,
        *,
        guild_id: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
        author_id: str | None = None,
        last_processed_message_id: str | None = None,
        last_processed_created_at: str | None = None,
        error: str | None = None,
    ) -> None:
        await self.ensure_distiller_schema()
        await _to_thread(
            self._upsert_distiller_state_sync,
            guild_id,
            channel_id,
            thread_id,
            author_id,
            last_processed_message_id,
            last_processed_created_at,
            error,
        )

    def _upsert_distiller_state_sync(
        self,
        guild_id: str,
        channel_id: str | None,
        thread_id: str | None,
        author_id: str | None,
        last_processed_message_id: str | None,
        last_processed_created_at: str | None,
        error: str | None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                now = utc_now_iso()
                existing = conn.execute(
                    """
                    SELECT id FROM memory_distiller_state
                    WHERE guild_id = ?
                      AND channel_id IS ?
                      AND thread_id IS ?
                      AND author_id IS ?
                    """,
                    (guild_id, channel_id, thread_id, author_id),
                ).fetchone()
                if existing:
                    conn.execute(
                        """
                        UPDATE memory_distiller_state
                        SET last_processed_message_id = ?,
                            last_processed_created_at = ?,
                            updated_at = ?,
                            error = ?
                        WHERE id = ?
                        """,
                        (
                            last_processed_message_id,
                            last_processed_created_at,
                            now,
                            error,
                            existing["id"],
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO memory_distiller_state(
                            guild_id, channel_id, thread_id, author_id,
                            last_processed_message_id, last_processed_created_at, updated_at, error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            guild_id,
                            channel_id,
                            thread_id,
                            author_id,
                            last_processed_message_id,
                            last_processed_created_at,
                            now,
                            error,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()

    async def start_distiller_run(self, run_id: str, *, started_at: str) -> None:
        await self.ensure_distiller_schema()
        await _to_thread(self._start_distiller_run_sync, run_id, started_at)

    def _start_distiller_run_sync(self, run_id: str, started_at: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO memory_distiller_runs(
                        run_id, started_at, finished_at, scanned_count, candidate_count,
                        accepted_count, rejected_count, merged_count, error
                    ) VALUES (?, ?, NULL, 0, 0, 0, 0, 0, NULL)
                    """,
                    (run_id, started_at),
                )
                conn.commit()
            finally:
                conn.close()

    async def finish_distiller_run(
        self,
        run_id: str,
        *,
        finished_at: str,
        scanned_count: int,
        candidate_count: int,
        accepted_count: int,
        rejected_count: int,
        merged_count: int,
        error: str | None = None,
    ) -> None:
        await self.ensure_distiller_schema()
        await _to_thread(
            self._finish_distiller_run_sync,
            run_id,
            finished_at,
            scanned_count,
            candidate_count,
            accepted_count,
            rejected_count,
            merged_count,
            error,
        )

    def _finish_distiller_run_sync(
        self,
        run_id: str,
        finished_at: str,
        scanned_count: int,
        candidate_count: int,
        accepted_count: int,
        rejected_count: int,
        merged_count: int,
        error: str | None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE memory_distiller_runs
                    SET finished_at = ?, scanned_count = ?, candidate_count = ?, accepted_count = ?,
                        rejected_count = ?, merged_count = ?, error = ?
                    WHERE run_id = ?
                    """,
                    (
                        finished_at,
                        scanned_count,
                        candidate_count,
                        accepted_count,
                        rejected_count,
                        merged_count,
                        error,
                        run_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def latest_distiller_run(self) -> dict[str, Any] | None:
        await self.ensure_distiller_schema()
        return await _to_thread(self._latest_distiller_run_sync)

    def _latest_distiller_run_sync(self) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM memory_distiller_runs ORDER BY started_at DESC LIMIT 1").fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    async def count_distiller_backlog(self, *, guild_id: str | None = None) -> int:
        await self.ensure_distiller_schema()
        return await _to_thread(self._count_distiller_backlog_sync, guild_id)

    def _count_distiller_backlog_sync(self, guild_id: str | None) -> int:
        with self._lock:
            conn = self._connect()
            try:
                if guild_id is None:
                    rows = conn.execute("SELECT guild_id, channel_id, thread_id, author_id, last_processed_created_at, last_processed_message_id FROM memory_distiller_state").fetchall()
                else:
                    rows = conn.execute(
                        "SELECT guild_id, channel_id, thread_id, author_id, last_processed_created_at, last_processed_message_id FROM memory_distiller_state WHERE guild_id = ?",
                        (guild_id,),
                    ).fetchall()
                total = 0
                for row in rows:
                    params: list[Any] = [row["guild_id"], row["author_id"]]
                    sql = "SELECT COUNT(*) FROM archive_messages WHERE deleted_at IS NULL AND guild_id = ? AND author_id = ?"
                    if row["channel_id"] is None:
                        sql += " AND channel_id IS NULL"
                    else:
                        sql += " AND channel_id = ?"
                        params.append(row["channel_id"])
                    if row["thread_id"] is None:
                        sql += " AND thread_id IS NULL"
                    else:
                        sql += " AND thread_id = ?"
                        params.append(row["thread_id"])
                    if row["last_processed_created_at"] and row["last_processed_message_id"]:
                        sql += " AND (created_at > ? OR (created_at = ? AND message_id > ?))"
                        params.extend(
                            [
                                row["last_processed_created_at"],
                                row["last_processed_created_at"],
                                row["last_processed_message_id"],
                            ],
                        )
                    total += int(conn.execute(sql, params).fetchone()[0])

                sql = """
                    SELECT COUNT(*)
                    FROM archive_messages AS m
                    LEFT JOIN memory_distiller_state AS s
                      ON s.guild_id = m.guild_id
                     AND s.channel_id IS m.channel_id
                     AND s.thread_id IS m.thread_id
                     AND s.author_id IS m.author_id
                    WHERE m.deleted_at IS NULL
                      AND s.id IS NULL
                """
                params: list[Any] = []
                if guild_id is not None:
                    sql += " AND m.guild_id = ?"
                    params.append(guild_id)
                total += int(conn.execute(sql, params).fetchone()[0])
                return total
            finally:
                conn.close()

    async def search(
        self,
        query: str,
        *,
        guild_id: str,
        channel_id: str | None = None,
        author_id: str | None = None,
        limit: int = 5,
    ) -> list[ArchiveSearchResult]:
        await self.initialize()
        return await _to_thread(self._search_sync, query, guild_id, channel_id, author_id, limit)

    def _search_sync(
        self,
        query: str,
        guild_id: str,
        channel_id: str | None,
        author_id: str | None,
        limit: int,
    ) -> list[ArchiveSearchResult]:
        normalized = normalize_query(query)
        if not normalized:
            return []
        limit = max(1, min(10, int(limit)))
        with self._lock:
            conn = self._connect()
            try:
                if self._fts_enabled:
                    sql = """
                        SELECT
                            m.message_id,
                            m.guild_id,
                            m.channel_id,
                            m.thread_id,
                            m.author_id,
                            u.display_name AS author_name,
                            c.name AS channel_name,
                            m.content,
                            m.clean_content,
                            snippet(archive_messages_fts, 5, '[', ']', '…', 10) AS snippet,
                            m.created_at,
                            m.edited_at,
                            m.jump_url,
                            bm25(archive_messages_fts) AS score
                        FROM archive_messages_fts
                        JOIN archive_messages AS m ON m.message_id = archive_messages_fts.message_id
                        LEFT JOIN archive_users AS u ON u.user_id = m.author_id
                        LEFT JOIN archive_channels AS c ON c.channel_id = m.channel_id
                        WHERE archive_messages_fts MATCH ?
                          AND m.guild_id = ?
                          AND m.deleted_at IS NULL
                    """
                    params: list[Any] = [normalized, guild_id]
                    if channel_id is not None:
                        sql += " AND m.channel_id = ?"
                        params.append(channel_id)
                    if author_id is not None:
                        sql += " AND m.author_id = ?"
                        params.append(author_id)
                    sql += " ORDER BY score ASC, m.created_at DESC LIMIT ?"
                    params.append(limit)
                    rows = conn.execute(sql, params).fetchall()
                else:
                    sql = """
                        SELECT
                            m.message_id,
                            m.guild_id,
                            m.channel_id,
                            m.thread_id,
                            m.author_id,
                            u.display_name AS author_name,
                            c.name AS channel_name,
                            m.content,
                            m.clean_content,
                            substr(m.clean_content, 1, 240) AS snippet,
                            m.created_at,
                            m.edited_at,
                            m.jump_url,
                            0.0 AS score
                        FROM archive_messages AS m
                        LEFT JOIN archive_users AS u ON u.user_id = m.author_id
                        LEFT JOIN archive_channels AS c ON c.channel_id = m.channel_id
                        WHERE m.guild_id = ?
                          AND m.deleted_at IS NULL
                          AND (m.content LIKE ? OR m.clean_content LIKE ?)
                    """
                    params = [guild_id, f"%{normalized}%", f"%{normalized}%"]
                    if channel_id is not None:
                        sql += " AND m.channel_id = ?"
                        params.append(channel_id)
                    if author_id is not None:
                        sql += " AND m.author_id = ?"
                        params.append(author_id)
                    sql += " ORDER BY m.created_at DESC LIMIT ?"
                    params.append(limit)
                    rows = conn.execute(sql, params).fetchall()
                results: list[ArchiveSearchResult] = []
                for row in rows:
                    data = dict(row)
                    data["snippet"] = sanitize_snippet(data.get("snippet") or data.get("clean_content") or data.get("content") or "")
                    results.append(ArchiveSearchResult.from_row(data))
                return results
            finally:
                conn.close()

    async def latest_message_id_for_scope(
        self,
        *,
        guild_id: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        await self.initialize()
        return await _to_thread(self._latest_message_id_sync, guild_id, channel_id, thread_id)

    def _latest_message_id_sync(self, guild_id: str, channel_id: str | None, thread_id: str | None) -> str | None:
        with self._lock:
            conn = self._connect()
            try:
                sql = "SELECT message_id FROM archive_messages WHERE guild_id = ? AND deleted_at IS NULL"
                params: list[Any] = [guild_id]
                if channel_id is not None:
                    sql += " AND channel_id = ?"
                    params.append(channel_id)
                if thread_id is not None:
                    sql += " AND thread_id = ?"
                    params.append(thread_id)
                sql += " ORDER BY created_at DESC LIMIT 1"
                row = conn.execute(sql, params).fetchone()
                return str(row[0]) if row else None
            finally:
                conn.close()

    async def get_channel_messages(
        self,
        channel_id: str,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size
        return await _to_thread(self._get_channel_messages_sync, channel_id, page, page_size, offset, after_id, before_id)

    def _get_channel_messages_sync(
        self,
        channel_id: str,
        page: int,
        page_size: int,
        offset: int,
        after_id: str | None,
        before_id: str | None,
    ) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                where_parts = ["am.channel_id = ? AND am.deleted_at IS NULL"]
                params: list[Any] = [channel_id]

                if after_id:
                    where_parts.append(
                        "am.created_at > (SELECT created_at FROM archive_messages WHERE message_id = ?)"
                    )
                    params.append(after_id)
                if before_id:
                    where_parts.append(
                        "am.created_at < (SELECT created_at FROM archive_messages WHERE message_id = ?)"
                    )
                    params.append(before_id)

                where_sql = " AND ".join(where_parts)
                count = conn.execute(
                    f"SELECT COUNT(*) FROM archive_messages am WHERE {where_sql}",  # nosec B608
                    params,
                ).fetchone()[0]

                rows = conn.execute(
                    f"""SELECT am.message_id, am.content, am.created_at, am.edited_at,
                               am.author_id, am.reply_to_message_id, am.deleted_at,
                               am.metadata_json, am.has_attachments,
                               au.username, au.global_name, au.display_name, au.bot,
                               au.avatar
                        FROM archive_messages am
                        LEFT JOIN archive_users au ON am.author_id = au.user_id
                        WHERE {where_sql}
                        ORDER BY am.created_at DESC LIMIT ? OFFSET ?""",  # nosec B608
                    [*params, page_size, offset],
                ).fetchall()

                # Batch-fetch attachments for messages that have them
                msg_ids = [dict(r)["message_id"] for r in rows if dict(r).get("has_attachments")]
                attachments_by_msg: dict[str, list[dict[str, Any]]] = {}
                if msg_ids:
                    placeholders = ",".join("?" * len(msg_ids))
                    att_rows = conn.execute(
                        f"SELECT message_id, attachment_id, filename, content_type, size, url, proxy_url "  # nosec B608
                        f"FROM archive_attachments WHERE message_id IN ({placeholders})",
                        msg_ids,
                    ).fetchall()
                    for att in att_rows:
                        a = dict(att)
                        attachments_by_msg.setdefault(a["message_id"], []).append({
                            "id": a.get("attachment_id"),
                            "filename": a.get("filename"),
                            "content_type": a.get("content_type"),
                            "size": a.get("size"),
                            "url": a.get("url") or "",
                            "proxy_url": a.get("proxy_url") or "",
                        })

                messages = []
                for row in rows:
                    r = dict(row)
                    username = r.get("username") or ""
                    display = r.get("display_name") or r.get("global_name") or username
                    mid = r["message_id"]
                    avatar = r.get("avatar")
                    if not avatar:
                        uid = r.get("author_id")
                        try:
                            avatar = f"https://cdn.discordapp.com/embed/avatars/{int(uid) % 6}.png"
                        except (TypeError, ValueError):
                            pass
                    messages.append({
                        "discord_message_id": mid,
                        "content": r.get("content") or "",
                        "created_at": r.get("created_at"),
                        "edited_at": r.get("edited_at"),
                        "deleted_at": r.get("deleted_at"),
                        "author_id": r.get("author_id"),
                        "author_username": username,
                        "author_display_name": display,
                        "author_avatar_url": avatar,
                        "author_is_bot": bool(r.get("bot")),
                        "is_own_bot": bool(r.get("bot")),
                        "reply_to_message_id": r.get("reply_to_message_id"),
                        "attachments_json": attachments_by_msg.get(mid, []),
                        "embeds_json": [],
                        "metadata_json": r.get("metadata_json") or {},
                    })

                total_pages = max(1, (count + page_size - 1) // page_size)
                return {
                    "messages": messages,
                    "channel_id": channel_id,
                    "page": page,
                    "page_size": page_size,
                    "total": count,
                    "total_pages": total_pages,
                }
            finally:
                conn.close()
