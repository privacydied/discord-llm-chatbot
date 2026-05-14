"""Append-only audit log backed by SQLite with WAL mode and retention cleanup."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from bot.utils.logging import get_logger, redact_sensitive_values

logger = get_logger(__name__)

# Event types
EVENT_DASHBOARD_LOGIN_SUCCESS = "dashboard.login.success"
EVENT_DASHBOARD_LOGIN_FAILURE = "dashboard.login.failure"
EVENT_DASHBOARD_LOGOUT = "dashboard.logout"
EVENT_DASHBOARD_VIEW_DMS = "dashboard.view.dms"
EVENT_DASHBOARD_SEND_DM = "dashboard.send.dm"
EVENT_DASHBOARD_SEND_GUILD_MESSAGE = "dashboard.send.guild_message"
EVENT_DASHBOARD_GUILD_JOIN = "dashboard.guild.join"
EVENT_DASHBOARD_GUILD_LEAVE = "dashboard.guild.leave"
EVENT_DASHBOARD_COMMAND_INVOKE = "dashboard.command.invoke"
EVENT_DASHBOARD_ALERT_SEND = "dashboard.alert.send"
EVENT_DASHBOARD_CONFIG_RELOAD = "dashboard.config.reload"
EVENT_DASHBOARD_START = "dashboard.start"
EVENT_DASHBOARD_STOP = "dashboard.stop"

# Max preview chars for message content in audit logs
_MAX_PREVIEW_CHARS = 200


def _make_preview(content: str, max_chars: int = _MAX_PREVIEW_CHARS) -> str:
    """Truncate and redact message content for audit log preview."""
    text = redact_sensitive_values(content)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _truncate_hash(value: str, length: int = 16) -> str:
    """SHA-256 hash truncated for accountability without full content."""
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _truncate_ip(ip: str) -> str:
    """Hash IP address for privacy while preserving accountability."""
    if not ip:
        return ""
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


class AuditStore:
    """Thread-safe SQLite audit log with WAL mode, bounded retention, and pagination."""

    _SCHEMA_VERSION = 1

    def __init__(self, db_path: str, retention_days: int = 180) -> None:
        self._db_path = db_path
        self._retention_days = retention_days
        self._lock = threading.RLock()
        self._initialized = False

    async def initialize(self) -> None:
        """Bootstrap the database schema (runs in thread pool)."""
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
            CREATE TABLE IF NOT EXISTS audit_events (
                audit_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                actor_user_id TEXT,
                actor_source_ip TEXT,
                actor_user_agent TEXT,
                target_user_id TEXT,
                target_guild_id TEXT,
                target_channel_id TEXT,
                message_id TEXT,
                result TEXT NOT NULL,
                error_code TEXT,
                content_preview TEXT,
                content_hash TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_events(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_events(actor_user_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_target_guild ON audit_events(target_guild_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_result ON audit_events(result, created_at)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    async def record(
        self,
        event_type: str,
        result: str = "success",
        actor_user_id: Optional[int] = None,
        actor_source_ip: Optional[str] = None,
        actor_user_agent: Optional[str] = None,
        target_user_id: Optional[int] = None,
        target_guild_id: Optional[int] = None,
        target_channel_id: Optional[int] = None,
        message_id: Optional[int] = None,
        error_code: Optional[str] = None,
        content_preview: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Append an audit event. Thread-safe via asyncio.to_thread."""
        await self.initialize()
        audit_id = str(uuid.uuid4())
        await asyncio.to_thread(
            self._insert_sync,
            audit_id=audit_id,
            event_type=event_type,
            result=result,
            actor_user_id=str(actor_user_id) if actor_user_id else None,
            actor_source_ip=_truncate_ip(actor_source_ip) if actor_source_ip else None,
            actor_user_agent=(actor_user_agent or "")[:256],
            target_user_id=str(target_user_id) if target_user_id else None,
            target_guild_id=str(target_guild_id) if target_guild_id else None,
            target_channel_id=str(target_channel_id) if target_channel_id else None,
            message_id=str(message_id) if message_id else None,
            error_code=error_code,
            content_preview=content_preview,
            content_hash=_truncate_hash(content_preview) if content_preview else None,
            metadata=metadata or {},
        )
        return audit_id

    def _insert_sync(
        self,
        audit_id: str,
        event_type: str,
        result: str,
        actor_user_id: Optional[str],
        actor_source_ip: Optional[str],
        actor_user_agent: str,
        target_user_id: Optional[str],
        target_guild_id: Optional[str],
        target_channel_id: Optional[str],
        message_id: Optional[str],
        error_code: Optional[str],
        content_preview: Optional[str],
        content_hash: Optional[str],
        metadata: dict[str, Any],
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        audit_id, event_type, actor_user_id, actor_source_ip,
                        actor_user_agent, target_user_id, target_guild_id,
                        target_channel_id, message_id, result, error_code,
                        content_preview, content_hash, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        audit_id,
                        event_type,
                        actor_user_id,
                        actor_source_ip,
                        actor_user_agent,
                        target_user_id,
                        target_guild_id,
                        target_channel_id,
                        message_id,
                        result,
                        error_code,
                        content_preview,
                        content_hash,
                        json.dumps(metadata),
                        _now_iso(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def query(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
        event_type: Optional[str] = None,
        actor_user_id: Optional[int] = None,
        target_guild_id: Optional[int] = None,
        target_user_id: Optional[int] = None,
        result: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> dict[str, Any]:
        """Query audit events with pagination and filters."""
        await self.initialize()
        page = max(1, page)
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size

        rows, total = await asyncio.to_thread(
            self._query_sync,
            page_size=page_size,
            offset=offset,
            event_type=event_type,
            actor_user_id=actor_user_id,
            target_guild_id=target_guild_id,
            target_user_id=target_user_id,
            result=result,
            date_from=date_from,
            date_to=date_to,
        )
        total_pages = max(1, (total + page_size - 1) // page_size)

        events = []
        for row in rows:
            events.append(
                {
                    "audit_id": row["audit_id"],
                    "event_type": row["event_type"],
                    "actor_user_id": row["actor_user_id"],
                    "actor_source_ip": row["actor_source_ip"],
                    "target_user_id": row["target_user_id"],
                    "target_guild_id": row["target_guild_id"],
                    "target_channel_id": row["target_channel_id"],
                    "message_id": row["message_id"],
                    "result": row["result"],
                    "error_code": row["error_code"],
                    "content_preview": row["content_preview"],
                    "content_hash": row["content_hash"],
                    "metadata": json.loads(row["metadata_json"]),
                    "created_at": row["created_at"],
                }
            )

        return {
            "events": events,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        }

    def _query_sync(
        self,
        page_size: int,
        offset: int,
        event_type: Optional[str] = None,
        actor_user_id: Optional[int] = None,
        target_guild_id: Optional[int] = None,
        target_user_id: Optional[int] = None,
        result: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> tuple[list, int]:
        with self._lock:
            conn = self._connect()
            try:
                where = []
                params: list = []

                if event_type:
                    where.append("event_type = ?")
                    params.append(event_type)
                if actor_user_id:
                    where.append("actor_user_id = ?")
                    params.append(str(actor_user_id))
                if target_guild_id:
                    where.append("target_guild_id = ?")
                    params.append(str(target_guild_id))
                if target_user_id:
                    where.append("target_user_id = ?")
                    params.append(str(target_user_id))
                if result:
                    where.append("result = ?")
                    params.append(result)
                if date_from:
                    where.append("created_at >= ?")
                    params.append(date_from)
                if date_to:
                    where.append("created_at <= ?")
                    params.append(date_to)

                where_sql = " AND ".join(where) if where else "1=1"

                count = conn.execute(f"SELECT COUNT(*) FROM audit_events WHERE {where_sql}", params).fetchone()[0]  # nosec B608

                rows = conn.execute(
                    # nosec B608 — where_sql built from whitelist of column names, values parameterized
                    f"""
                    SELECT * FROM audit_events
                    WHERE {where_sql}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    params + [page_size, offset],
                ).fetchall()
                return rows, count
            finally:
                conn.close()

    async def cleanup_retention(self) -> int:
        """Remove audit events older than retention period. Returns deleted count."""
        await self.initialize()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return await asyncio.to_thread(self._cleanup_sync, cutoff)

    def _cleanup_sync(self, cutoff: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM audit_events WHERE created_at < ?", (cutoff,))
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()
