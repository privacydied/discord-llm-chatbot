"""Bounded backfill system for the Discord-like dashboard.

Provides:
- BackfillJobStore: SQLite-backed job tracking (queued → running → completed/failed/cancelled)
- BackfillService: async backfill of channels, guilds, and DM conversations

All operations are dispatched through asyncio.to_thread for the SQLite work
and use async Discord API calls for message history fetching. The system
respects Discord permission boundaries and never fetches data the bot cannot
see.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from bot.utils.logging import get_logger

if TYPE_CHECKING:
    import discord
    from discord.ext.commands import Bot as DiscordBot

    from .audit_store import AuditStore
    from .message_store import MessageStore

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKFILL_STATUS_QUEUED = "queued"
BACKFILL_STATUS_RUNNING = "running"
BACKFILL_STATUS_COMPLETED = "completed"
BACKFILL_STATUS_FAILED = "failed"
BACKFILL_STATUS_CANCELLED = "cancelled"

VALID_STATUS_TRANSITIONS: dict[str, set[str]] = {
    BACKFILL_STATUS_QUEUED: {BACKFILL_STATUS_RUNNING, BACKFILL_STATUS_CANCELLED},
    BACKFILL_STATUS_RUNNING: {BACKFILL_STATUS_COMPLETED, BACKFILL_STATUS_FAILED, BACKFILL_STATUS_CANCELLED},
    BACKFILL_STATUS_COMPLETED: set(),
    BACKFILL_STATUS_FAILED: set(),
    BACKFILL_STATUS_CANCELLED: set(),
}

BACKFILL_TARGET_CHANNEL = "channel"
BACKFILL_TARGET_GUILD = "guild"
BACKFILL_TARGET_DM = "dm"

DEFAULT_PER_CHANNEL_LIMIT = 100
DEFAULT_CHANNEL_LIMIT = 500
DEFAULT_DM_LIMIT = 500
DEFAULT_MAX_CHANNELS = 50
DEFAULT_SLEEP_BETWEEN_CHANNELS = 1.0  # seconds


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# BackfillJobStore
# ---------------------------------------------------------------------------


class BackfillJobStore:
    """SQLite-backed store for tracking backfill jobs.

    Thread-safe (threading.RLock + WAL). All public methods are async wrappers
    around sync implementations dispatched via asyncio.to_thread.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
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
            CREATE TABLE IF NOT EXISTS backfill_jobs (
                job_id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                messages_seen INTEGER NOT NULL DEFAULT 0,
                messages_inserted INTEGER NOT NULL DEFAULT 0,
                channels_seen INTEGER NOT NULL DEFAULT 0,
                channels_skipped INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_backfill_target
            ON backfill_jobs(target_type, target_id)
        """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    async def create_job(
        self,
        target_type: str,
        target_id: str,
    ) -> str:
        """Create a new backfill job in queued status. Returns job_id."""
        await self.initialize()
        return await asyncio.to_thread(
            self._create_job_sync,
            target_type=target_type,
            target_id=target_id,
        )

    def _create_job_sync(self, target_type: str, target_id: str) -> str:
        with self._lock:
            conn = self._connect()
            try:
                job_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT OR IGNORE INTO backfill_jobs
                    (job_id, target_type, target_id, status, created_at)
                    VALUES (?, ?, ?, ?, ?)""",
                    (job_id, target_type, target_id, BACKFILL_STATUS_QUEUED, _now_iso()),
                )
                conn.commit()
                # If INSERT IGNORE did nothing, find the existing job
                row = conn.execute(
                    "SELECT job_id FROM backfill_jobs WHERE target_type = ? AND target_id = ?",
                    (target_type, target_id),
                ).fetchone()
                return row["job_id"] if row else job_id
            finally:
                conn.close()

    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get a single job by ID."""
        await self.initialize()
        return await asyncio.to_thread(self._get_job_sync, job_id=job_id)

    def _get_job_sync(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM backfill_jobs WHERE job_id = ?", (job_id,)).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    async def get_active_job(self, target_type: str, target_id: str) -> Optional[dict[str, Any]]:
        """Get the most recent non-terminal job for a target, if any."""
        await self.initialize()
        return await asyncio.to_thread(self._get_active_job_sync, target_type=target_type, target_id=target_id)

    def _get_active_job_sync(self, target_type: str, target_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT * FROM backfill_jobs
                    WHERE target_type = ? AND target_id = ? AND status IN (?, ?)
                    ORDER BY created_at DESC LIMIT 1""",
                    (target_type, target_id, BACKFILL_STATUS_QUEUED, BACKFILL_STATUS_RUNNING),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    async def list_jobs(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
        status_filter: Optional[str] = None,
        target_type_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """List backfill jobs with pagination and optional filters."""
        await self.initialize()
        page = max(1, page)
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size

        rows, total = await asyncio.to_thread(
            self._list_jobs_sync,
            page_size=page_size,
            offset=offset,
            status_filter=status_filter,
            target_type_filter=target_type_filter,
        )

        jobs = [dict(r) for r in rows]
        total_pages = max(1, (total + page_size - 1) // page_size)
        return {
            "jobs": jobs,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        }

    def _list_jobs_sync(
        self,
        page_size: int,
        offset: int,
        status_filter: Optional[str] = None,
        target_type_filter: Optional[str] = None,
    ) -> tuple[list, int]:
        with self._lock:
            conn = self._connect()
            try:
                where_parts: list[str] = []
                params: list = []

                if status_filter:
                    where_parts.append("status = ?")
                    params.append(status_filter)
                if target_type_filter:
                    where_parts.append("target_type = ?")
                    params.append(target_type_filter)

                where_sql = " AND ".join(where_parts) if where_parts else "1=1"

                total = conn.execute(
                    f"SELECT COUNT(*) FROM backfill_jobs WHERE {where_sql}",  # nosec B608
                    params,
                ).fetchone()[0]

                rows = conn.execute(
                    f"SELECT * FROM backfill_jobs WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # nosec B608
                    params + [page_size, offset],
                ).fetchall()
                return rows, total
            finally:
                conn.close()

    async def update_status(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None,
        messages_seen: Optional[int] = None,
        messages_inserted: Optional[int] = None,
        channels_seen: Optional[int] = None,
        channels_skipped: Optional[int] = None,
    ) -> bool:
        """Update job status and optional counters. Returns False if transition is invalid."""
        await self.initialize()
        return await asyncio.to_thread(
            self._update_status_sync,
            job_id=job_id,
            status=status,
            error=error,
            messages_seen=messages_seen,
            messages_inserted=messages_inserted,
            channels_seen=channels_seen,
            channels_skipped=channels_skipped,
        )

    def _update_status_sync(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None,
        messages_seen: Optional[int] = None,
        messages_inserted: Optional[int] = None,
        channels_seen: Optional[int] = None,
        channels_skipped: Optional[int] = None,
    ) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                current = conn.execute("SELECT status FROM backfill_jobs WHERE job_id = ?", (job_id,)).fetchone()
                if current is None:
                    logger.warning("Backfill job %s not found for status update", job_id)
                    return False

                current_status = current["status"]
                allowed = VALID_STATUS_TRANSITIONS.get(current_status, set())
                if status not in allowed:
                    logger.warning(
                        "Invalid backfill status transition: %s -> %s for job %s",
                        current_status,
                        status,
                        job_id,
                    )
                    return False

                set_parts = ["status = ?"]
                set_params: list = [status]

                if status == BACKFILL_STATUS_RUNNING:
                    set_parts.append("started_at = ?")
                    set_params.append(_now_iso())
                elif status in (BACKFILL_STATUS_COMPLETED, BACKFILL_STATUS_FAILED, BACKFILL_STATUS_CANCELLED):
                    set_parts.append("finished_at = ?")
                    set_params.append(_now_iso())

                if error is not None:
                    set_parts.append("error = ?")
                    set_params.append(error[:1024])
                if messages_seen is not None:
                    set_parts.append("messages_seen = ?")
                    set_params.append(messages_seen)
                if messages_inserted is not None:
                    set_parts.append("messages_inserted = ?")
                    set_params.append(messages_inserted)
                if channels_seen is not None:
                    set_parts.append("channels_seen = ?")
                    set_params.append(channels_seen)
                if channels_skipped is not None:
                    set_parts.append("channels_skipped = ?")
                    set_params.append(channels_skipped)

                set_params.append(job_id)
                conn.execute(
                    f"UPDATE backfill_jobs SET {', '.join(set_parts)} WHERE job_id = ?",  # nosec B608
                    set_params,
                )
                conn.commit()
                return True
            finally:
                conn.close()

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        return await self.update_status(job_id, BACKFILL_STATUS_CANCELLED)

    async def reset_stale_jobs(self) -> int:
        """Reset jobs stuck in 'running' to 'queued' (e.g. after bot restart)."""
        await self.initialize()
        return await asyncio.to_thread(self._reset_stale_jobs_sync)

    def _reset_stale_jobs_sync(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE backfill_jobs SET status = ? WHERE status = ?",
                    (BACKFILL_STATUS_QUEUED, BACKFILL_STATUS_RUNNING),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# BackfillService
# ---------------------------------------------------------------------------


class BackfillService:
    """Async backfill service that fetches message history from Discord.

    Respects permissions, coalesces duplicate requests, and tracks progress
    via BackfillJobStore. All Discord API calls are async; SQLite writes are
    dispatched to a thread pool.
    """

    def __init__(
        self,
        bot: "DiscordBot",
        message_store: "MessageStore",
        job_store: BackfillJobStore,
        audit_store: Optional["AuditStore"] = None,
        sleep_between_channels: float = DEFAULT_SLEEP_BETWEEN_CHANNELS,
    ) -> None:
        self._bot = bot
        self._message_store = message_store
        self._job_store = job_store
        self._audit_store = audit_store
        self._sleep_between_channels = sleep_between_channels
        self._active_lock = asyncio.Lock()

    @property
    def bot(self) -> "DiscordBot":
        return self._bot

    @property
    def job_store(self) -> BackfillJobStore:
        return self._job_store

    # ------------------------------------------------------------------
    # Public backfill methods
    # ------------------------------------------------------------------

    async def backfill_channel(
        self,
        channel_id: int,
        limit: int = DEFAULT_CHANNEL_LIMIT,
        job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Backfill messages from a single text channel.

        Returns a result dict with messages_seen, messages_inserted, job_id, status.
        """
        job = await self._ensure_job(BACKFILL_TARGET_CHANNEL, str(channel_id), job_id)
        job_id = job["job_id"]

        # Resolve channel
        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except Exception as e:
                await self._fail_job(job_id, f"Channel not found: {e}")
                return {"job_id": job_id, "status": BACKFILL_STATUS_FAILED, "error": str(e)}

        # Permission check
        if not hasattr(channel, "history"):
            await self._fail_job(job_id, "Not a text-based channel")
            return {"job_id": job_id, "status": BACKFILL_STATUS_FAILED, "error": "Not a text channel"}

        try:
            perms = channel.permissions_for(channel.guild.me) if channel.guild else None
        except Exception:
            perms = None

        if perms is not None and not perms.read_message_history:
            await self._fail_job(job_id, "Bot lacks read_message_history permission")
            return {"job_id": job_id, "status": BACKFILL_STATUS_FAILED, "error": "Permission denied"}

        guild_id = channel.guild.id if channel.guild else None

        # Fetch messages
        messages_seen = 0
        messages_inserted = 0
        try:
            async for msg in channel.history(limit=limit):
                messages_seen += 1
                inserted = await self._store_message(msg, guild_id)
                if inserted:
                    messages_inserted += 1
        except Exception as e:
            await self._fail_job(job_id, str(e))
            return {
                "job_id": job_id,
                "status": BACKFILL_STATUS_FAILED,
                "error": str(e),
                "messages_seen": messages_seen,
                "messages_inserted": messages_inserted,
            }

        # Mark complete
        await self._job_store.update_status(
            job_id,
            BACKFILL_STATUS_COMPLETED,
            messages_seen=messages_seen,
            messages_inserted=messages_inserted,
            channels_seen=1,
        )

        await self._audit("backfill.channel", result="success", target_channel_id=channel_id)

        return {
            "job_id": job_id,
            "status": BACKFILL_STATUS_COMPLETED,
            "messages_seen": messages_seen,
            "messages_inserted": messages_inserted,
        }

    async def backfill_guild(
        self,
        guild_id: int,
        per_channel_limit: int = DEFAULT_PER_CHANNEL_LIMIT,
        max_channels: int = DEFAULT_MAX_CHANNELS,
        job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Backfill visible text channels in a guild, up to max_channels."""
        job = await self._ensure_job(BACKFILL_TARGET_GUILD, str(guild_id), job_id)
        job_id = job["job_id"]

        guild = self._bot.get_guild(guild_id)
        if guild is None:
            await self._fail_job(job_id, f"Guild {guild_id} not found")
            return {"job_id": job_id, "status": BACKFILL_STATUS_FAILED, "error": "Guild not found"}

        # Collect visible text channels
        text_channels = []
        for ch in guild.text_channels:
            try:
                perms = ch.permissions_for(guild.me)
                if perms.read_messages and perms.read_message_history:
                    text_channels.append(ch)
                else:
                    logger.debug("Skipping channel %s (no read permission)", ch.id)
            except Exception:
                pass
            if len(text_channels) >= max_channels:
                break

        channels_seen = 0
        channels_skipped = len(guild.text_channels) - len(text_channels) if guild.text_channels else 0
        total_messages_seen = 0
        total_messages_inserted = 0

        for idx, ch in enumerate(text_channels):
            channels_seen += 1
            logger.info(
                "Backfilling guild %s channel %d/%d: #%s (%s)",
                guild_id,
                idx + 1,
                len(text_channels),
                ch.name,
                ch.id,
            )

            try:
                async for msg in ch.history(limit=per_channel_limit):
                    total_messages_seen += 1
                    inserted = await self._store_message(msg, guild_id)
                    if inserted:
                        total_messages_inserted += 1
            except Exception as e:
                channels_skipped += 1
                logger.warning("Failed to backfill channel %s in guild %s: %s", ch.id, guild_id, e)
                # Continue with next channel

            # Rate-limiting sleep between channels
            if idx < len(text_channels) - 1 and self._sleep_between_channels > 0:
                await asyncio.sleep(self._sleep_between_channels)

        await self._job_store.update_status(
            job_id,
            BACKFILL_STATUS_COMPLETED,
            messages_seen=total_messages_seen,
            messages_inserted=total_messages_inserted,
            channels_seen=channels_seen,
            channels_skipped=channels_skipped,
        )

        await self._audit("backfill.guild", result="success", target_guild_id=guild_id)

        return {
            "job_id": job_id,
            "status": BACKFILL_STATUS_COMPLETED,
            "messages_seen": total_messages_seen,
            "messages_inserted": total_messages_inserted,
            "channels_seen": channels_seen,
            "channels_skipped": channels_skipped,
        }

    async def backfill_dm(
        self,
        user_id_or_channel_id: int,
        limit: int = DEFAULT_DM_LIMIT,
        job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Backfill DM conversation history with a user.

        Uses the user's DM channel. Falls back to creating a DM channel if
        one doesn't exist (which requires the user to share a guild or have
        DMs enabled).
        """
        job = await self._ensure_job(BACKFILL_TARGET_DM, str(user_id_or_channel_id), job_id)
        job_id = job["job_id"]

        # Try to get or create DM channel
        user = self._bot.get_user(user_id_or_channel_id)
        if user is None:
            try:
                user = await self._bot.fetch_user(user_id_or_channel_id)
            except Exception as e:
                await self._fail_job(job_id, f"User not found: {e}")
                return {"job_id": job_id, "status": BACKFILL_STATUS_FAILED, "error": str(e)}

        try:
            dm_channel = user.dm_channel or await user.create_dm()
        except Exception as e:
            await self._fail_job(job_id, f"Cannot create DM channel: {e}")
            return {"job_id": job_id, "status": BACKFILL_STATUS_FAILED, "error": str(e)}

        messages_seen = 0
        messages_inserted = 0

        try:
            async for msg in dm_channel.history(limit=limit):
                messages_seen += 1
                inserted = await self._store_message(msg, guild_id=None)
                if inserted:
                    messages_inserted += 1

                # Upsert DM thread for the other participant
                other = msg.author
                if other.id != self._bot.user.id:
                    await self._message_store.upsert_dm_thread(
                        dm_channel_id=dm_channel.id,
                        user_id=other.id,
                        username=other.name,
                        display_name=other.display_name,
                        avatar_url=str(other.display_avatar.url) if other.display_avatar else None,
                        last_message_id=msg.id,
                        last_message_at=msg.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if msg.created_at else None,
                        increment_count=True,
                    )
        except Exception as e:
            await self._fail_job(job_id, str(e))
            return {
                "job_id": job_id,
                "status": BACKFILL_STATUS_FAILED,
                "error": str(e),
                "messages_seen": messages_seen,
                "messages_inserted": messages_inserted,
            }

        await self._job_store.update_status(
            job_id,
            BACKFILL_STATUS_COMPLETED,
            messages_seen=messages_seen,
            messages_inserted=messages_inserted,
        )

        await self._audit(
            "backfill.dm",
            result="success",
            target_user_id=user_id_or_channel_id,
        )

        return {
            "job_id": job_id,
            "status": BACKFILL_STATUS_COMPLETED,
            "messages_seen": messages_seen,
            "messages_inserted": messages_inserted,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_job(
        self,
        target_type: str,
        target_id: str,
        existing_job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ensure a job exists. If an active job is found, return it.
        Otherwise create a new one. Uses asyncio.Lock to coalesce.
        """
        async with self._active_lock:
            if existing_job_id:
                job = await self._job_store.get_job(existing_job_id)
                if job:
                    return job

            # Check for an active (non-terminal) job for this target
            active = await self._job_store.get_active_job(target_type, target_id)
            if active:
                return active

            # Create a new job
            new_job_id = await self._job_store.create_job(target_type, target_id)
            # Transition to running
            await self._job_store.update_status(new_job_id, BACKFILL_STATUS_RUNNING)
            job = await self._job_store.get_job(new_job_id)
            return job or {"job_id": new_job_id}

    async def _fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed with an error message."""
        logger.warning("Backfill job %s failed: %s", job_id, error)
        await self._job_store.update_status(job_id, BACKFILL_STATUS_FAILED, error=error[:1024])
        await self._audit("backfill.failed", result="failed", metadata={"job_id": job_id, "error": error[:500]})

    async def _store_message(
        self,
        msg: "discord.Message",
        guild_id: Optional[int],
    ) -> bool:
        """Store a single Discord message into the message store.

        Returns True if the message was newly inserted, False if duplicate.
        All exceptions are caught and logged — never propagates.
        """
        try:
            # Determine author details
            author = msg.author
            bot_user = self._bot.user
            is_own_bot = bot_user is not None and author.id == bot_user.id
            direction = "outbound" if is_own_bot else "inbound"

            # Attachments
            attachments = []
            if msg.attachments:
                attachments = [
                    {
                        "id": str(a.id),
                        "filename": a.filename,
                        "url": a.url,
                        "size": a.size,
                        "content_type": a.content_type,
                    }
                    for a in msg.attachments
                ]

            # Embeds — store a safe summary
            embeds = []
            if msg.embeds:
                for e in msg.embeds[:5]:  # Limit to first 5 embeds
                    embeds.append(
                        {
                            "type": str(e.type),
                            "title": e.title,
                            "description": e.description[:200] if e.description else None,
                            "url": e.url,
                        }
                    )

            # Channel info
            channel_name = None
            channel_type = None
            if msg.channel:
                channel_name = getattr(msg.channel, "name", None)
                channel_type = str(getattr(msg.channel, "type", "text"))

            created_at = msg.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if msg.created_at else None
            edited_at = msg.edited_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if msg.edited_at else None

            return await self._message_store.insert_message(
                discord_message_id=msg.id,
                channel_id=msg.channel.id if msg.channel else 0,
                content=msg.content or "",
                guild_id=guild_id,
                channel_name=channel_name,
                channel_type=channel_type,
                author_id=author.id,
                author_username=author.name,
                author_display_name=author.display_name,
                author_avatar_url=str(author.display_avatar.url) if author.display_avatar else None,
                author_is_bot=author.bot,
                is_own_bot=is_own_bot,
                direction=direction,
                created_at=created_at,
                edited_at=edited_at,
                reply_to_message_id=msg.reference.message_id if msg.reference and msg.reference.message_id else None,
                attachments=attachments,
                embeds=embeds,
                metadata={"jump_url": msg.jump_url} if msg.jump_url else None,
            )
        except Exception as e:
            logger.warning("Failed to store message %s: %s", msg.id, e)
            return False

    async def _audit(
        self,
        event_type: str,
        result: str = "success",
        target_guild_id: Optional[int] = None,
        target_channel_id: Optional[int] = None,
        target_user_id: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record an audit event if audit_store is configured."""
        if self._audit_store is None:
            return
        try:
            await self._audit_store.record(
                event_type=event_type,
                result=result,
                target_guild_id=target_guild_id,
                target_channel_id=target_channel_id,
                target_user_id=target_user_id,
                metadata=metadata,
            )
        except Exception as e:
            logger.debug("Failed to record audit event %s: %s", event_type, e)

    async def initialize(self) -> None:
        """Initialize the job store and reset any stale running jobs."""
        await self._job_store.initialize()
        stale = await self._job_store.reset_stale_jobs()
        if stale:
            logger.info("Reset %d stale backfill job(s) to queued", stale)
