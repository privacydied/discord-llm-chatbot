"""Service singleton for the raw server archive."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bot.config import load_config

from .ingestion_queue import ArchiveIngestionQueue
from .store import ServerArchiveStore
from .sync import (
    build_bundle_from_message,
)
from .sync import (
    sync_channel_archive as _sync_channel_archive,
)
from .sync import (
    sync_guild_archive as _sync_guild_archive,
)
from .sync import (
    sync_thread_archive as _sync_thread_archive,
)

if TYPE_CHECKING:
    from .models import ArchiveMessageBundle, ArchiveSearchResult

logger = logging.getLogger(__name__)

_service: ServerArchiveService | None = None
_service_lock = asyncio.Lock()


class ServerArchiveService:
    def __init__(self, bot: Any | None = None) -> None:
        self.bot = bot
        self.refresh_config()
        self.store = ServerArchiveStore(self.db_path)
        self.queue = ArchiveIngestionQueue(
            self._persist_batch,
            max_size=self.queue_max,
            workers=1,
            batch_size=self.batch_size,
            enabled=self.enabled,
        )
        self._started = False
        self._paused = False
        self._start_lock = asyncio.Lock()
        self._guild_sync_tasks: dict[str, asyncio.Task[int]] = {}
        self._channel_sync_tasks: dict[str, asyncio.Task[int]] = {}
        self._sync_start_task: asyncio.Task | None = None

    def refresh_config(self) -> None:
        cfg = load_config()
        self.config = cfg
        self.enabled = bool(cfg.get("SERVER_ARCHIVE_ENABLED", cfg.get("SERVER_ARCHIVE_ENABLE", False)))
        self.db_path = Path(cfg.get("SERVER_ARCHIVE_DB_PATH", "./data/server_archive.db"))
        self.queue_max = max(1, int(cfg.get("SERVER_ARCHIVE_QUEUE_MAX", 1000)))
        self.batch_size = max(1, int(cfg.get("SERVER_ARCHIVE_BATCH_SIZE", 100)))
        self.search_limit = max(1, min(10, int(cfg.get("SERVER_ARCHIVE_SEARCH_LIMIT", 10))))
        self.admin_only = bool(cfg.get("SERVER_ARCHIVE_ADMIN_ONLY", True))
        self.sync_on_start = bool(cfg.get("SERVER_ARCHIVE_SYNC_ON_START", True))
        self.live_tail = bool(cfg.get("SERVER_ARCHIVE_LIVE_TAIL", True))
        self.max_message_chars = max(1, int(cfg.get("SERVER_ARCHIVE_MAX_MESSAGE_CHARS", 8000)))
        self.include_bot_messages = bool(
            cfg.get(
                "SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES",
                cfg.get("SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES", False),
            ),
        )

    async def start(self) -> None:
        if not self.enabled:
            return
        async with self._start_lock:
            if self._started:
                return
            await self.store.initialize()
            await self.queue.start()
            self._started = True
            if self.sync_on_start and self.bot is not None:
                self._sync_start_task = asyncio.create_task(self._sync_all_visible_guilds(), name="server-archive-sync-on-start")
            logger.info(
                "Server archive service started",
                extra={
                    "subsys": "server_archive",
                    "event": "archive_service_started",
                    "detail": {"db_path": str(self.db_path)},
                },
            )

    async def stop(self) -> None:
        self._paused = True
        if self._sync_start_task and not self._sync_start_task.done():
            self._sync_start_task.cancel()
        for task in list(self._guild_sync_tasks.values()) + list(self._channel_sync_tasks.values()):
            if not task.done():
                task.cancel()
        await self.queue.stop(timeout=5.0)
        tasks = [
            t
            for t in [
                self._sync_start_task,
                *self._guild_sync_tasks.values(),
                *self._channel_sync_tasks.values(),
            ]
            if t is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._guild_sync_tasks.clear()
        self._channel_sync_tasks.clear()
        self._sync_start_task = None
        self._started = False

    async def _persist_batch(self, bundles: list[ArchiveMessageBundle]) -> None:
        await self.store.upsert_bundles(bundles)

    def _should_skip_message(self, message: Any) -> bool:
        if not self.live_tail:
            return True
        guild = getattr(message, "guild", None)
        channel = getattr(message, "channel", None)
        author = getattr(message, "author", None)
        if guild is None or channel is None or author is None:
            return True
        if not self.include_bot_messages and bool(getattr(author, "bot", False)):
            return True
        if not getattr(message, "content", "") and not getattr(message, "attachments", None):
            return True
        return bool(self._looks_like_command(message))

    def _looks_like_command(self, message: Any) -> bool:
        content = str(getattr(message, "content", "") or "").lstrip()
        if not content:
            return False
        prefixes = getattr(self.bot, "command_prefix", "!") if self.bot is not None else "!"
        if isinstance(prefixes, (list, tuple, set)):
            return any(prefix and content.startswith(prefix) for prefix in prefixes)
        return bool(prefixes and content.startswith(str(prefixes)))

    async def enqueue_live_message(self, message: Any) -> bool:
        if not self.enabled or self._paused or self._should_skip_message(message):
            return False
        bundle = build_bundle_from_message(
            message,
            max_message_chars=self.max_message_chars,
            include_bot_messages=self.include_bot_messages,
        )
        if bundle is None:
            return False
        return await self.queue.enqueue(bundle)

    async def search(
        self,
        query: str,
        *,
        guild_id: str,
        channel_id: str | None = None,
        author_id: str | None = None,
        limit: int | None = None,
    ) -> list[ArchiveSearchResult]:
        if not self.enabled:
            return []
        return await self.store.search(
            query,
            guild_id=guild_id,
            channel_id=channel_id,
            author_id=author_id,
            limit=limit or self.search_limit,
        )

    async def get_message_by_id(
        self,
        message_id: str,
        *,
        guild_id: str | None = None,
        channel_id: str | None = None,
    ) -> dict | None:
        """Exact archived-message lookup with scope verification.

        Returns None when the archive is disabled, the ID is unknown, or the
        stored guild/channel does not match the caller's scope (when given).
        """
        if not self.enabled:
            return None
        try:
            record = await self.store.get_message_by_id(str(message_id))
        except Exception:
            return None
        if record is None:
            return None
        if guild_id is not None and str(record.get("guild_id") or "") != str(guild_id):
            return None
        if channel_id is not None and str(record.get("channel_id") or "") != str(channel_id):
            want_thread = str(channel_id)
            if str(record.get("thread_id") or "") != want_thread:
                return None
        return record

    async def sync_channel(self, channel: Any, *, force: bool = False) -> int:
        if not self.enabled or self._paused:
            return 0
        guild_id = str(getattr(getattr(channel, "guild", None), "id", ""))
        key = f"{guild_id}:{getattr(channel, 'id', '')}"
        task = self._channel_sync_tasks.get(key)
        if task and not task.done() and not force:
            return 0

        async def _runner() -> int:
            return await _sync_channel_archive(self.store, channel, force=force)

        task = asyncio.create_task(_runner(), name=f"server-archive-sync-channel-{key}")
        self._channel_sync_tasks[key] = task
        task.add_done_callback(lambda _t: self._channel_sync_tasks.pop(key, None))
        return await task

    async def sync_thread(self, thread: Any, *, force: bool = False) -> int:
        if not self.enabled or self._paused:
            return 0
        guild_id = str(getattr(getattr(thread, "guild", None), "id", ""))
        key = f"{guild_id}:thread:{getattr(thread, 'id', '')}"

        task = self._channel_sync_tasks.get(key)
        if task and not task.done() and not force:
            return 0

        async def _runner() -> int:
            return await _sync_thread_archive(self.store, thread, force=force)

        task = asyncio.create_task(_runner(), name=f"server-archive-sync-thread-{getattr(thread, 'id', '')}")
        self._channel_sync_tasks[key] = task
        task.add_done_callback(lambda _t: self._channel_sync_tasks.pop(key, None))
        return await task

    async def sync_guild(self, guild: Any, *, force: bool = False) -> int:
        if not self.enabled or self._paused:
            return 0
        guild_id = str(getattr(guild, "id", ""))
        task = self._guild_sync_tasks.get(guild_id)
        if task and not task.done() and not force:
            return 0

        async def _runner() -> int:
            return await _sync_guild_archive(self.store, guild, force=force)

        task = asyncio.create_task(_runner(), name=f"server-archive-sync-guild-{guild_id}")
        self._guild_sync_tasks[guild_id] = task
        task.add_done_callback(lambda _t: self._guild_sync_tasks.pop(guild_id, None))
        return await task

    async def _sync_all_visible_guilds(self) -> None:
        if self.bot is None:
            return
        try:
            for guild in list(getattr(self.bot, "guilds", []) or []):
                if self._paused:
                    break
                await self.sync_guild(guild)
        except Exception:
            logger.exception(
                "Server archive sync-on-start failed",
                extra={
                    "subsys": "server_archive",
                    "event": "archive_sync_on_start_failed",
                },
            )

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    async def get_status(self, *, guild_id: str | None = None) -> dict[str, Any]:
        counts = await self.store.counts(guild_id=guild_id)
        states = [state.to_row() for state in await self.store.list_sync_states(guild_id=guild_id)]
        guild_sync_running = False
        if guild_id is not None:
            guild_task = self._guild_sync_tasks.get(guild_id)
            guild_sync_running = bool(guild_task and not guild_task.done())
        else:
            guild_sync_running = any(task and not task.done() for task in self._guild_sync_tasks.values())
        if guild_id is not None:
            channel_running = any(key.startswith(f"{guild_id}:") and task and not task.done() for key, task in self._channel_sync_tasks.items())
        else:
            channel_running = any(task and not task.done() for task in self._channel_sync_tasks.values())
        return {
            "enabled": self.enabled,
            "started": self._started,
            "paused": self._paused,
            "db_path": str(self.db_path),
            "queue_size": self.queue.size,
            "queue_max": self.queue.max_size,
            "batch_size": self.batch_size,
            "search_limit": self.search_limit,
            "stats": asdict(self.queue.stats),
            "counts": counts,
            "sync_states": states,
            "sync_running": guild_sync_running or channel_running or (self._sync_start_task is not None and not self._sync_start_task.done()),
        }


async def get_server_archive_service(bot: Any | None = None) -> ServerArchiveService:
    global _service
    async with _service_lock:
        if _service is None:
            _service = ServerArchiveService(bot)
        elif bot is not None:
            _service.bot = bot
        return _service


async def start_server_archive_service(bot: Any | None = None) -> ServerArchiveService:
    service = await get_server_archive_service(bot)
    await service.start()
    return service


async def stop_server_archive_service() -> None:
    global _service
    if _service is None:
        return
    await _service.stop()


async def enqueue_live_message(message: Any) -> bool:
    service = await get_server_archive_service()
    return await service.enqueue_live_message(message)


async def search_archive(
    query: str,
    *,
    guild_id: str,
    channel_id: str | None = None,
    author_id: str | None = None,
    limit: int | None = None,
) -> list[ArchiveSearchResult]:
    service = await get_server_archive_service()
    return await service.search(
        query,
        guild_id=guild_id,
        channel_id=channel_id,
        author_id=author_id,
        limit=limit,
    )


async def get_archived_message(
    message_id: str,
    *,
    guild_id: str | None = None,
    channel_id: str | None = None,
) -> dict | None:
    """Exact archived-message lookup by Discord message ID (module helper)."""
    service = await get_server_archive_service()
    return await service.get_message_by_id(message_id, guild_id=guild_id, channel_id=channel_id)


async def get_server_archive_status(*, guild_id: str | None = None) -> dict[str, Any]:
    service = await get_server_archive_service()
    return await service.get_status(guild_id=guild_id)
