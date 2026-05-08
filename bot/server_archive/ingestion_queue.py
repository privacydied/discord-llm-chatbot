"""Bounded async ingestion queue for the raw server archive."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Sequence

from .models import ArchiveMessageBundle

logger = logging.getLogger(__name__)

PersistCallback = Callable[[Sequence[ArchiveMessageBundle]], Awaitable[None]]


@dataclass(slots=True)
class ArchiveQueueStats:
    enqueued: int = 0
    processed: int = 0
    dropped: int = 0
    failed: int = 0
    batches: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ArchiveIngestionQueue:
    def __init__(
        self,
        persist_callback: PersistCallback,
        *,
        max_size: int = 1000,
        workers: int = 1,
        batch_size: int = 100,
        enabled: bool = True,
    ) -> None:
        self.persist_callback = persist_callback
        self.max_size = max(1, int(max_size))
        self.workers = max(1, min(2, int(workers)))
        self.batch_size = max(1, int(batch_size))
        self.enabled = bool(enabled)
        self.stats = ArchiveQueueStats()
        self._queue: asyncio.Queue[ArchiveMessageBundle] = asyncio.Queue(maxsize=self.max_size)
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()
        self._start_lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if not self.enabled:
            return
        async with self._start_lock:
            if self._worker_tasks:
                return
            self._shutdown.clear()
            for index in range(self.workers):
                task = asyncio.create_task(self._worker_loop(index), name=f"server-archive-worker-{index}")
                self._worker_tasks.append(task)
            logger.info(
                "Server archive queue started",
                extra={
                    "subsys": "server_archive",
                    "event": "archive_queue_started",
                    "detail": {"workers": self.workers, "max_size": self.max_size},
                },
            )

    async def stop(self, timeout: float = 5.0) -> None:
        if not self._worker_tasks:
            return
        self._shutdown.set()
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for server archive queue to drain",
                extra={"subsys": "server_archive", "event": "archive_queue_drain_timeout"},
            )
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    async def enqueue(self, bundle: ArchiveMessageBundle) -> bool:
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait(bundle)
            self.stats.enqueued += 1
            return True
        except asyncio.QueueFull:
            self.stats.dropped += 1
            logger.warning(
                "Server archive queue full; dropping bundle",
                extra={
                    "subsys": "server_archive",
                    "event": "archive_queue_full",
                    "detail": {
                        "message_id": bundle.message.message_id,
                        "guild_id": bundle.message.guild_id,
                        "channel_id": bundle.message.channel_id,
                    },
                },
            )
            return False

    async def _worker_loop(self, worker_id: int) -> None:
        try:
            while not self._shutdown.is_set():
                batch: list[ArchiveMessageBundle] = []
                try:
                    first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    batch.append(first)
                    for _ in range(self.batch_size - 1):
                        try:
                            batch.append(self._queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    started = time.perf_counter()
                    await self.persist_callback(batch)
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                    self.stats.processed += len(batch)
                    self.stats.batches += 1
                    logger.debug(
                        "Server archive batch persisted",
                        extra={
                            "subsys": "server_archive",
                            "event": "archive_batch_persisted",
                            "detail": {
                                "worker_id": worker_id,
                                "batch_size": len(batch),
                                "elapsed_ms": elapsed_ms,
                            },
                        },
                    )
                    for _ in batch:
                        self._queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception:
                    self.stats.failed += len(batch) or 1
                    logger.exception(
                        "Server archive worker failed",
                        extra={
                            "subsys": "server_archive",
                            "event": "archive_worker_failed",
                            "detail": {"worker_id": worker_id},
                        },
                    )
                    for _ in batch:
                        self._queue.task_done()
        finally:
            logger.debug("Server archive worker stopped", extra={"worker_id": worker_id})
