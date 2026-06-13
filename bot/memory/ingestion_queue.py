"""Bounded async ingestion queue for curated memory candidates."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .curator import MemoryCandidate

logger = logging.getLogger(__name__)

PersistCallback = Callable[[list[MemoryCandidate]], Awaitable[None]]


@dataclass(slots=True)
class QueueStats:
    enqueued: int = 0
    processed: int = 0
    dropped: int = 0
    overflow: int = 0
    failed: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class CuratedMemoryIngestionQueue:
    """Small bounded queue with worker backpressure."""

    def __init__(
        self,
        persist_callback: PersistCallback,
        *,
        max_size: int = 256,
        workers: int = 1,
        batch_size: int = 8,
        enabled: bool = True,
    ) -> None:
        self.persist_callback = persist_callback
        self.max_size = max(1, int(max_size))
        self.workers = max(1, int(workers))
        self.batch_size = max(1, int(batch_size))
        self.enabled = enabled
        self._queue: asyncio.Queue[MemoryCandidate] = asyncio.Queue(maxsize=self.max_size)
        self._worker_tasks: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()
        self._start_lock = asyncio.Lock()
        self.stats = QueueStats()

    async def start(self) -> None:
        if not self.enabled:
            return
        async with self._start_lock:
            if self._worker_tasks:
                return
            self._shutdown.clear()
            for idx in range(self.workers):
                task = asyncio.create_task(self._worker_loop(idx), name=f"curated-memory-worker-{idx}")
                self._worker_tasks.append(task)
            logger.info(
                "Curated memory workers started",
                extra={
                    "subsys": "memory",
                    "event": "memory_queue_started",
                    "detail": {"workers": self.workers, "max_size": self.max_size},
                },
            )

    async def stop(self, timeout: float = 3.0) -> None:
        if not self._worker_tasks:
            return
        self._shutdown.set()
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "Timed out while draining curated memory queue",
                extra={"subsys": "memory", "event": "memory_queue_drain_timeout"},
            )
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    def qsize(self) -> int:
        return self._queue.qsize()

    async def enqueue(self, candidate: MemoryCandidate) -> bool:
        if not self.enabled:
            # Still process via callback path if queueing is disabled.
            await self.persist_callback([candidate])
            return True

        try:
            self._queue.put_nowait(candidate)
            self.stats.enqueued += 1
            return True
        except asyncio.QueueFull:
            self.stats.overflow += 1
            self.stats.dropped += 1
            logger.warning(
                "Curated memory queue full; dropping inferred candidate",
                extra={
                    "subsys": "memory",
                    "event": "memory_queue_overflow",
                    "detail": {
                        "memory_id": candidate.memory_id,
                        "source": candidate.source,
                    },
                },
            )
            return False

    async def _worker_loop(self, worker_id: int) -> None:
        logger.info("Curated memory worker starting", extra={"worker_id": worker_id})
        try:
            while not self._shutdown.is_set():
                batch: list[MemoryCandidate] = []
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    batch.append(item)
                    for _ in range(self.batch_size - 1):
                        try:
                            batch.append(self._queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    start = time.perf_counter()
                    await self.persist_callback(batch)
                    elapsed = time.perf_counter() - start
                    self.stats.processed += len(batch)
                    logger.debug(
                        "Curated memory batch persisted",
                        extra={
                            "subsys": "memory",
                            "event": "memory_batch_persisted",
                            "detail": {
                                "worker_id": worker_id,
                                "batch_size": len(batch),
                                "elapsed_ms": round(elapsed * 1000, 2),
                            },
                        },
                    )
                    for _ in batch:
                        self._queue.task_done()
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self.stats.failed += len(batch) or 1
                    logger.exception(
                        "Curated memory worker error",
                        extra={
                            "subsys": "memory",
                            "event": "memory_worker_error",
                            "detail": {"worker_id": worker_id, "error": str(exc)},
                        },
                    )
                    for _ in batch:
                        self._queue.task_done()
        finally:
            logger.info("Curated memory worker stopped", extra={"worker_id": worker_id})
