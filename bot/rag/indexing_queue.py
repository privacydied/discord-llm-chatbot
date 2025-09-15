"""
Background indexing queue and workers for RAG system.

This module implements asynchronous document indexing with bounded concurrency,
backpressure handling, and graceful shutdown capabilities.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class IndexingTaskStatus(Enum):
    """Status of an indexing task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DROPPED = "dropped"


@dataclass
class IndexingTask:
    """A document indexing task."""

    source_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    file_type: str = "text"
    priority: int = 1  # Higher = more important
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    max_attempts: int = 3
    status: IndexingTaskStatus = IndexingTaskStatus.PENDING
    error_message: Optional[str] = None

    def __post_init__(self):
        """Ensure metadata is not None."""
        if self.metadata is None:
            self.metadata = {}


class IndexingQueue:
    """
    Asynchronous indexing queue with bounded workers and backpressure handling.

    Features:
    - Bounded queue size with overflow handling
    - Configurable number of worker threads
    - Batch processing for efficiency
    - Retry logic with exponential backoff
    - Graceful shutdown with task draining
    - Comprehensive metrics and logging
    """

    def __init__(
        self,
        rag_backend,
        max_queue_size: int = 1000,
        num_workers: int = 2,
        batch_size: int = 10,
        enabled: bool = True,
    ):
        self.rag_backend = rag_backend
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.enabled = enabled

        # Queue and worker management
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._worker_lock = asyncio.Lock()

        # Metrics and monitoring
        self._stats = {
            "tasks_enqueued": 0,
            "tasks_processed": 0,
            "tasks_failed": 0,
            "tasks_dropped": 0,
            "queue_overflows": 0,
            "worker_errors": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "started_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
        }

        logger.info(
            f"[RAG Indexing] Queue initialized: max_size={max_queue_size}, "
            f"workers={num_workers}, batch_size={batch_size}, enabled={enabled}"
        )

    async def start_workers(self) -> None:
        """Start the background worker tasks."""
        if not self.enabled:
            logger.info(
                "[RAG Indexing] Background indexing disabled, skipping worker startup"
            )
            return

        async with self._worker_lock:
            if self._workers:
                logger.warning("[RAG Indexing] Workers already started")
                return

            logger.info(
                f"[RAG Indexing] Starting {self.num_workers} background workers"
            )

            for i in range(self.num_workers):
                worker_task = asyncio.create_task(
                    self._worker_loop(worker_id=i), name=f"rag-indexing-worker-{i}"
                )
                self._workers.append(worker_task)

            logger.info(
                f"[RAG Indexing] ✔ {len(self._workers)} workers started successfully"
            )

    async def enqueue_task(self, task: IndexingTask) -> bool:
        """
        Enqueue a document for background indexing.

        Args:
            task: The indexing task to enqueue

        Returns:
            True if enqueued successfully, False if dropped due to overflow
        """
        if not self.enabled:
            # Synchronous processing when background indexing is disabled
            logger.debug(
                f"[RAG Indexing] Background indexing disabled, processing synchronously: {task.source_id}"
            )
            return await self._process_task_sync(task)

        try:
            # Try to enqueue without blocking
            self._queue.put_nowait(task)
            self._stats["tasks_enqueued"] += 1
            self._stats["last_activity"] = datetime.utcnow()

            logger.debug(
                f"[RAG Indexing] ✔ Enqueued task: {task.source_id} "
                f"(queue_depth={self._queue.qsize()})"
            )
            return True

        except asyncio.QueueFull:
            # Handle queue overflow
            self._stats["queue_overflows"] += 1
            self._stats["tasks_dropped"] += 1

            logger.warning(
                f"[RAG Indexing] ⚠ Queue overflow, dropping task: {task.source_id} "
                f"(queue_size={self._queue.qsize()}/{self.max_queue_size})"
            )

            # Could implement priority-based dropping here
            # For now, just drop the new task
            task.status = IndexingTaskStatus.DROPPED
            task.error_message = "Queue overflow"
            return False

    async def _process_task_sync(self, task: IndexingTask) -> bool:
        """Process a task synchronously when background indexing is disabled."""
        try:
            start_time = time.time()
            task.status = IndexingTaskStatus.PROCESSING

            # Process the document directly
            success = await self.rag_backend.add_document(
                source_id=task.source_id,
                text=task.text,
                metadata=task.metadata,
                file_type=task.file_type,
            )

            processing_time = time.time() - start_time

            if success:
                task.status = IndexingTaskStatus.COMPLETED
                self._stats["tasks_processed"] += 1
                logger.debug(
                    f"[RAG Indexing] ✔ Synchronously processed: {task.source_id} "
                    f"({processing_time:.2f}s)"
                )
                return True
            else:
                task.status = IndexingTaskStatus.FAILED
                task.error_message = "Backend processing failed"
                self._stats["tasks_failed"] += 1
                logger.error(
                    f"[RAG Indexing] ✖ Synchronous processing failed: {task.source_id}"
                )
                return False

        except Exception as e:
            task.status = IndexingTaskStatus.FAILED
            task.error_message = str(e)
            self._stats["tasks_failed"] += 1
            logger.error(
                f"[RAG Indexing] ✖ Synchronous processing error: {task.source_id} - {e}"
            )
            return False

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop for processing indexing tasks."""
        logger.info(f"[RAG Indexing] Worker {worker_id} started")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for tasks with timeout to allow shutdown checking
                    task = await asyncio.wait_for(self._queue.get(), timeout=1.0)

                    # Process the task
                    await self._process_task(task, worker_id)
                    self._queue.task_done()

                except asyncio.TimeoutError:
                    # No task available, continue loop to check shutdown
                    continue

                except Exception as e:
                    self._stats["worker_errors"] += 1
                    logger.error(f"[RAG Indexing] Worker {worker_id} error: {e}")

        except asyncio.CancelledError:
            logger.info(f"[RAG Indexing] Worker {worker_id} cancelled")
            raise

        finally:
            logger.info(f"[RAG Indexing] Worker {worker_id} stopped")

    async def _process_task(self, task: IndexingTask, worker_id: int) -> None:
        """Process a single indexing task with retry logic."""
        start_time = time.time()
        task.status = IndexingTaskStatus.PROCESSING
        task.attempts += 1

        try:
            logger.debug(
                f"[RAG Indexing] Worker {worker_id} processing: {task.source_id} "
                f"(attempt {task.attempts}/{task.max_attempts})"
            )

            # Process the document
            success = await self.rag_backend.add_document(
                source_id=task.source_id,
                text=task.text,
                metadata=task.metadata,
                file_type=task.file_type,
            )

            processing_time = time.time() - start_time
            self._stats["total_processing_time"] += processing_time
            self._stats["last_activity"] = datetime.utcnow()

            if success:
                task.status = IndexingTaskStatus.COMPLETED
                self._stats["tasks_processed"] += 1

                # Update average processing time
                if self._stats["tasks_processed"] > 0:
                    self._stats["avg_processing_time"] = (
                        self._stats["total_processing_time"]
                        / self._stats["tasks_processed"]
                    )

                logger.debug(
                    f"[RAG Indexing] ✔ Worker {worker_id} completed: {task.source_id} "
                    f"({processing_time:.2f}s)"
                )
            else:
                await self._handle_task_failure(
                    task, "Backend processing failed", worker_id
                )

        except Exception as e:
            await self._handle_task_failure(task, str(e), worker_id)

    async def _handle_task_failure(
        self, task: IndexingTask, error_message: str, worker_id: int
    ) -> None:
        """Handle task failure with retry logic."""
        task.error_message = error_message

        if task.attempts < task.max_attempts:
            # Retry with exponential backoff
            backoff_delay = min(2 ** (task.attempts - 1), 30)  # Cap at 30 seconds

            logger.warning(
                f"[RAG Indexing] Worker {worker_id} task failed, retrying in {backoff_delay}s: "
                f"{task.source_id} (attempt {task.attempts}/{task.max_attempts}) - {error_message}"
            )

            # Reset status for retry
            task.status = IndexingTaskStatus.PENDING

            # Re-enqueue after delay
            await asyncio.sleep(backoff_delay)
            try:
                self._queue.put_nowait(task)
            except asyncio.QueueFull:
                # If queue is full during retry, mark as failed
                task.status = IndexingTaskStatus.FAILED
                self._stats["tasks_failed"] += 1
                logger.error(
                    f"[RAG Indexing] Failed to re-enqueue task due to queue overflow: {task.source_id}"
                )
        else:
            # Max attempts reached
            task.status = IndexingTaskStatus.FAILED
            self._stats["tasks_failed"] += 1

            logger.error(
                f"[RAG Indexing] ✖ Worker {worker_id} task failed permanently: "
                f"{task.source_id} - {error_message}"
            )

    async def shutdown(self, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Gracefully shutdown the indexing queue.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            Shutdown statistics
        """
        if not self.enabled:
            logger.info(
                "[RAG Indexing] Background indexing was disabled, nothing to shutdown"
            )
            return {"status": "disabled"}

        logger.info(f"[RAG Indexing] Starting graceful shutdown (timeout={timeout}s)")
        shutdown_start = time.time()

        # Signal shutdown to workers
        self._shutdown_event.set()

        # Wait for queue to drain (with timeout)
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout / 2)
            logger.info("[RAG Indexing] ✔ Queue drained successfully")
        except asyncio.TimeoutError:
            logger.warning(
                f"[RAG Indexing] ⚠ Queue drain timeout, {self._queue.qsize()} tasks remaining"
            )

        # Cancel and wait for workers
        async with self._worker_lock:
            if self._workers:
                logger.info(f"[RAG Indexing] Cancelling {len(self._workers)} workers")

                for worker in self._workers:
                    worker.cancel()

                # Wait for workers to finish
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._workers, return_exceptions=True),
                        timeout=timeout / 2,
                    )
                    logger.info("[RAG Indexing] ✔ All workers stopped")
                except asyncio.TimeoutError:
                    logger.warning("[RAG Indexing] ⚠ Worker shutdown timeout")

                self._workers.clear()

        shutdown_time = time.time() - shutdown_start

        shutdown_stats = {
            "status": "completed",
            "shutdown_time": shutdown_time,
            "remaining_queue_size": self._queue.qsize(),
            "final_stats": self.get_stats(),
        }

        logger.info(f"[RAG Indexing] ✔ Shutdown completed in {shutdown_time:.2f}s")
        return shutdown_stats

    def get_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        uptime = (datetime.utcnow() - self._stats["started_at"]).total_seconds()

        return {
            "enabled": self.enabled,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "num_workers": len(self._workers),
            "uptime_seconds": uptime,
            **self._stats,
        }

    def is_healthy(self) -> bool:
        """Check if the indexing queue is healthy."""
        if not self.enabled:
            return True

        # Check if workers are running
        if not self._workers:
            return False

        # Check if any workers have died
        dead_workers = [w for w in self._workers if w.done()]
        if dead_workers:
            logger.warning(f"[RAG Indexing] Found {len(dead_workers)} dead workers")
            return False

        # Check queue overflow rate
        if self._stats["tasks_enqueued"] > 0:
            overflow_rate = (
                self._stats["queue_overflows"] / self._stats["tasks_enqueued"]
            )
            if overflow_rate > 0.1:  # More than 10% overflow
                logger.warning(
                    f"[RAG Indexing] High overflow rate: {overflow_rate:.2%}"
                )
                return False

        return True
