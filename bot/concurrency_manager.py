"""
Bounded concurrency pools with hierarchical cancellation for router optimization. [PA][RM]

This module provides separate execution pools for different types of work:
- LIGHT: Planning, parsing, lightweight transforms (fast, CPU-light)
- NETWORK: HTTP requests, API calls, downloads (I/O-bound)
- HEAVY: OCR, STT, ffmpeg, model inference (CPU/memory-intensive)

Key features:
- Separate thread pools to prevent HEAVY work from blocking LIGHT/NETWORK
- Hierarchical cancellation: cancelling a branch cancels all children
- Backpressure control: HEAVY never starves LIGHT/NETWORK
- Per-pool metrics and monitoring
- Graceful degradation under load
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, TypeVar
from contextlib import asynccontextmanager

from .utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class PoolType(Enum):
    """Types of execution pools. [CMV]"""

    LIGHT = "light"  # Planning, parsing, lightweight transforms
    NETWORK = "network"  # HTTP requests, API calls, downloads
    HEAVY = "heavy"  # OCR, STT, ffmpeg, model inference


@dataclass
class PoolMetrics:
    """Metrics for monitoring pool usage. [PA]"""

    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0
    current_active: int = 0
    max_active_seen: int = 0
    total_execution_time: float = 0.0
    avg_execution_time_ms: float = 0.0
    queue_wait_time: float = 0.0
    pool_utilization: float = 0.0  # 0.0 to 1.0


@dataclass
class TaskContext:
    """Context for tracking individual tasks. [CA]"""

    task_id: str
    pool_type: PoolType
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    parent_context: Optional["TaskContext"] = None
    children: Set["TaskContext"] = field(default_factory=set)
    cancelled: bool = False
    result: Any = None
    error: Optional[Exception] = None


class CancellationTree:
    """Manages hierarchical task cancellation. [RM]"""

    def __init__(self):
        self.contexts: Dict[str, TaskContext] = {}
        self.root_contexts: Set[TaskContext] = set()

    def create_context(
        self, task_id: str, pool_type: PoolType, parent_id: Optional[str] = None
    ) -> TaskContext:
        """Create a new task context with optional parent. [CA]"""
        context = TaskContext(
            task_id=task_id, pool_type=pool_type, created_at=time.time()
        )

        if parent_id and parent_id in self.contexts:
            parent = self.contexts[parent_id]
            context.parent_context = parent
            parent.children.add(context)
        else:
            self.root_contexts.add(context)

        self.contexts[task_id] = context
        return context

    def cancel_branch(self, task_id: str) -> List[str]:
        """Cancel a task and all its children recursively. [RM]"""
        if task_id not in self.contexts:
            return []

        cancelled_ids = []
        context = self.contexts[task_id]

        # Cancel this task
        context.cancelled = True
        cancelled_ids.append(task_id)

        # Cancel all children recursively
        for child in context.children.copy():
            cancelled_ids.extend(self.cancel_branch(child.task_id))

        return cancelled_ids

    def cleanup_context(self, task_id: str) -> None:
        """Clean up completed task context. [RM]"""
        if task_id not in self.contexts:
            return

        context = self.contexts[task_id]

        # Remove from parent's children
        if context.parent_context:
            context.parent_context.children.discard(context)
        else:
            self.root_contexts.discard(context)

        # Remove from contexts map
        del self.contexts[task_id]


class BoundedExecutionPool:
    """Bounded execution pool with metrics and cancellation support. [PA][RM]"""

    def __init__(
        self,
        pool_type: PoolType,
        max_workers: int,
        thread_name_prefix: Optional[str] = None,
    ):
        self.pool_type = pool_type
        self.max_workers = max_workers
        self.metrics = PoolMetrics()
        self.cancellation_tree = CancellationTree()

        # Thread pool for CPU-bound work
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix or f"router-{pool_type.value}",
        )

        # Task tracking
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_futures: Dict[str, concurrent.futures.Future] = {}

        logger.info(
            f"🏊 {pool_type.value.upper()} pool initialized with {max_workers} workers"
        )

    async def submit_async(
        self,
        task_id: str,
        coro: Awaitable[T],
        parent_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> T:
        """Submit async coroutine to the pool. [PA]"""
        context = self.cancellation_tree.create_context(
            task_id, self.pool_type, parent_id
        )

        try:
            self.metrics.tasks_submitted += 1
            self.metrics.current_active += 1
            self.metrics.max_active_seen = max(
                self.metrics.max_active_seen, self.metrics.current_active
            )

            context.started_at = time.time()

            # Create task with optional timeout
            if timeout:
                task = asyncio.create_task(asyncio.wait_for(coro, timeout=timeout))
            else:
                task = asyncio.create_task(coro)

            self.active_tasks[task_id] = task

            # Wait for completion or cancellation
            try:
                result = await task
                context.completed_at = time.time()
                context.result = result

                self.metrics.tasks_completed += 1
                if context.started_at:
                    execution_time = context.completed_at - context.started_at
                    self.metrics.total_execution_time += execution_time
                    if self.metrics.tasks_completed > 0:
                        self.metrics.avg_execution_time_ms = (
                            self.metrics.total_execution_time
                            * 1000
                            / self.metrics.tasks_completed
                        )

                return result

            except asyncio.CancelledError:
                context.cancelled = True
                self.metrics.tasks_cancelled += 1
                raise
            except Exception as e:
                context.error = e
                self.metrics.tasks_failed += 1
                raise

        finally:
            self.metrics.current_active -= 1
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            self.cancellation_tree.cleanup_context(task_id)

    async def submit_sync(
        self,
        task_id: str,
        func: Callable[..., T],
        *args,
        parent_id: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> T:
        """Submit synchronous function to thread pool. [PA]"""
        context = self.cancellation_tree.create_context(
            task_id, self.pool_type, parent_id
        )

        try:
            self.metrics.tasks_submitted += 1
            self.metrics.current_active += 1
            self.metrics.max_active_seen = max(
                self.metrics.max_active_seen, self.metrics.current_active
            )

            context.started_at = time.time()

            # Submit to thread pool
            loop = asyncio.get_event_loop()
            future = self.thread_pool.submit(func, *args, **kwargs)
            self.task_futures[task_id] = future

            # Wait for completion with optional timeout
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, future.result), timeout=timeout
                    )
                else:
                    result = await loop.run_in_executor(None, future.result)

                context.completed_at = time.time()
                context.result = result

                self.metrics.tasks_completed += 1
                if context.started_at:
                    execution_time = context.completed_at - context.started_at
                    self.metrics.total_execution_time += execution_time
                    if self.metrics.tasks_completed > 0:
                        self.metrics.avg_execution_time_ms = (
                            self.metrics.total_execution_time
                            * 1000
                            / self.metrics.tasks_completed
                        )

                return result

            except asyncio.CancelledError:
                # Cancel the future
                future.cancel()
                context.cancelled = True
                self.metrics.tasks_cancelled += 1
                raise
            except Exception as e:
                context.error = e
                self.metrics.tasks_failed += 1
                raise

        finally:
            self.metrics.current_active -= 1
            if task_id in self.task_futures:
                del self.task_futures[task_id]
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            self.cancellation_tree.cleanup_context(task_id)

    def cancel_task_branch(self, task_id: str) -> List[str]:
        """Cancel a task and all its children. [RM]"""
        cancelled_ids = self.cancellation_tree.cancel_branch(task_id)

        for cancelled_id in cancelled_ids:
            # Cancel async task if it exists
            if cancelled_id in self.active_tasks:
                task = self.active_tasks[cancelled_id]
                if not task.done():
                    task.cancel()

            # Cancel thread pool future if it exists
            if cancelled_id in self.task_futures:
                future = self.task_futures[cancelled_id]
                if not future.done():
                    future.cancel()

        logger.info(
            f"🚫 Cancelled {len(cancelled_ids)} tasks in {self.pool_type.value} pool"
        )
        return cancelled_ids

    def get_metrics(self) -> PoolMetrics:
        """Get current pool metrics. [PA]"""
        self.metrics.pool_utilization = self.metrics.current_active / self.max_workers
        return self.metrics

    async def shutdown(self) -> None:
        """Shutdown the pool gracefully. [RM]"""
        logger.info(f"🛑 Shutting down {self.pool_type.value} pool...")

        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            self.cancel_task_branch(task_id)

        # Wait a bit for tasks to finish
        await asyncio.sleep(0.1)

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, cancel_futures=True)

        logger.info(f"✅ {self.pool_type.value} pool shutdown complete")


class ConcurrencyManager:
    """Manages all execution pools with hierarchical cancellation. [PA][RM]"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize concurrency manager with configuration."""
        self.config = config or {}

        # Pool configurations from environment
        light_workers = int(self.config.get("ROUTER_MAX_CONCURRENCY_LIGHT", 8))
        network_workers = int(self.config.get("ROUTER_MAX_CONCURRENCY_NETWORK", 32))
        heavy_workers = int(self.config.get("ROUTER_MAX_CONCURRENCY_HEAVY", 2))

        # Create pools
        self.pools = {
            PoolType.LIGHT: BoundedExecutionPool(PoolType.LIGHT, light_workers),
            PoolType.NETWORK: BoundedExecutionPool(PoolType.NETWORK, network_workers),
            PoolType.HEAVY: BoundedExecutionPool(PoolType.HEAVY, heavy_workers),
        }

        # Global task counter for unique IDs
        self._task_counter = 0
        self._task_lock = asyncio.Lock()

        logger.info("🎯 ConcurrencyManager initialized with all pools")

    async def _generate_task_id(self, prefix: str = "task") -> str:
        """Generate unique task ID. [CA]"""
        async with self._task_lock:
            self._task_counter += 1
            return f"{prefix}_{self._task_counter}_{int(time.time() * 1000)}"

    @asynccontextmanager
    async def submit_to_pool(
        self,
        pool_type: PoolType,
        task_name: str = "task",
        parent_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Context manager for submitting work to a specific pool. [CA]"""
        task_id = await self._generate_task_id(f"{pool_type.value}_{task_name}")
        pool = self.pools[pool_type]

        class PoolSubmitter:
            async def async_task(self, coro: Awaitable[T]) -> T:
                return await pool.submit_async(task_id, coro, parent_id, timeout)

            async def sync_task(self, func: Callable[..., T], *args, **kwargs) -> T:
                return await pool.submit_sync(
                    task_id, func, *args, parent_id=parent_id, timeout=timeout, **kwargs
                )

            def cancel(self) -> List[str]:
                return pool.cancel_task_branch(task_id)

        try:
            yield PoolSubmitter()
        except Exception:
            # Cancel the task branch on exception
            pool.cancel_task_branch(task_id)
            raise

    async def run_light_task(
        self,
        coro: Awaitable[T],
        task_name: str = "light_task",
        timeout: Optional[float] = None,
    ) -> T:
        """Run a lightweight task (planning, parsing). [PA]"""
        async with self.submit_to_pool(
            PoolType.LIGHT, task_name, timeout=timeout
        ) as submitter:
            return await submitter.async_task(coro)

    async def run_network_task(
        self,
        coro: Awaitable[T],
        task_name: str = "network_task",
        timeout: Optional[float] = None,
    ) -> T:
        """Run a network-bound task (HTTP, API calls). [PA]"""
        async with self.submit_to_pool(
            PoolType.NETWORK, task_name, timeout=timeout
        ) as submitter:
            return await submitter.async_task(coro)

    async def run_heavy_task(
        self,
        coro: Awaitable[T],
        task_name: str = "heavy_task",
        timeout: Optional[float] = None,
    ) -> T:
        """Run a heavy task (OCR, STT, ffmpeg). [PA]"""
        async with self.submit_to_pool(
            PoolType.HEAVY, task_name, timeout=timeout
        ) as submitter:
            return await submitter.async_task(coro)

    async def run_heavy_sync(
        self,
        func: Callable[..., T],
        *args,
        task_name: str = "heavy_sync",
        timeout: Optional[float] = None,
        **kwargs,
    ) -> T:
        """Run a heavy synchronous task in thread pool. [PA]"""
        async with self.submit_to_pool(
            PoolType.HEAVY, task_name, timeout=timeout
        ) as submitter:
            return await submitter.sync_task(func, *args, **kwargs)

    def get_all_metrics(self) -> Dict[PoolType, PoolMetrics]:
        """Get metrics for all pools. [PA]"""
        return {pool_type: pool.get_metrics() for pool_type, pool in self.pools.items()}

    def cancel_all_tasks_in_pool(self, pool_type: PoolType) -> int:
        """Cancel all tasks in a specific pool. [RM]"""
        pool = self.pools[pool_type]
        total_cancelled = 0

        for task_id in list(pool.active_tasks.keys()):
            cancelled_ids = pool.cancel_task_branch(task_id)
            total_cancelled += len(cancelled_ids)

        return total_cancelled

    async def shutdown_all(self) -> None:
        """Shutdown all pools gracefully. [RM]"""
        logger.info("🛑 Shutting down all execution pools...")

        # Shutdown in reverse order of dependency
        for pool_type in [PoolType.HEAVY, PoolType.NETWORK, PoolType.LIGHT]:
            await self.pools[pool_type].shutdown()

        logger.info("✅ All execution pools shutdown complete")


# Global singleton instance
_concurrency_manager_instance: Optional[ConcurrencyManager] = None


def get_concurrency_manager(
    config: Optional[Dict[str, Any]] = None,
) -> ConcurrencyManager:
    """Get or create the global concurrency manager instance. [CA]"""
    global _concurrency_manager_instance

    if _concurrency_manager_instance is None:
        _concurrency_manager_instance = ConcurrencyManager(config)

    return _concurrency_manager_instance


async def shutdown_concurrency_manager() -> None:
    """Shutdown the global concurrency manager."""
    global _concurrency_manager_instance

    if _concurrency_manager_instance is not None:
        await _concurrency_manager_instance.shutdown_all()
        _concurrency_manager_instance = None
