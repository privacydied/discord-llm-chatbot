"""Background task monitoring with heartbeat, watchdog, and restart policies.

Provides robust monitoring and automatic recovery for long-running background tasks
with heartbeat tracking, staleness detection, and configurable restart policies.

[RAT: REH, RM] - Robust Error Handling, Resource Management
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional
from concurrent.futures import CancelledError

from bot.util.logging import get_logger
from bot.metrics import (
    metrics,
    METRIC_BACKGROUND_HEARTBEAT,
    METRIC_BACKGROUND_LAST_HEARTBEAT,
    METRIC_BACKGROUND_CONSECUTIVE_ERRORS,
    METRIC_BACKGROUND_STALENESS_SECONDS,
)


class RestartPolicy(Enum):
    """Background task restart policy enumeration."""

    NEVER = "never"  # Never restart on failure
    ON_FAILURE = "on_failure"  # Restart only on unexpected failure
    ALWAYS = "always"  # Always restart when task exits
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Restart with exponential delay


class TaskStatus(Enum):
    """Background task status enumeration."""

    STARTING = "starting"
    RUNNING = "running"
    HEALTHY = "healthy"  # Running with recent heartbeat
    STALE = "stale"  # Running but no recent heartbeat
    FAILED = "failed"  # Task failed/crashed
    STOPPED = "stopped"  # Task stopped gracefully
    RESTARTING = "restarting"  # Task being restarted


@dataclass
class TaskConfig:
    """Configuration for a background task."""

    name: str
    task_func: Callable[[], Awaitable[Any]]
    heartbeat_interval: float = 30.0  # Seconds between heartbeats
    staleness_threshold: float = 90.0  # Seconds before considering stale
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    max_consecutive_failures: int = 5  # Max failures before giving up
    restart_delay_base: float = 1.0  # Base delay for exponential backoff
    restart_delay_max: float = 300.0  # Max delay for exponential backoff
    enabled: bool = True  # Whether task should run
    critical: bool = False  # Whether failure affects system health


@dataclass
class TaskState:
    """Runtime state of a background task."""

    config: TaskConfig
    status: TaskStatus = TaskStatus.STARTING
    task: Optional[asyncio.Task] = None
    last_heartbeat: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    restart_count: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    total_runtime: float = 0.0


class HeartbeatWrapper:
    """Wrapper that adds automatic heartbeat functionality to background tasks.

    [RAT: REH, RM] - Robust Error Handling, Resource Management
    """

    def __init__(self, task_name: str, monitor: BackgroundTaskMonitor):
        """Initialize heartbeat wrapper.

        Args:
            task_name: Name of the task being wrapped
            monitor: Background task monitor instance
        """
        self.task_name = task_name
        self.monitor = monitor
        self.logger = get_logger(__name__)

    async def __aenter__(self):
        """Enter heartbeat context."""
        self.logger.debug(
            f"💓 Starting heartbeat wrapper for {self.task_name}",
            extra={"subsys": "background_task"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit heartbeat context."""
        if exc_type:
            self.logger.warning(
                f"💓 Heartbeat wrapper exiting due to {exc_type.__name__}: {exc_val}",
                extra={"subsys": "background_task"},
            )
        else:
            self.logger.debug(
                f"💓 Heartbeat wrapper completed for {self.task_name}",
                extra={"subsys": "background_task"},
            )

    def heartbeat(self, message: Optional[str] = None) -> None:
        """Send heartbeat signal to monitor."""
        self.monitor.record_heartbeat(self.task_name, message)

    async def heartbeat_async(self, message: Optional[str] = None) -> None:
        """Send heartbeat signal asynchronously."""
        self.heartbeat(message)
        # Small yield to prevent blocking
        await asyncio.sleep(0)


class BackgroundTaskMonitor:
    """Comprehensive background task monitoring with watchdog and restart capabilities.

    Features:
    - Heartbeat tracking for each background task
    - Staleness detection with configurable thresholds
    - Automatic restart with multiple policy options
    - Exponential backoff for failed restarts
    - Resource cleanup on task failures
    - Comprehensive metrics and logging

    [RAT: REH, RM] - Robust Error Handling, Resource Management
    """

    def __init__(self, watchdog_interval: float = 10.0):
        """Initialize background task monitor.

        Args:
            watchdog_interval: Seconds between watchdog checks
        """
        self.logger = get_logger(__name__)
        self.tasks: Dict[str, TaskState] = {}
        self.watchdog_interval = watchdog_interval
        self.watchdog_task: Optional[asyncio.Task] = None
        self.running = False
        self._shutdown_event = asyncio.Event()

    def register_task(self, config: TaskConfig) -> None:
        """Register a background task for monitoring.

        Args:
            config: Task configuration with restart policy and thresholds
        """
        if config.name in self.tasks:
            raise ValueError(f"Task '{config.name}' already registered")

        self.tasks[config.name] = TaskState(config=config)

        self.logger.info(
            f"📋 Registered background task: {config.name} "
            f"(policy: {config.restart_policy.value}, critical: {config.critical})",
            extra={"subsys": "background_task", "task": config.name},
        )

    async def start_task(self, task_name: str) -> bool:
        """Start a registered background task.

        Args:
            task_name: Name of task to start

        Returns:
            True if task started successfully
        """
        if task_name not in self.tasks:
            self.logger.error(
                f"Cannot start unregistered task: {task_name}",
                extra={"subsys": "background_task"},
            )
            return False

        task_state = self.tasks[task_name]

        if not task_state.config.enabled:
            self.logger.info(
                f"Skipping disabled task: {task_name}",
                extra={"subsys": "background_task", "task": task_name},
            )
            return False

        if task_state.task and not task_state.task.done():
            self.logger.warning(
                f"Task {task_name} already running",
                extra={"subsys": "background_task", "task": task_name},
            )
            return True

        try:
            task_state.status = TaskStatus.STARTING
            task_state.start_time = time.time()
            task_state.last_heartbeat = time.time()

            # Wrap task function with error handling and metrics
            wrapped_func = self._wrap_task_function(task_state)
            task_state.task = asyncio.create_task(
                wrapped_func(), name=f"bg_{task_name}"
            )

            self.logger.info(
                f"🚀 Started background task: {task_name}",
                extra={"subsys": "background_task", "task": task_name},
            )

            # Send initial heartbeat
            self.record_heartbeat(task_name, "Task started")

            return True

        except Exception as e:
            task_state.status = TaskStatus.FAILED
            task_state.last_error = str(e)
            task_state.consecutive_failures += 1

            self.logger.error(
                f"Failed to start task {task_name}: {e}",
                exc_info=True,
                extra={"subsys": "background_task", "task": task_name},
            )

            metrics.increment(METRIC_BACKGROUND_CONSECUTIVE_ERRORS, {"task": task_name})

            return False

    def _wrap_task_function(
        self, task_state: TaskState
    ) -> Callable[[], Awaitable[None]]:
        """Wrap task function with monitoring, error handling, and cleanup."""

        async def wrapped_task():
            task_name = task_state.config.name

            try:
                task_state.status = TaskStatus.RUNNING

                self.logger.debug(
                    f"🔄 Executing task function: {task_name}",
                    extra={"subsys": "background_task", "task": task_name},
                )

                # Execute the actual task function
                await task_state.config.task_func()

                # Task completed normally
                task_state.status = TaskStatus.STOPPED
                task_state.total_runtime += time.time() - task_state.start_time

                self.logger.info(
                    f"✅ Task completed normally: {task_name}",
                    extra={"subsys": "background_task", "task": task_name},
                )

            except CancelledError:
                # Task was cancelled - this is normal during shutdown
                task_state.status = TaskStatus.STOPPED
                task_state.total_runtime += time.time() - task_state.start_time

                self.logger.info(
                    f"🛑 Task cancelled: {task_name}",
                    extra={"subsys": "background_task", "task": task_name},
                )
                raise  # Re-raise to properly handle cancellation

            except Exception as e:
                # Task failed unexpectedly
                task_state.status = TaskStatus.FAILED
                task_state.last_error = str(e)
                task_state.consecutive_failures += 1
                task_state.total_runtime += time.time() - task_state.start_time

                error_details = f"{type(e).__name__}: {e}"
                self.logger.error(
                    f"💥 Task failed: {task_name} - {error_details}",
                    exc_info=True,
                    extra={"subsys": "background_task", "task": task_name},
                )

                # Record failure metrics
                metrics.increment(
                    METRIC_BACKGROUND_CONSECUTIVE_ERRORS, {"task": task_name}
                )

                # Don't re-raise - let watchdog handle restart logic

        return wrapped_task

    def record_heartbeat(self, task_name: str, message: Optional[str] = None) -> None:
        """Record a heartbeat from a background task.

        Args:
            task_name: Name of task sending heartbeat
            message: Optional heartbeat message
        """
        if task_name not in self.tasks:
            self.logger.warning(
                f"Heartbeat from unknown task: {task_name}",
                extra={"subsys": "background_task"},
            )
            return

        task_state = self.tasks[task_name]
        current_time = time.time()

        task_state.last_heartbeat = current_time
        if task_state.status == TaskStatus.RUNNING:
            task_state.status = TaskStatus.HEALTHY

        # Record metrics
        metrics.increment(METRIC_BACKGROUND_HEARTBEAT, {"task": task_name})
        metrics.gauge(
            METRIC_BACKGROUND_LAST_HEARTBEAT, current_time, {"task": task_name}
        )

        log_msg = f"💓 Heartbeat: {task_name}"
        if message:
            log_msg += f" - {message}"

        self.logger.debug(
            log_msg, extra={"subsys": "background_task", "task": task_name}
        )

    def get_heartbeat_wrapper(self, task_name: str) -> HeartbeatWrapper:
        """Get heartbeat wrapper for a task.

        Args:
            task_name: Name of task requesting wrapper

        Returns:
            HeartbeatWrapper instance
        """
        return HeartbeatWrapper(task_name, self)

    async def start_watchdog(self) -> None:
        """Start the watchdog monitoring loop."""
        if self.watchdog_task and not self.watchdog_task.done():
            self.logger.warning(
                "Watchdog already running", extra={"subsys": "background_task"}
            )
            return

        self.running = True
        self.watchdog_task = asyncio.create_task(
            self._watchdog_loop(), name="bg_watchdog"
        )

        self.logger.info(
            f"🐕 Started background task watchdog (interval: {self.watchdog_interval}s)",
            extra={"subsys": "background_task"},
        )

    async def _watchdog_loop(self) -> None:
        """Main watchdog monitoring loop."""
        while self.running and not self._shutdown_event.is_set():
            try:
                await self._check_all_tasks()

                # Wait for next check or shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self.watchdog_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue monitoring

            except Exception as e:
                self.logger.error(
                    f"Watchdog error: {e}",
                    exc_info=True,
                    extra={"subsys": "background_task"},
                )
                await asyncio.sleep(self.watchdog_interval)

    async def _check_all_tasks(self) -> None:
        """Check health of all registered tasks and restart if needed."""
        current_time = time.time()

        for task_name, task_state in self.tasks.items():
            try:
                await self._check_single_task(task_name, task_state, current_time)
            except Exception as e:
                self.logger.error(
                    f"Error checking task {task_name}: {e}",
                    exc_info=True,
                    extra={"subsys": "background_task", "task": task_name},
                )

    async def _check_single_task(
        self, task_name: str, task_state: TaskState, current_time: float
    ) -> None:
        """Check health of a single task and restart if needed."""
        config = task_state.config

        # Skip disabled tasks
        if not config.enabled:
            return

        # Calculate staleness
        staleness = current_time - task_state.last_heartbeat
        metrics.gauge(
            METRIC_BACKGROUND_STALENESS_SECONDS, staleness, {"task": task_name}
        )

        # Check if task is stale
        if (
            staleness > config.staleness_threshold
            and task_state.status == TaskStatus.HEALTHY
        ):
            task_state.status = TaskStatus.STALE
            self.logger.warning(
                f"⏰ Task is stale: {task_name} (last heartbeat {staleness:.1f}s ago)",
                extra={"subsys": "background_task", "task": task_name},
            )

        # Check if task needs restart
        should_restart = False
        restart_reason = ""

        if task_state.task is None:
            should_restart = True
            restart_reason = "task not started"

        elif task_state.task.done():
            if task_state.task.cancelled():
                if config.restart_policy in (
                    RestartPolicy.ALWAYS,
                    RestartPolicy.EXPONENTIAL_BACKOFF,
                ):
                    should_restart = True
                    restart_reason = "task was cancelled"
            elif task_state.task.exception():
                if config.restart_policy != RestartPolicy.NEVER:
                    should_restart = True
                    restart_reason = f"task failed: {task_state.task.exception()}"
            else:
                # Task completed normally
                if config.restart_policy in (
                    RestartPolicy.ALWAYS,
                    RestartPolicy.EXPONENTIAL_BACKOFF,
                ):
                    should_restart = True
                    restart_reason = "task completed normally"

        elif (
            staleness > config.staleness_threshold * 2
        ):  # Double the threshold for restart
            should_restart = True
            restart_reason = f"task severely stale ({staleness:.1f}s)"

        # Check consecutive failure limit
        if (
            should_restart
            and task_state.consecutive_failures >= config.max_consecutive_failures
        ):
            should_restart = False
            if task_state.status != TaskStatus.FAILED:
                task_state.status = TaskStatus.FAILED
                self.logger.error(
                    f"🚫 Task {task_name} exceeded max failures ({config.max_consecutive_failures}), giving up",
                    extra={"subsys": "background_task", "task": task_name},
                )

        # Perform restart if needed
        if should_restart:
            await self._restart_task(task_name, task_state, restart_reason)

    async def _restart_task(
        self, task_name: str, task_state: TaskState, reason: str
    ) -> None:
        """Restart a background task with appropriate delay."""
        config = task_state.config

        # Calculate restart delay based on policy
        delay = 0.0
        if config.restart_policy == RestartPolicy.EXPONENTIAL_BACKOFF:
            delay = min(
                config.restart_delay_base * (2**task_state.restart_count),
                config.restart_delay_max,
            )

        if delay > 0:
            self.logger.info(
                f"🔄 Restarting {task_name} in {delay:.1f}s: {reason}",
                extra={"subsys": "background_task", "task": task_name},
            )
            await asyncio.sleep(delay)
        else:
            self.logger.info(
                f"🔄 Restarting {task_name}: {reason}",
                extra={"subsys": "background_task", "task": task_name},
            )

        # Cancel existing task if still running
        if task_state.task and not task_state.task.done():
            task_state.task.cancel()
            try:
                await asyncio.wait_for(task_state.task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # Expected

        # Increment restart counter
        task_state.restart_count += 1
        task_state.status = TaskStatus.RESTARTING

        # Start the task
        success = await self.start_task(task_name)
        if success:
            # Reset consecutive failures on successful restart
            task_state.consecutive_failures = 0

    async def stop_all_tasks(self, timeout: float = 10.0) -> None:
        """Stop all background tasks gracefully.

        Args:
            timeout: Maximum time to wait for tasks to stop
        """
        self.logger.info(
            "🛑 Stopping all background tasks...", extra={"subsys": "background_task"}
        )

        # Signal shutdown
        self.running = False
        self._shutdown_event.set()

        # Stop watchdog first
        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
            try:
                await asyncio.wait_for(self.watchdog_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cancel all background tasks
        tasks_to_cancel = []
        for task_name, task_state in self.tasks.items():
            if task_state.task and not task_state.task.done():
                task_state.task.cancel()
                tasks_to_cancel.append((task_name, task_state.task))

        if tasks_to_cancel:
            self.logger.info(
                f"Waiting for {len(tasks_to_cancel)} tasks to stop...",
                extra={"subsys": "background_task"},
            )

            # Wait for all tasks to complete cancellation
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *[task for _, task in tasks_to_cancel], return_exceptions=True
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Some tasks did not stop within {timeout}s timeout",
                    extra={"subsys": "background_task"},
                )

        self.logger.info(
            "✅ All background tasks stopped", extra={"subsys": "background_task"}
        )

    def get_task_status(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get status information for a specific task."""
        if task_name not in self.tasks:
            return None

        task_state = self.tasks[task_name]
        current_time = time.time()

        return {
            "name": task_name,
            "status": task_state.status.value,
            "enabled": task_state.config.enabled,
            "critical": task_state.config.critical,
            "restart_policy": task_state.config.restart_policy.value,
            "last_heartbeat": task_state.last_heartbeat,
            "staleness_seconds": current_time - task_state.last_heartbeat,
            "restart_count": task_state.restart_count,
            "consecutive_failures": task_state.consecutive_failures,
            "last_error": task_state.last_error,
            "total_runtime": task_state.total_runtime,
            "is_running": task_state.task is not None and not task_state.task.done(),
        }

    def get_all_task_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all registered tasks."""
        return {
            task_name: self.get_task_status(task_name)
            for task_name in self.tasks.keys()
        }


# Global background task monitor instance
_task_monitor: Optional[BackgroundTaskMonitor] = None


def get_task_monitor() -> BackgroundTaskMonitor:
    """Get global background task monitor instance."""
    global _task_monitor
    if _task_monitor is None:
        _task_monitor = BackgroundTaskMonitor()
    return _task_monitor
