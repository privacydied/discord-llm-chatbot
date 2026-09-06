"""Central registry for all background tasks.

Provides a single view of running tasks for diagnostics and status commands.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Possible states for a tracked background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskEntry:
    """Metadata about a tracked background task."""

    name: str
    task: asyncio.Task
    feature: str  # which feature this task belongs to (e.g. "memory", "tts", "janitor")
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> TaskState:
        if self.task.done():
            try:
                exc = self.task.exception()
            except asyncio.CancelledError:
                return TaskState.CANCELLED
            except Exception:  # noqa: BLE001 - task-state probe must not raise; CancelledError handled above
                return TaskState.FAILED
            if exc is None:
                return TaskState.COMPLETED
            return TaskState.FAILED
        return TaskState.RUNNING


class BackgroundTaskRegistry:
    """Tracks all background tasks with feature tagging.

    Single instance managed at module level.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskEntry] = {}
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

    def register(
        self,
        task: asyncio.Task,
        *,
        name: str,
        feature: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskEntry:
        """Register a background task in the registry.

        Returns the TaskEntry (always non-None).
        If a task with the same name already exists and is still running,
        the old entry is replaced and the caller can decide whether to cancel it.
        """
        entry = TaskEntry(
            name=name,
            task=task,
            feature=feature,
            created_at=asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0,
            metadata=dict(metadata or {}),
        )

        # Replace or add
        old = self._tasks.get(name)
        if old is not None and not old.task.done():
            logger.debug(
                "background_task_registry.replace",
                extra={
                    "event": "background_task_registry.replace",
                    "task_name": name,
                    "detail": {"feature": feature},
                },
            )
        self._tasks[name] = entry

        # Auto-clean on completion
        task.add_done_callback(lambda _t: None)

        logger.debug(
            "background_task_registry.register",
            extra={
                "event": "background_task_registry.register",
                "task_name": name,
                "detail": {"feature": feature},
            },
        )
        return entry

    def unregister(self, name: str) -> TaskEntry | None:
        """Remove a task from the registry by name."""
        return self._tasks.pop(name, None)

    def get(self, name: str) -> TaskEntry | None:
        """Look up a task by name."""
        return self._tasks.get(name)

    def is_running(self, name: str) -> bool:
        """Return True if a task with the given name is registered and not done."""
        entry = self._tasks.get(name)
        return entry is not None and not entry.task.done()

    def list_tasks(self) -> list[dict[str, Any]]:
        """Return a serialisable snapshot of all registered tasks."""
        result = []
        for entry in self._tasks.values():
            try:
                done = entry.task.done()
                cancelled = entry.task.cancelled()
            except Exception:  # noqa: BLE001 - best-effort snapshot of possibly-dead tasks; defaults below
                done = True
                cancelled = False
            result.append(
                {
                    "name": entry.name,
                    "feature": entry.feature,
                    "state": entry.state.value,
                    "done": done,
                    "cancelled": cancelled,
                    "metadata": entry.metadata,
                },
            )
        return result

    def summary(self) -> dict[str, Any]:
        """Return a summary of tasks by feature."""
        features: dict[str, dict[str, int]] = {}
        for entry in self._tasks.values():
            f = entry.feature
            if f not in features:
                features[f] = {"total": 0, "running": 0, "done": 0, "cancelled": 0, "failed": 0}
            features[f]["total"] += 1
            s = entry.state
            if s == TaskState.RUNNING:
                features[f]["running"] += 1
            elif s == TaskState.COMPLETED:
                features[f]["done"] += 1
            elif s == TaskState.CANCELLED:
                features[f]["cancelled"] += 1
            elif s == TaskState.FAILED:
                features[f]["failed"] += 1

        return {
            "total_registered": len(self._tasks),
            "features": features,
        }

    def get_active_names(self) -> set[str]:
        """Return names of tasks that are still running."""
        return {name for name, entry in self._tasks.items() if not entry.task.done()}


# Global singleton
_registry: BackgroundTaskRegistry | None = None


def get_registry() -> BackgroundTaskRegistry:
    """Get the global task registry (creates one on first call)."""
    global _registry
    if _registry is None:
        _registry = BackgroundTaskRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (mainly for tests)."""
    global _registry
    _registry = None
