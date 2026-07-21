"""Cooperative deadline propagation for a message-processing pipeline. [PA][REH]

The router guards multimodal processing with a total wall-clock budget
(MULTIMODAL_TOTAL_BUDGET_S, asyncio.wait_for). Deep inside that task, the text
fallback ladder used to grant itself a fixed TEXT_PER_ITEM_BUDGET regardless of
how much of the outer budget earlier stages (video download, STT, VL) had
already consumed — so it would happily start ~120s of provider attempts with
30s left, guaranteeing a user-facing total_timeout instead of a fast, graceful
degradation.

A ContextVar carries the absolute deadline: set it where the outer wait_for
starts, and any downstream stage can clamp its own sub-budget to what's
actually left. ContextVars flow into tasks created after the value is set, so
the wait_for-wrapped task and everything it awaits see the deadline without
any parameter threading.
"""

from __future__ import annotations

import contextvars
import time

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("request_deadline", default=None)


def set_deadline(seconds_from_now: float) -> contextvars.Token:
    """Arm the deadline; returns a token for clear_deadline. Monotonic-based."""
    return _deadline.set(time.monotonic() + max(0.0, seconds_from_now))


def clear_deadline(token: contextvars.Token) -> None:
    """Disarm the deadline set by the matching set_deadline call. Never raises."""
    try:
        _deadline.reset(token)
    except ValueError:
        # Token from a different context (e.g. captured across tasks) — the
        # value dies with its context anyway, so this is safe to ignore.
        pass


def remaining_seconds() -> float | None:
    """Seconds left until the ambient deadline, or None if no deadline is armed."""
    deadline = _deadline.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
