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

# Seconds held back from a clamped sub-budget so the reply can still be
# dispatched after the ladder gives up, and the floor below which clamping
# would starve even a single fast attempt. [CMV]
DISPATCH_RESERVE_S = 10.0
LADDER_MIN_BUDGET_S = 8.0


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


def narrow_deadline(seconds_from_now: float) -> contextvars.Token:
    """Arm a deadline that can only *tighten* the ambient one; returns a reset token.

    Used to publish an inner guard (e.g. a per-item asyncio.wait_for) to nested
    stages, so a sub-ladder cannot re-grant itself more time than its enclosing
    wait_for will actually allow. Never widens an existing deadline. [REH][PA]
    """
    target = time.monotonic() + max(0.0, seconds_from_now)
    current = _deadline.get()
    if current is not None:
        target = min(current, target)
    return _deadline.set(target)


def clamp_to_deadline(
    budget: float,
    *,
    reserve: float = DISPATCH_RESERVE_S,
    floor: float = LADDER_MIN_BUDGET_S,
) -> tuple[float, float | None]:
    """Clamp a provider-ladder sub-budget to the ambient deadline. [PA][REH]

    Returns (effective_budget, ambient_remaining). ``ambient_remaining`` is None
    when no deadline is armed, in which case ``budget`` is returned untouched.
    Keeps ``reserve`` seconds so the reply can still be dispatched, and never
    goes below ``floor`` so at least one fast attempt runs.
    """
    ambient = remaining_seconds()
    if ambient is None:
        return budget, None
    clamped = max(floor, ambient - reserve)
    return (clamped if clamped < budget else budget), ambient
