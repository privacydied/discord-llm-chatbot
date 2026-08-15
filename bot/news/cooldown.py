"""Per-user cooldown for the ambient news digest.
[CA][RM][CMV][PA].

The natural-language digest path has no discord.py command decorator, so it
gets no `@commands.cooldown` for free. Each trigger costs an upstream API call
against a rate-limited developer tier, so it needs its own throttle.
"""

from __future__ import annotations

import time

# Entries older than this are dropped on the next sweep, so the map cannot
# grow without bound in a busy guild. [RM][CMV]
_SWEEP_AFTER_S = 3600.0


class UserCooldown:
    """Allow one action per user per interval, with bounded memory."""

    def __init__(self) -> None:
        self._last: dict[int, float] = {}
        self._last_sweep: float = time.monotonic()

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < _SWEEP_AFTER_S:
            return
        cutoff = now - _SWEEP_AFTER_S
        self._last = {uid: ts for uid, ts in self._last.items() if ts > cutoff}
        self._last_sweep = now

    def allow(self, user_id: int | None, interval_s: float) -> bool:
        """Return True when the user may act now, recording the attempt."""
        if user_id is None or interval_s <= 0:
            return True
        now = time.monotonic()
        self._sweep(now)
        previous = self._last.get(user_id)
        if previous is not None and (now - previous) < interval_s:
            return False
        self._last[user_id] = now
        return True

    def reset(self, user_id: int | None = None) -> None:
        """Clear one user's cooldown, or all of them."""
        if user_id is None:
            self._last.clear()
        else:
            self._last.pop(user_id, None)


# Shared instance for the ambient digest path.
digest_cooldown = UserCooldown()
