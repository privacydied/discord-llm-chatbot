"""Repeated-warning suppression utility. [CA][REH].

NOTE: The dual-sink setup this module used to implement (`LoggingEnforcer`,
`setup_dual_sinks`, `initialize_logging`) is dead code -- `bot/utils/logging.py`
(wired up from `bot/main.py`) is the live, authoritative dual-sink implementation
and the one the "Dual Sink Strategy" mandate in CLAUDE.md actually enforces.
The old `setup_dual_sinks()` unconditionally forced the root logger and both
handlers to DEBUG, which -- if this module were ever imported by mistake instead
of `bot.utils.logging` -- would silently flip every `.debug()` call site in the
codebase (~900+) into a live disk write. It has been removed rather than fixed
in place, precisely so it can't be reintroduced by accident. [PA][REH]

What's left is `SuppressingLogger` / `_is_warning_suppressed`, a small,
independently tested rate-limiter for repeated warning messages (caps an
identical warning to once per `_SUPPRESS_WINDOW` seconds). It is not currently
installed as the active logger class anywhere, but is kept -- it's real,
covered by tests/test_logging_enforcer.py, and useful if a call site starts
emitting the same warning in a hot loop.
"""

from __future__ import annotations

import logging
import time

# ---- Repeated-warning suppressor [Phase 18] ----
# Caps identical warning messages to once per SUPPRESS_WINDOW seconds.
_SUPPRESS_WINDOW: float = 60.0
_warning_last_seen: dict[str, float] = {}


def _is_warning_suppressed(msg: str) -> bool:
    """Return True if this warning was already logged within the suppress window."""
    now = time.monotonic()
    prev = _warning_last_seen.get(msg)
    if prev is not None and (now - prev) < _SUPPRESS_WINDOW:
        return True
    _warning_last_seen[msg] = now
    return False


class SuppressingLogger(logging.Logger):
    """Logger subclass that rate-limits repeated warning/critical messages."""

    def warning(self, msg, *args, **kwargs) -> None:
        if _is_warning_suppressed(msg):
            return
        super().warning(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs) -> None:
        self.warning(msg, *args, **kwargs)


__all__ = [
    "SuppressingLogger",
    "_is_warning_suppressed",
]
