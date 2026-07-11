"""Vision Budget Manager - compatibility re-export shim. [CA][REH][RM][PA]

This module used to contain its own full `VisionBudgetManager` implementation
(non-Money floats, no atomic writes), but that class was never actually
instantiated -- every consumer resolves `VisionBudgetManager` to whatever this
module's bottom "compatibility re-export" section reassigns the name to, so the
original ~500-line class body was pure dead code sitting underneath its own
shim, silently diverging from what actually runs. It has been removed rather
than kept "for reference", per the same reasoning as `bot/logging_enforcer.py`:
a second, subtly different, unreachable implementation is a hazard (someone
"fixing a bug" here would be editing code nothing calls), not a safety net.

`bot/vision/budget_manager_v2.py` is the canonical, actually-live implementation
(Money type, atomic+fsync'd per-user file writes, imported directly by
`bot/vision/orchestrator.py`/`orchestrator_v2.py`). This module just re-exports
its public symbols so `from bot.vision.budget_manager import ...` / `from
bot.vision import ...` keep working unchanged.
"""

from __future__ import annotations

from .budget_manager_v2 import (
    BudgetResult,
    TransactionRecord,
    UserBudget,
    VisionBudgetManager,
)

__all__ = [
    "BudgetResult",
    "TransactionRecord",
    "UserBudget",
    "VisionBudgetManager",
]
