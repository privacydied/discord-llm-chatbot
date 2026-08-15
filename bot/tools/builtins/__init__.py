"""Built-in tool registrations.
[CA][SFT].

Every tool the bot exposes is listed here explicitly. There is no directory
scan, no entry-point discovery and no dynamic import: a tool exists because
someone added it to BUILTIN_SPECS by hand and to the registry allowlist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .clock import SPEC as CLOCK_SPEC
from .history import SPEC as HISTORY_SPEC

if TYPE_CHECKING:
    from ..registry import ToolRegistry
    from ..types import ToolSpec

# The literal, hand-maintained tool list. [SFT][CMV]
BUILTIN_SPECS: tuple[ToolSpec, ...] = (HISTORY_SPEC, CLOCK_SPEC)


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register every built-in tool into ``registry``."""
    for spec in BUILTIN_SPECS:
        registry.register(spec)


__all__ = ["BUILTIN_SPECS", "register_builtin_tools"]
