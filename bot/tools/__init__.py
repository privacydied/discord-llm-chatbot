"""Model-callable tools.
[CA][SFT].

The model may call only the tools registered here, by name. It cannot delete
files, run shell commands, or reach any capability that is not written by hand
in bot/tools/builtins/ and listed in registry.ALLOWED_TOOL_NAMES.
"""

from __future__ import annotations

from .registry import ALLOWED_TOOL_NAMES, ToolRegistrationError, ToolRegistry, execute_tool, get_registry
from .types import ToolContext, ToolResult, ToolSpec

__all__ = [
    "ALLOWED_TOOL_NAMES",
    "ToolContext",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "execute_tool",
    "get_registry",
]
