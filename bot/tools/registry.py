"""Allowlist-only tool registry.
[CA][SFT][IV][REH].

Two independent gates stand between the model and any capability:

1. **The allowlist.** ``ALLOWED_TOOL_NAMES`` below is the complete set of tools
   this bot may ever expose. Registration of any other name raises. Adding a
   capability is therefore a deliberate, reviewable edit to a constant in this
   file -- never an accident, and never something the model can do at runtime.

2. **Named dispatch only.** ``execute_tool`` looks the name up in a dict of
   pre-registered coroutines. There is no eval, no import-by-string, no
   getattr-on-module, and no shell. A name the model invents simply misses.

Consequently the model cannot delete files, run shell commands, or reach any
capability not written here by hand. See tests/test_tools_safety.py, which
fails the build if this file or any tool grows a dangerous import.
"""

from __future__ import annotations

from typing import Any

from bot.utils.logging import get_logger

from .types import DEFAULT_TOOL_TIMEOUT_S, ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

# The complete set of permissible tool names. Registration outside this set is
# rejected. Every entry here is read-only with respect to the host machine.
# [SFT][CMV]
ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_channel_history",
        "get_current_time",
        "view_image",
    }
)

# Fallback wall clock when a tool declares no budget of its own. [CMV][PA]
TOOL_TIMEOUT_S = DEFAULT_TOOL_TIMEOUT_S


class ToolRegistrationError(RuntimeError):
    """Raised when a tool is registered outside the allowlist or twice."""


class ToolRegistry:
    """Holds the registered tools. Not a plugin loader -- registration is manual."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Add a tool. Raises unless the name is allowlisted and unused. [SFT]"""
        if spec.name not in ALLOWED_TOOL_NAMES:
            msg = f"Tool {spec.name!r} is not in ALLOWED_TOOL_NAMES; refusing to register it. Add it to the allowlist deliberately if it is intended."
            raise ToolRegistrationError(msg)
        if spec.name in self._tools:
            msg = f"Tool {spec.name!r} is already registered"
            raise ToolRegistrationError(msg)
        self._tools[spec.name] = spec
        logger.debug("tool.registered name=%s", spec.name)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI-style tool definitions for every registered tool."""
        return [self._tools[name].to_openai_schema() for name in sorted(self._tools)]

    def clear(self) -> None:
        """Drop all registrations. For tests only."""
        self._tools.clear()


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Return the process-wide registry, populated on first use."""
    if not _registry.names():
        from .builtins import register_builtin_tools

        register_builtin_tools(_registry)
    return _registry


async def execute_tool(name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Run one registered tool by name. Never raises. [REH][SFT]

    An unknown name is a miss, not an error path into anything else -- there is
    no fallback that could interpret the name as code, a path, or a command.
    """
    import asyncio

    spec = get_registry().get(name)
    if spec is None:
        logger.warning("tool.unknown name=%s", str(name)[:60])
        return ToolResult.failure(f"unknown tool: {name}")

    if not isinstance(arguments, dict):
        return ToolResult.failure("arguments must be an object")

    try:
        budget = getattr(spec, "timeout_s", None) or TOOL_TIMEOUT_S
        return await asyncio.wait_for(spec.handler(ctx, arguments), timeout=budget)
    except TimeoutError:
        logger.warning("tool.timeout name=%s", name)
        return ToolResult.failure(f"{name} timed out")
    except Exception as exc:  # noqa: BLE001 - tool sandbox; a tool fault must not break the turn [REH]
        logger.warning("tool.failed name=%s error=%s", name, exc)
        return ToolResult.failure(f"{name} failed: {type(exc).__name__}")
