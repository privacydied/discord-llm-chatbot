"""Tool: current UTC date and time.
[CA][CMV].

The model has no clock. Without this it cannot reason about "today", "this
week" or how stale its own training data is -- it will confidently date things
to its cutoff. Takes no arguments and touches nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..types import ToolContext, ToolResult, ToolSpec

PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

DESCRIPTION = "Get the current date and time in UTC. Use whenever the answer depends on what day or time it is now, such as questions about today, this week, or how recent something is."


async def get_current_time(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Return the current UTC timestamp. Never raises."""
    now = datetime.now(UTC)
    return ToolResult.success(f"{now.strftime('%A, %d %B %Y, %H:%M')} UTC (ISO: {now.isoformat(timespec='seconds')})")


SPEC = ToolSpec(
    name="get_current_time",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=get_current_time,
)
