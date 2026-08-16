"""Types for model-callable tools.
[CA][CMV][SFT].

A tool is a *named, pre-registered Python coroutine* with a JSON-schema
parameter contract. There is deliberately no generic "run this" primitive:
the model can only name a tool that already exists in the registry, and the
arguments it supplies are data passed to that coroutine -- never code, never a
path, never a command line.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# Hard ceiling on how much text one tool may return into the prompt. [CMV][PA]
MAX_TOOL_RESULT_CHARS = 4000


@dataclass(frozen=True)
class ToolContext:
    """The only handles a tool receives.

    Narrow on purpose: a tool gets the originating Discord message, the bot,
    and config. It gets no filesystem handle, no subprocess facility, and no
    database session.
    """

    message: Any = None
    bot: Any = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def channel(self) -> Any:
        return getattr(self.message, "channel", None)


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool invocation. Tools return this; they never raise."""

    ok: bool
    content: str = ""
    error: str | None = None

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        return cls(ok=False, content="", error=error)

    @classmethod
    def success(cls, content: str) -> ToolResult:
        text = content or ""
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]"
        return cls(ok=True, content=text)

    def to_message_content(self) -> str:
        """Render for the `tool` role turn sent back to the model."""
        return self.content if self.ok else f"ERROR: {self.error or 'tool failed'}"


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolResult]]


# Default per-invocation wall clock. Tools that call a model need far longer
# and override it; see ToolSpec.timeout_s. [CMV]
DEFAULT_TOOL_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: its contract and its implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    # Per-tool budget. A vision tool re-running inference cannot complete in
    # the default 10s, and would otherwise always time out. [PA]
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S

    def to_openai_schema(self) -> dict[str, Any]:
        """Render as an OpenAI-style function-tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
