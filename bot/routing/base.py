"""Base types and protocol for extracted routing handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from discord import Message

    from ..modality import InputItem


@dataclass
class RouteContext:
    """Context passed to route handlers.

    Caries the resolved input item, original message, and runtime
    dependencies needed by handlers without requiring the full Router.
    """

    message: Optional[Message] = None
    author_id: Optional[int] = None
    source_type: Optional[str] = None
    payload: Any = None
    model_override: Optional[str] = None
    # Optional progress callback for streaming updates
    # signature: async (stage: str, step: int) -> None
    progress_cb: Optional[Any] = None
    item: Optional[InputItem] = None


@dataclass
class RouteResult:
    """Result returned by a route handler.

    text: The text/content payload for the bot to send back
    """

    text: str = ""

    @classmethod
    def text_only(cls, text: str) -> "RouteResult":
        """Convenience factory for a text-only result."""
        return cls(text=text)


class RouteHandler(Protocol):
    """Protocol for extracted route handlers.

    can_handle: Returns True when this handler can process the context
    handle: Async method that processes the context and returns a RouteResult
    """

    async def can_handle(self, ctx: RouteContext) -> bool: ...

    async def handle(self, ctx: RouteContext) -> RouteResult: ...
