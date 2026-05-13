"""Handler for unknown or unsupported input items.

Extracted from ``bot/router.py:Router._handle_unknown`` as the first
handler in the extraction protocol (Phase 12).
"""

from __future__ import annotations


from bot.routing.base import RouteContext, RouteHandler, RouteResult


class UnknownHandler(RouteHandler):
    """Catch-all handler for input types no other handler could process."""

    def __init__(self, *, logger=None) -> None:
        self.logger = logger

    async def can_handle(self, ctx: RouteContext) -> bool:
        # Always acts as the final fallback — but only gets invoked if the
        # router explicitly routes an item here after no other handler matched.
        return True

    async def handle(self, ctx: RouteContext) -> RouteResult:
        """Log a warning and return a fallback message."""
        source = ctx.source_type
        payload_type = type(ctx.payload).__name__ if ctx.payload is not None else "None"

        if self.logger is not None:
            self.logger.warning(
                "Unknown input item type: %s with payload type %s",
                source,
                payload_type,
            )

        text = f"Unsupported input type detected: {source}. Unable to process this item."
        return RouteResult.text_only(text)
