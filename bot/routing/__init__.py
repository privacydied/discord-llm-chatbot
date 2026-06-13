"""Routing handlers extracted from bot/router.py.

Each handler implements the RouteHandler protocol:
- can_handle(ctx) -> bool
- async handle(ctx) -> RouteResult

Extraction protocol:
1. Router delegates to handler (preserves existing behavior)
2. Handler is independently testable
3. Router remains the compatibility shell
4. Imports outside router.py are preserved
"""

from .base import RouteContext, RouteHandler, RouteResult
from .screenshot_handler import ScreenshotHandler, handle_screenshot_url
from .unknown_handler import UnknownHandler

__all__ = [
    "RouteContext",
    "RouteHandler",
    "RouteResult",
    "ScreenshotHandler",
    "UnknownHandler",
    "handle_screenshot_url",
]
