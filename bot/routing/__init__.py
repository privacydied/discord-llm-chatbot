"""Routing layer — extracted from bot/router.py (Phase 12).

All route handlers live here.  The router in bot/router.py delegates to
these handlers while the extraction is in progress to preserve backward
compatibility.
"""

from bot.routing.base import RouteContext, RouteHandler, RouteResult
from bot.routing.unknown_handler import UnknownHandler

__all__ = [
    "RouteContext",
    "RouteHandler",
    "RouteResult",
    "UnknownHandler",
]
