"""Owner-only web dashboard for operational visibility and controlled bot actions.

Exposes an aiohttp web server with:
- Bot health and metrics overview
- Guild inventory
- Audit log (append-only, filterable, paginated)
- DM archive (bot-visible conversations only)
- Owner-only DM and guild message sending
- CSRF-protected POST endpoints
- Rate-limited send actions
"""

from __future__ import annotations

from .config import DashboardConfig, load_dashboard_config
from .audit_store import AuditStore
from .dm_store import DMStore
from .server import DashboardServer

__all__ = [
    "DashboardConfig",
    "load_dashboard_config",
    "AuditStore",
    "DMStore",
    "DashboardServer",
]
