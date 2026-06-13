"""Owner-only web dashboard for operational visibility and controlled bot actions.

Exposes an aiohttp web server with:
- Bot health and metrics overview
- Guild inventory
- Audit log (append-only, filterable, paginated)
- DM archive (bot-visible conversations only)
- Owner-only DM and guild message sending
- CSRF-protected POST endpoints
- Rate-limited send actions
- Unified message store (guild + DM)
- Backfill service for populating message history
"""

from __future__ import annotations

from .audit_store import AuditStore
from .backfill import BackfillJobStore, BackfillService
from .config import DashboardConfig, load_dashboard_config
from .dm_store import DMStore
from .message_store import MessageStore
from .permissions import (
    can_read_message_history,
    can_send_dm,
    can_send_messages,
    can_view_channel,
    get_channel_permissions,
)
from .redaction import (
    ContentSecurityPolicy,
    contains_mention_warning,
    make_preview,
    redact_secrets,
    sanitize_for_html,
)
from .server import DashboardServer

__all__ = [
    "AuditStore",
    "BackfillJobStore",
    "BackfillService",
    "ContentSecurityPolicy",
    "DMStore",
    "DashboardConfig",
    "DashboardServer",
    "MessageStore",
    "can_read_message_history",
    "can_send_dm",
    "can_send_messages",
    "can_view_channel",
    "contains_mention_warning",
    "get_channel_permissions",
    "load_dashboard_config",
    "make_preview",
    "redact_secrets",
    "sanitize_for_html",
]
