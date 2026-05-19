"""API routes for the dashboard server.

All API endpoints enforce authentication (except /healthz and /api/session)
and CSRF protection (for POST endpoints).
"""

from __future__ import annotations

from pathlib import Path

from aiohttp import web

from bot.utils.logging import get_logger

from .audit_store import AuditStore
from .audit_store import (
    EVENT_DASHBOARD_CHANNEL_VIEW,
    EVENT_DASHBOARD_DM_VIEW,
    EVENT_DASHBOARD_BACKFILL_REQUESTED,
    EVENT_DASHBOARD_BACKFILL_STARTED,
    EVENT_DASHBOARD_BACKFILL_FAILED,
)
from .auth import _get_client_ip, _get_user_agent, auth_required, csrf_required, login_handler, logout_handler
from .config import DashboardConfig
from .dm_store import DMStore
from .message_store import MessageStore
from .services import DashboardServices

from bot.utils.playwright_helpers import get_playwright_health

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

# Security headers applied to all responses
_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' https://cdn.discordapp.com https://media.discordapp.net data:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'; "
        "base-uri 'self'; "
        "object-src 'none'"
    ),
}


def _with_security(response: web.StreamResponse) -> web.StreamResponse:
    """Apply security headers to a response."""
    for key, value in _SECURITY_HEADERS.items():
        response.headers[key] = value
    return response


def _json_response(data, status: int = 200) -> web.Response:
    """Create a JSON response with security headers."""
    return _with_security(web.json_response(data, status=status))


def _int_param(request: web.Request, name: str, default: int = 1) -> int:
    try:
        return int(request.query.get(name, default))
    except (ValueError, TypeError):
        return default


def _str_param(request: web.Request, name: str, default: str = "") -> str:
    return request.query.get(name, default) or default


async def _record_dm_view_audit(request: web.Request, target_channel_id: int) -> bool:
    """Record a DM view audit event. Returns True if recorded."""
    audit_store: AuditStore = request.app.get("audit_store")
    if audit_store is None:
        return False
    session = request.get("session", {})
    actor_id = session.get("user_id")
    try:
        await audit_store.record(
            event_type=EVENT_DASHBOARD_DM_VIEW,
            result="success",
            actor_user_id=actor_id,
            target_channel_id=target_channel_id,
        )
        return True
    except Exception:
        return False


class DashboardRoutes:
    """Request handlers for the dashboard."""

    def __init__(self, services: DashboardServices) -> None:
        self._services = services

    # ------------------------------------------------------------------
    # Public endpoints (no auth required)
    # ------------------------------------------------------------------

    async def healthz(self, request: web.Request) -> web.StreamResponse:
        """Minimal health check — no auth required."""
        config: DashboardConfig = request.app["dashboard_config"]
        bot = self._services.bot
        running = bot is not None and bot.is_ready()
        return _json_response(
            {
                "status": "ok" if running else "starting",
                "enabled": config.enabled,
                "uptime": self._get_uptime_simple(),
            }
        )

    def _get_uptime_simple(self) -> int:
        """Get uptime in seconds, best-effort."""
        bot = self._services.bot
        if bot and hasattr(bot, "ready_at") and bot.ready_at:
            from datetime import datetime, timezone

            return int((datetime.now(timezone.utc) - bot.ready_at).total_seconds())
        return 0

    async def session_check(self, request: web.Request) -> web.Response:
        """Return current session info if authenticated, or null."""
        config: DashboardConfig = request.app["dashboard_config"]
        session_store = request.app["session_store"]

        # Check bearer auth
        from .auth import _check_bearer_auth

        if config.auth_token and _check_bearer_auth(request, config.auth_token):
            return _json_response(
                {
                    "authenticated": True,
                    "method": "bearer",
                    "user_id": next(iter(config.owner_ids), None),
                }
            )

        # Check session cookie
        from .auth import _check_session_auth

        session = _check_session_auth(request, session_store)
        if session:
            return _json_response(
                {
                    "authenticated": True,
                    "method": "session",
                    "user_id": session.get("user_id"),
                    "csrf_token": session.get("csrf_token"),
                }
            )

        return _json_response({"authenticated": False})

    # ------------------------------------------------------------------
    # Static / Index
    # ------------------------------------------------------------------

    async def index(self, request: web.Request) -> web.StreamResponse:
        """Serve the dashboard HTML shell."""
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            resp = web.Response(text="Dashboard HTML not found", status=500)
            return _with_security(resp)
        return _with_security(web.FileResponse(index_path))

    async def static(self, request: web.Request) -> web.StreamResponse:
        """Serve static files."""
        filename = request.match_info.get("filename", "")
        filepath = STATIC_DIR / filename
        # Prevent directory traversal
        try:
            filepath.resolve().relative_to(STATIC_DIR.resolve())
        except ValueError:
            return _json_response({"error": "Not found"}, 404)
        if not filepath.exists():
            return _json_response({"error": "Not found"}, 404)
        config: DashboardConfig = request.app["dashboard_config"]
        resp = web.FileResponse(filepath)
        if config.static_cache_seconds > 0:
            resp.headers["Cache-Control"] = f"public, max-age={config.static_cache_seconds}"
        return _with_security(resp)

    # ------------------------------------------------------------------
    # API: Overview
    # ------------------------------------------------------------------

    @auth_required
    async def get_overview(self, request: web.Request) -> web.Response:
        """Rich overview data about bot status with guild list and metrics."""
        summary = await self._services.get_summary()

        # Collect guild list for overview cards
        bot = self._services.bot
        guilds = []
        if bot and bot.is_ready():
            for g in bot.guilds:
                try:
                    guilds.append({
                        "id": str(g.id),
                        "name": g.name,
                        "icon_url": str(g.icon.url) if g.icon else None,
                        "member_count": g.member_count,
                    })
                except Exception:
                    pass

        # Bot user info
        bot_username = summary.get("bot_username", "unknown")
        bot_discriminator = None
        bot_user_id = None
        if bot and bot.user:
            try:
                bot_username = bot.user.display_name or bot.user.name or "unknown"
                bot_discriminator = bot.user.discriminator if hasattr(bot.user, "discriminator") and bot.user.discriminator and bot.user.discriminator != "0" else None
                bot_user_id = str(bot.user.id)
            except Exception:
                pass

        # Collect recent errors from audit log
        audit_store: AuditStore = request.app.get("audit_store")
        recent_errors = []
        if audit_store:
            try:
                error_result = await audit_store.query(
                    page=1,
                    page_size=5,
                    result="failed",
                )
                recent_errors = error_result.get("events", [])
            except Exception:
                pass

        # Check Playwright status
        pw_health = get_playwright_health()

        overview = {
            **summary,
            "total_guilds": summary.get("guild_count", 0),
            "total_channels": summary.get("channel_count", 0),
            "total_dms": summary.get("dm_count", 0),
            "total_archived_messages": summary.get("archived_message_count", 0),
            "total_audit_events": summary.get("audit_event_count", 0),
            "bot_uptime_seconds": summary.get("uptime_seconds", 0),
            "bot_username": bot_username,
            "bot_discriminator": bot_discriminator,
            "bot_user_id": bot_user_id,
            "guilds": guilds,
            "recent_errors": recent_errors,
            "playwright_available": pw_health.get("available"),
            "playwright_degraded": pw_health.get("degraded", False),
        }
        return _json_response(overview)

    # ------------------------------------------------------------------
    # API: Summary / Metrics (legacy)
    # ------------------------------------------------------------------

    @auth_required
    async def get_summary(self, request: web.Request) -> web.Response:
        summary = await self._services.get_summary()
        return _json_response(summary)

    @auth_required
    async def get_metrics(self, request: web.Request) -> web.Response:
        """Return basic metrics summary."""
        summary = await self._services.get_summary()
        return _json_response(
            {
                "uptime_seconds": summary.get("uptime_seconds", 0),
                "guild_count": summary.get("guild_count", 0),
                "latency_ms": summary.get("latency_ms", 0),
                "cog_count": summary.get("cog_count", 0),
            }
        )

    # ------------------------------------------------------------------
    # API: Guilds
    # ------------------------------------------------------------------

    @auth_required
    async def get_guilds(self, request: web.Request) -> web.Response:
        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)
        search = _str_param(request, "search")

        result = await self._services.get_guilds(page=page, page_size=page_size, max_page_size=MAX_PAGE_SIZE, search=search or None)
        return _json_response(result)

    @auth_required
    async def get_guild_detail(self, request: web.Request) -> web.Response:
        guild_id_str = request.match_info.get("guild_id", "")
        try:
            guild_id = int(guild_id_str)
        except ValueError:
            return _json_response({"error": "invalid guild_id"}, 400)

        bot = self._services.bot
        if bot is None:
            return _json_response({"error": "bot not ready"}, 503)

        guild = bot.get_guild(guild_id)
        if guild is None:
            return _json_response({"error": "guild not found"}, 404)

        # Group channels by category
        categories = {}
        uncategorized = []

        try:
            for ch in guild.channels:
                try:
                    perms = ch.permissions_for(guild.me) if hasattr(ch, "permissions_for") else None
                    ch_type = str(getattr(ch, "type", "text"))
                    channel_data = {
                        "id": str(ch.id),
                        "name": ch.name,
                        "type": ch_type,
                        "position": getattr(ch, "position", 0),
                        "can_send": bool(perms and perms.send_messages) if perms else False,
                        "can_read": bool(perms and perms.read_messages) if perms else False,
                        "topic": getattr(ch, "topic", None),
                        "nsfw": getattr(ch, "nsfw", False),
                    }

                    # Check if this channel has a category
                    category_id = None
                    if hasattr(ch, "category") and ch.category:
                        category_id = str(ch.category.id)

                    if category_id:
                        if category_id not in categories:
                            cat = ch.category
                            categories[category_id] = {
                                "id": str(cat.id),
                                "name": cat.name,
                                "position": cat.position,
                                "channels": [],
                            }
                        categories[category_id]["channels"].append(channel_data)
                    else:
                        uncategorized.append(channel_data)
                except Exception:
                    pass
        except Exception:
            pass

        # Sort categories by position
        sorted_categories = sorted(categories.values(), key=lambda c: c["position"])
        # Sort channels within each category by position
        for cat in sorted_categories:
            cat["channels"].sort(key=lambda ch: ch.get("position", 0))
        uncategorized.sort(key=lambda ch: ch.get("position", 0))

        return _json_response(
            {
                "id": str(guild.id),
                "name": guild.name,
                "owner_id": str(guild.owner_id) if guild.owner_id else None,
                "member_count": guild.member_count,
                "icon_url": str(guild.icon.url) if guild.icon else None,
                "banner_url": str(guild.banner.url) if guild.banner else None,
                "description": guild.description,
                "features": list(guild.features) if guild.features else [],
                "premium_tier": guild.premium_tier,
                "premium_subscription_count": guild.premium_subscription_count,
                "approximate_member_count": guild.approximate_member_count,
                "approximate_presence_count": guild.approximate_presence_count,
                "categories": sorted_categories,
                "uncategorized_channels": uncategorized,
            }
        )

    @auth_required
    async def get_guild_channels(self, request: web.Request) -> web.Response:
        """Detailed channel list with categories and permissions."""
        guild_id_str = request.match_info.get("guild_id", "")
        try:
            guild_id = int(guild_id_str)
        except ValueError:
            return _json_response({"error": "invalid guild_id"}, 400)

        bot = self._services.bot
        if bot is None:
            return _json_response({"error": "bot not ready"}, 503)

        guild = bot.get_guild(guild_id)
        if guild is None:
            return _json_response({"error": "guild not found"}, 404)

        channels = []
        try:
            for ch in guild.channels:
                try:
                    perms = ch.permissions_for(guild.me) if hasattr(ch, "permissions_for") else None
                    ch_type = str(getattr(ch, "type", "text"))
                    category_id = None
                    category_name = None
                    if hasattr(ch, "category") and ch.category:
                        category_id = str(ch.category.id)
                        category_name = ch.category.name

                    channels.append(
                        {
                            "id": str(ch.id),
                            "name": ch.name,
                            "type": ch_type,
                            "position": getattr(ch, "position", 0),
                            "category_id": category_id,
                            "category_name": category_name,
                            "topic": getattr(ch, "topic", None),
                            "nsfw": getattr(ch, "nsfw", False),
                            "can_send_messages": bool(perms and perms.send_messages) if perms else False,
                            "can_read_messages": bool(perms and perms.read_messages) if perms else False,
                            "can_read_history": bool(perms and perms.read_message_history) if perms else False,
                            "can_embed_links": bool(perms and perms.embed_links) if perms else False,
                            "can_attach_files": bool(perms and perms.attach_files) if perms else False,
                            "is_administrator": bool(perms and perms.administrator) if perms else False,
                            "member_count": getattr(ch, "members", None) and len(ch.members),
                        }
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # Sort by position
        channels.sort(key=lambda ch: ch.get("position", 0))

        return _json_response({"channels": channels, "total": len(channels)})

    # ------------------------------------------------------------------
    # API: Channel Messages (from MessageStore)
    # ------------------------------------------------------------------

    @auth_required
    async def get_channel_messages(self, request: web.Request) -> web.Response:
        """Paginated message history from MessageStore for a guild channel.

        Supports auto_backfill=true to live-fetch from Discord when store is empty.
        """
        channel_id_str = request.match_info.get("channel_id", "")
        try:
            channel_id = int(channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid channel_id"}, 400)

        message_store: MessageStore = request.app.get("message_store")
        if message_store is None:
            return _json_response({"error": "message store not available"}, 500)

        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)
        before_id_str = _str_param(request, "before_id")
        after_id_str = _str_param(request, "after_id")
        auto_backfill = _str_param(request, "auto_backfill", "true") == "true"

        before_id = int(before_id_str) if before_id_str else None
        after_id = int(after_id_str) if after_id_str else None

        result = await message_store.get_channel_messages(
            channel_id=channel_id,
            page=page,
            page_size=page_size,
            max_page_size=MAX_PAGE_SIZE,
            before_id=before_id,
            after_id=after_id,
        )

        # Auto-backfill when store is empty and page=1 (first load)
        if auto_backfill and page == 1 and result.get("total", 0) == 0:
            bot = self._services.bot
            if bot is not None and bot.is_ready():
                channel = bot.get_channel(channel_id)
                if channel is not None:
                    guild = getattr(channel, "guild", None)
                    if guild is not None:
                        try:
                            perms = channel.permissions_for(guild.me)
                            if perms and perms.read_message_history:
                                live = await self._services.live_channel_messages(channel_id=channel_id, limit=50)
                                if live.get("messages"):
                                    # Return live messages directly instead of re-querying the store,
                                    # avoiding a race where the thread-pool SQLite write hasn't committed yet.
                                    result = {
                                        "messages": live["messages"],
                                        "channel_id": str(channel_id),
                                        "page": page,
                                        "page_size": page_size,
                                        "total": live.get("count", len(live["messages"])),
                                        "total_pages": 1,
                                    }
                            else:
                                logger.debug("get_channel_messages: no read_message_history permission for channel %s", channel_id)
                        except Exception as e:
                            logger.debug("get_channel_messages: auto-backfill failed for channel %s: %s", channel_id, e)
                    else:
                        # DM channel — try live DM fetch
                        live = await self._services.live_dm_messages(channel_id=channel_id, limit=50)
                        if live.get("messages"):
                            result = await message_store.get_channel_messages(
                                channel_id=channel_id,
                                page=page,
                                page_size=page_size,
                                max_page_size=MAX_PAGE_SIZE,
                                before_id=before_id,
                                after_id=after_id,
                            )

        # Audit log channel view
        audit_store: AuditStore = request.app.get("audit_store")
        session = request.get("session", {})
        actor_id = session.get("user_id")
        if audit_store:
            try:
                await audit_store.record(
                    event_type=EVENT_DASHBOARD_CHANNEL_VIEW,
                    result="success",
                    actor_user_id=actor_id,
                    target_channel_id=channel_id,
                )
            except Exception:
                pass

        return _json_response(result)

    # ------------------------------------------------------------------
    # API: DMs
    # ------------------------------------------------------------------

    @auth_required
    async def get_dms(self, request: web.Request) -> web.Response:
        """List DM threads with richer data."""
        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)

        config: DashboardConfig = request.app["dashboard_config"]
        if not config.dm_archive_enabled:
            return _json_response({"error": "DM archive disabled"}, 403)

        # Try MessageStore DM threads first (richer), fall back to DMStore
        message_store: MessageStore = request.app.get("message_store")
        if message_store:
            try:
                result = await message_store.get_dm_threads(
                    page=page,
                    page_size=page_size,
                    max_page_size=MAX_PAGE_SIZE,
                )
                return _json_response(result)
            except Exception:
                pass

        dm_store: DMStore = request.app.get("dm_store")
        if dm_store is None:
            return _json_response({"error": "DM store not available"}, 500)

        result = await dm_store.list_threads(page=page, page_size=page_size, max_page_size=MAX_PAGE_SIZE)
        return _json_response(result)

    @auth_required
    async def get_dm_thread(self, request: web.Request) -> web.Response:
        """Get paginated DM messages for a user or channel, with live fallback."""
        user_id_or_channel_id_str = request.match_info.get("user_id_or_channel_id", "")
        try:
            user_id_or_channel_id = int(user_id_or_channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid user_id or channel_id"}, 400)

        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)
        auto_backfill = _str_param(request, "auto_backfill", "true") == "true"

        # Try MessageStore first
        message_store: MessageStore = request.app.get("message_store")
        if message_store:
            try:
                result = await message_store.get_dm_thread_messages(
                    dm_channel_id=user_id_or_channel_id,
                    page=page,
                    page_size=page_size,
                    max_page_size=MAX_PAGE_SIZE,
                )
                if result.get("total", 0) > 0:
                    # Audit log DM view
                    _audit: AuditStore = request.app.get("audit_store")
                    _session = request.get("session", {})
                    _actor_id = _session.get("user_id")
                    if _audit:
                        try:
                            await _audit.record(
                                event_type=EVENT_DASHBOARD_DM_VIEW,
                                result="success",
                                actor_user_id=_actor_id,
                                target_channel_id=user_id_or_channel_id,
                            )
                        except Exception:
                            pass
                    return _json_response(result)

                # Auto-backfill when empty and page=1
                if auto_backfill and page == 1:
                    live = await self._services.live_dm_messages(channel_id=user_id_or_channel_id, limit=50)
                    if live.get("messages"):
                        # Also fetch from DMStore for the response
                        dm_store: DMStore = request.app.get("dm_store")
                        if dm_store:
                            result2 = await dm_store.get_thread_messages(
                                channel_id=user_id_or_channel_id,
                                page=page,
                                page_size=page_size,
                                max_page_size=MAX_PAGE_SIZE,
                            )
                            if result2.get("total", 0) > 0:
                                return_audit = await _record_dm_view_audit(request, user_id_or_channel_id)
                                if return_audit:
                                    pass
                                return _json_response(result2)

                        # Fallback: re-query message store
                        result = await message_store.get_dm_thread_messages(
                            dm_channel_id=user_id_or_channel_id,
                            page=page,
                            page_size=page_size,
                            max_page_size=MAX_PAGE_SIZE,
                        )
                        return_audit = await _record_dm_view_audit(request, user_id_or_channel_id)
                        return _json_response(result)

                    # live_dm_messages returned empty — maybe user_id_or_channel_id is a user ID, not a channel ID
                    # Try to resolve it to a DM channel
                    bot = self._services.bot
                    if bot and bot.is_ready():
                        import asyncio

                        try:
                            # Try to find user
                            user = bot.get_user(user_id_or_channel_id)
                            if user is None:
                                user = await asyncio.wait_for(
                                    bot.fetch_user(user_id_or_channel_id), timeout=10.0
                                )
                            if user is not None:
                                # Get or create DM channel
                                dm_channel = user.dm_channel
                                if dm_channel is None:
                                    dm_channel = await asyncio.wait_for(
                                        user.create_dm(), timeout=10.0
                                    )
                                if dm_channel is not None:
                                    # Retry live fetch with the real DM channel ID
                                    live2 = await self._services.live_dm_messages(
                                        channel_id=dm_channel.id, limit=50
                                    )
                                    if live2.get("messages"):
                                        # Fetch from DMStore using the real channel ID
                                        dm_store: DMStore = request.app.get("dm_store")
                                        if dm_store:
                                            result3 = await dm_store.get_thread_messages(
                                                channel_id=dm_channel.id,
                                                page=page,
                                                page_size=page_size,
                                                max_page_size=MAX_PAGE_SIZE,
                                            )
                                            if result3.get("total", 0) > 0:
                                                return_audit = await _record_dm_view_audit(request, user_id_or_channel_id)
                                                return _json_response(result3)
                                        # Fallback to MessageStore
                                        result = await message_store.get_dm_thread_messages(
                                            dm_channel_id=dm_channel.id,
                                            page=page,
                                            page_size=page_size,
                                            max_page_size=MAX_PAGE_SIZE,
                                        )
                                        return_audit = await _record_dm_view_audit(request, user_id_or_channel_id)
                                        return _json_response(result)
                        except Exception:
                            pass
            except Exception:
                pass

        # Fall back to DMStore
        dm_store: DMStore = request.app.get("dm_store")
        if dm_store is None:
            return _json_response({"error": "DM store not available"}, 500)

        config: DashboardConfig = request.app["dashboard_config"]
        if not config.dm_archive_enabled:
            return _json_response({"error": "DM archive disabled"}, 403)

        result = await dm_store.get_thread_messages(
            channel_id=user_id_or_channel_id,
            page=page,
            page_size=page_size,
            max_page_size=MAX_PAGE_SIZE,
        )
        return_audit = await _record_dm_view_audit(request, user_id_or_channel_id)
        return _json_response(result)

    @auth_required
    async def get_dm_thread_by_user_id(self, request: web.Request) -> web.Response:
        """Get DM messages for a specific user (legacy route)."""
        user_id_str = request.match_info.get("user_id", "")
        try:
            user_id = int(user_id_str)
        except ValueError:
            return _json_response({"error": "invalid user_id"}, 400)

        dm_store: DMStore = request.app.get("dm_store")
        if dm_store is None:
            return _json_response({"error": "DM store not available"}, 500)

        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)

        result = await dm_store.get_thread_messages(
            channel_id=user_id,
            page=page,
            page_size=page_size,
            max_page_size=MAX_PAGE_SIZE,
        )
        return _json_response(result)

    # ------------------------------------------------------------------
    # API: Message Search
    # ------------------------------------------------------------------

    @auth_required
    async def search_messages(self, request: web.Request) -> web.Response:
        """Search across all archived messages."""
        query = _str_param(request, "q")
        if not query:
            return _json_response({"error": "query parameter 'q' is required"}, 400)

        message_store: MessageStore = request.app.get("message_store")
        if message_store is None:
            return _json_response({"error": "message store not available"}, 500)

        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)
        guild_id_str = _str_param(request, "guild_id")
        channel_id_str = _str_param(request, "channel_id")
        author_id_str = _str_param(request, "author_id")

        guild_id = int(guild_id_str) if guild_id_str else None
        channel_id = int(channel_id_str) if channel_id_str else None
        author_id = int(author_id_str) if author_id_str else None

        result = await message_store.search_messages(
            query=query,
            page=page,
            page_size=page_size,
            max_page_size=MAX_PAGE_SIZE,
            guild_id=guild_id,
            channel_id=channel_id,
            author_id=author_id,
        )
        return _json_response(result)

    # ------------------------------------------------------------------
    # API: Send (POST)
    # ------------------------------------------------------------------

    @auth_required
    @csrf_required
    async def send_to_channel(self, request: web.Request) -> web.Response:
        """Send a message to a guild channel."""
        channel_id_str = request.match_info.get("channel_id", "")
        try:
            channel_id = int(channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid channel_id"}, 400)

        try:
            body = await request.json()
        except Exception:
            return _json_response({"error": "invalid json body"}, 400)

        content = body.get("content", "").strip()
        if not content:
            return _json_response({"error": "content is required"}, 400)

        session = request.get("session", {})
        actor_id = session.get("user_id")

        bot = self._services.bot
        if bot is None:
            return _json_response({"error": "bot not ready"}, 503)

        # Resolve guild_id from channel
        channel = bot.get_channel(channel_id)
        guild_id = channel.guild.id if channel and hasattr(channel, "guild") and channel.guild else None
        if guild_id is None:
            return _json_response({"error": "channel not found or not a guild channel"}, 404)

        result = await self._services.send_guild_message(
            guild_id=guild_id,
            channel_id=channel_id,
            content=content,
            actor_id=actor_id,
            source_ip=_get_client_ip(request),
            user_agent=_get_user_agent(request),
        )

        status_code = 200 if result.get("success") else 400
        return _json_response(result, status=status_code)

    @auth_required
    @csrf_required
    async def send_dm_to_user(self, request: web.Request) -> web.Response:
        """Send a DM to a user."""
        user_id_str = request.match_info.get("user_id", "")
        try:
            user_id = int(user_id_str)
        except ValueError:
            return _json_response({"error": "invalid user_id"}, 400)

        try:
            body = await request.json()
        except Exception:
            return _json_response({"error": "invalid json body"}, 400)

        content = body.get("content", "").strip()
        if not content:
            return _json_response({"error": "content is required"}, 400)

        session = request.get("session", {})
        actor_id = session.get("user_id")

        result = await self._services.send_dm(
            target_user_id=user_id,
            content=content,
            actor_id=actor_id,
            source_ip=_get_client_ip(request),
            user_agent=_get_user_agent(request),
        )

        status_code = 200 if result.get("success") else 400
        return _json_response(result, status=status_code)

    @auth_required
    @csrf_required
    async def send_dm_to_channel(self, request: web.Request) -> web.Response:
        """Send a DM via channel ID (for DM channels) or user ID (for arbitrary recipients).

        If the body includes a ``user_id`` field, sends to that user directly
        via ``send_dm()``. Otherwise treats the path param as a DM channel ID.
        """
        user_id_or_channel_id_str = request.match_info.get("user_id_or_channel_id", "")
        try:
            user_id_or_channel_id = int(user_id_or_channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid user_id or channel_id"}, 400)

        try:
            body = await request.json()
        except Exception:
            return _json_response({"error": "invalid json body"}, 400)

        content = body.get("content", "").strip()
        if not content:
            return _json_response({"error": "content is required"}, 400)

        # If a specific user_id is provided, send directly to that user
        user_id_from_body = body.get("user_id")
        if user_id_from_body is not None:
            try:
                target_user_id = int(user_id_from_body)
            except (ValueError, TypeError):
                return _json_response({"error": "invalid user_id in body"}, 400)

            session = request.get("session", {})
            actor_id = session.get("user_id")

            result = await self._services.send_dm(
                target_user_id=target_user_id,
                content=content,
                actor_id=actor_id,
                source_ip=_get_client_ip(request),
                user_agent=_get_user_agent(request),
            )
            status_code = 200 if result.get("success") else 400
            return _json_response(result, status=status_code)

        session = request.get("session", {})
        actor_id = session.get("user_id")

        # Use the reply_dm which handles DM channel resolution
        result = await self._services.reply_dm(
            channel_id=user_id_or_channel_id,
            content=content,
            actor_id=actor_id,
            source_ip=_get_client_ip(request),
            user_agent=_get_user_agent(request),
        )

        # Fallback: if reply_dm returns channel_not_found, the path param
        # might actually be a user ID — try send_dm directly
        if result.get("status") == "channel_not_found":
            try:
                result = await self._services.send_dm(
                    target_user_id=user_id_or_channel_id,
                    content=content,
                    actor_id=actor_id,
                    source_ip=_get_client_ip(request),
                    user_agent=_get_user_agent(request),
                )
            except Exception as exc:
                result = {"success": False, "error": str(exc), "status": "error"}

        status_code = 200 if result.get("success") else 400
        return _json_response(result, status=status_code)

    @auth_required
    @csrf_required
    async def reply_to_message(self, request: web.Request) -> web.Response:
        """Reply to a specific message. channel_id can be resolved from message store."""
        message_id_str = request.match_info.get("message_id", "")
        try:
            message_id = int(message_id_str)
        except ValueError:
            return _json_response({"error": "invalid message_id"}, 400)

        try:
            body = await request.json()
        except Exception:
            return _json_response({"error": "invalid json body"}, 400)

        content = body.get("content", "").strip()
        if not content:
            return _json_response({"error": "content is required"}, 400)

        channel_id_str = body.get("channel_id", "")

        # If channel_id not provided, try to resolve from message store
        if not channel_id_str:
            message_store: MessageStore = request.app.get("message_store")
            if message_store:
                try:
                    stored = await message_store.get_message_by_discord_id(message_id)
                    if stored and stored.get("channel_id"):
                        channel_id_str = stored["channel_id"]
                except Exception:
                    pass

        if not channel_id_str:
            return _json_response({"error": "channel_id is required in body (could not resolve from message store)"}, 400)

        try:
            channel_id = int(channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid channel_id"}, 400)

        session = request.get("session", {})
        actor_id = session.get("user_id")

        result = await self._services.reply_guild_message(
            message_id=message_id,
            channel_id=channel_id,
            content=content,
            actor_id=actor_id,
            source_ip=_get_client_ip(request),
            user_agent=_get_user_agent(request),
        )

        status_code = 200 if result.get("success") else 400
        return _json_response(result, status=status_code)

    # Legacy send routes
    @auth_required
    @csrf_required
    async def send_dm_legacy(self, request: web.Request) -> web.Response:
        user_id_str = request.match_info.get("user_id", "")
        try:
            user_id = int(user_id_str)
        except ValueError:
            return _json_response({"error": "invalid user_id"}, 400)

        try:
            body = await request.json()
        except Exception:
            return _json_response({"error": "invalid json body"}, 400)

        content = body.get("content", "").strip()
        if not content:
            return _json_response({"error": "content is required"}, 400)

        session = request.get("session", {})
        actor_id = session.get("user_id")

        result = await self._services.send_dm(
            target_user_id=user_id,
            content=content,
            actor_id=actor_id,
            source_ip=_get_client_ip(request),
            user_agent=_get_user_agent(request),
        )

        status_code = 200 if result.get("success") else 400
        return _json_response(result, status=status_code)

    @auth_required
    @csrf_required
    async def send_guild_message_legacy(self, request: web.Request) -> web.Response:
        guild_id_str = request.match_info.get("guild_id", "")
        channel_id_str = request.match_info.get("channel_id", "")

        try:
            guild_id = int(guild_id_str)
            channel_id = int(channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid guild_id or channel_id"}, 400)

        try:
            body = await request.json()
        except Exception:
            return _json_response({"error": "invalid json body"}, 400)

        content = body.get("content", "").strip()
        if not content:
            return _json_response({"error": "content is required"}, 400)

        session = request.get("session", {})
        actor_id = session.get("user_id")

        result = await self._services.send_guild_message(
            guild_id=guild_id,
            channel_id=channel_id,
            content=content,
            actor_id=actor_id,
            source_ip=_get_client_ip(request),
            user_agent=_get_user_agent(request),
        )

        status_code = 200 if result.get("success") else 400
        return _json_response(result, status=status_code)

    # ------------------------------------------------------------------
    # API: Backfill
    # ------------------------------------------------------------------

    @auth_required
    @csrf_required
    async def backfill_channel(self, request: web.Request) -> web.Response:
        """Trigger a channel backfill."""
        channel_id_str = request.match_info.get("channel_id", "")
        try:
            channel_id = int(channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid channel_id"}, 400)

        backfill_service = request.app.get("backfill_service")
        if backfill_service is None:
            return _json_response({"error": "backfill service not available"}, 500)

        config: DashboardConfig = request.app["dashboard_config"]
        if not config.backfill_enabled:
            return _json_response({"error": "backfill disabled"}, 403)

        # Audit log backfill requested
        _audit: AuditStore = request.app.get("audit_store")
        _session = request.get("session", {})
        _actor_id = _session.get("user_id")
        if _audit:
            try:
                await _audit.record(
                    event_type=EVENT_DASHBOARD_BACKFILL_REQUESTED,
                    result="pending",
                    actor_user_id=_actor_id,
                    target_channel_id=channel_id,
                )
            except Exception:
                pass

        try:
            result = await backfill_service.backfill_channel(
                channel_id=channel_id,
                limit=config.backfill_max_messages_per_channel,
            )
            status_code = 200 if result.get("status") != "failed" else 400
            # Audit log backfill result
            if _audit:
                try:
                    await _audit.record(
                        event_type=EVENT_DASHBOARD_BACKFILL_STARTED if result.get("status") in ("completed", "running", "queued") else EVENT_DASHBOARD_BACKFILL_FAILED,
                        result=result.get("status", "unknown"),
                        actor_user_id=_actor_id,
                        target_channel_id=channel_id,
                        metadata={"messages_seen": result.get("messages_seen", 0), "messages_inserted": result.get("messages_inserted", 0), "error": result.get("error")},
                    )
                except Exception:
                    pass
            return _json_response(result, status=status_code)
        except Exception as e:
            logger.warning("Backfill channel failed: %s", e)
            return _json_response({"error": str(e), "status": "error"}, 500)

    @auth_required
    @csrf_required
    async def backfill_guild(self, request: web.Request) -> web.Response:
        """Trigger a guild backfill."""
        guild_id_str = request.match_info.get("guild_id", "")
        try:
            guild_id = int(guild_id_str)
        except ValueError:
            return _json_response({"error": "invalid guild_id"}, 400)

        backfill_service = request.app.get("backfill_service")
        if backfill_service is None:
            return _json_response({"error": "backfill service not available"}, 500)

        config: DashboardConfig = request.app["dashboard_config"]
        if not config.backfill_enabled:
            return _json_response({"error": "backfill disabled"}, 403)

        # Audit log backfill requested
        _audit: AuditStore = request.app.get("audit_store")
        _session = request.get("session", {})
        _actor_id = _session.get("user_id")
        if _audit:
            try:
                await _audit.record(
                    event_type=EVENT_DASHBOARD_BACKFILL_REQUESTED,
                    result="pending",
                    actor_user_id=_actor_id,
                    target_guild_id=guild_id,
                )
            except Exception:
                pass

        try:
            result = await backfill_service.backfill_guild(
                guild_id=guild_id,
                per_channel_limit=config.backfill_max_messages_per_channel,
                max_channels=config.backfill_max_channels_per_run,
            )
            status_code = 200 if result.get("status") != "failed" else 400
            return _json_response(result, status=status_code)
        except Exception as e:
            logger.warning("Backfill guild failed: %s", e)
            return _json_response({"error": str(e), "status": "error"}, 500)

    @auth_required
    @csrf_required
    async def backfill_dm(self, request: web.Request) -> web.Response:
        """Trigger a DM backfill."""
        user_id_or_channel_id_str = request.match_info.get("user_id_or_channel_id", "")
        try:
            user_id_or_channel_id = int(user_id_or_channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid user_id or channel_id"}, 400)

        backfill_service = request.app.get("backfill_service")
        if backfill_service is None:
            return _json_response({"error": "backfill service not available"}, 500)

        config: DashboardConfig = request.app["dashboard_config"]
        if not config.backfill_enabled:
            return _json_response({"error": "backfill disabled"}, 403)

        # Audit log backfill requested
        _audit: AuditStore = request.app.get("audit_store")
        _session = request.get("session", {})
        _actor_id = _session.get("user_id")
        if _audit:
            try:
                await _audit.record(
                    event_type=EVENT_DASHBOARD_BACKFILL_REQUESTED,
                    result="pending",
                    actor_user_id=_actor_id,
                    target_channel_id=user_id_or_channel_id,
                )
            except Exception:
                pass

        try:
            result = await backfill_service.backfill_dm(
                user_id_or_channel_id=user_id_or_channel_id,
                limit=config.backfill_max_messages_per_channel,
            )
            status_code = 200 if result.get("status") != "failed" else 400
            return _json_response(result, status=status_code)
        except Exception as e:
            logger.warning("Backfill DM failed: %s", e)
            return _json_response({"error": str(e), "status": "error"}, 500)

    @auth_required
    async def list_backfill_jobs(self, request: web.Request) -> web.Response:
        """List backfill jobs."""
        backfill_store = request.app.get("backfill_store")
        if backfill_store is None:
            return _json_response({"error": "backfill store not available"}, 500)

        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)
        status_filter = _str_param(request, "status") or None
        target_type_filter = _str_param(request, "target_type") or None

        result = await backfill_store.list_jobs(
            page=page,
            page_size=page_size,
            max_page_size=MAX_PAGE_SIZE,
            status_filter=status_filter,
            target_type_filter=target_type_filter,
        )
        return _json_response(result)

    @auth_required
    @csrf_required
    async def cancel_backfill_job(self, request: web.Request) -> web.Response:
        """Cancel a backfill job."""
        job_id = request.match_info.get("job_id", "")
        if not job_id:
            return _json_response({"error": "job_id is required"}, 400)

        backfill_store = request.app.get("backfill_store")
        if backfill_store is None:
            return _json_response({"error": "backfill store not available"}, 500)

        try:
            success = await backfill_store.cancel_job(job_id)
            if success:
                return _json_response({"success": True, "job_id": job_id})
            return _json_response({"error": "Job not found or cannot be cancelled"}, 400)
        except Exception as e:
            return _json_response({"error": str(e)}, 500)

    # ------------------------------------------------------------------
    # API: Audit
    # ------------------------------------------------------------------

    @auth_required
    async def get_audit(self, request: web.Request) -> web.Response:
        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)
        event_type = _str_param(request, "event_type") or None
        actor_id_str = _str_param(request, "actor_user_id")
        guild_id_str = _str_param(request, "target_guild_id")
        user_id_str = _str_param(request, "target_user_id")
        result = _str_param(request, "result") or None
        date_from = _str_param(request, "date_from") or None
        date_to = _str_param(request, "date_to") or None

        audit_store: AuditStore = request.app.get("audit_store")
        if audit_store is None:
            return _json_response({"error": "audit store not available"}, 500)

        actor_id = int(actor_id_str) if actor_id_str else None
        guild_id = int(guild_id_str) if guild_id_str else None
        user_id = int(user_id_str) if user_id_str else None

        result_data = await audit_store.query(
            page=page,
            page_size=page_size,
            max_page_size=MAX_PAGE_SIZE,
            event_type=event_type,
            actor_user_id=actor_id,
            target_guild_id=guild_id,
            target_user_id=user_id,
            result=result,
            date_from=date_from,
            date_to=date_to,
        )
        return _json_response(result_data)

    @auth_required
    async def get_audit_event(self, request: web.Request) -> web.Response:
        """Get a single audit event by ID."""
        event_id = request.match_info.get("event_id", "")
        if not event_id:
            return _json_response({"error": "event_id required"}, 400)

        audit_store: AuditStore = request.app.get("audit_store")
        if audit_store is None:
            return _json_response({"error": "audit store not available"}, 500)

        # Fetch single event by querying with filter. AuditStore doesn't have
        # a direct get-by-id, so we query with page_size=1 and event_type filter
        # won't help. Instead, query recent events and match audit_id.
        try:
            result = await audit_store.query(page=1, page_size=MAX_PAGE_SIZE)
            for event in result.get("events", []):
                if event.get("audit_id") == event_id:
                    return _json_response(event)
            return _json_response({"error": "event not found"}, 404)
        except Exception as e:
            return _json_response({"error": str(e)}, 500)

    # ------------------------------------------------------------------
    # API: CSRF
    # ------------------------------------------------------------------

    @auth_required
    async def get_csrf_token(self, request: web.Request) -> web.Response:
        """Return CSRF token for the current session."""
        session = request.get("session")
        if session is None:
            return _json_response({"error": "not authenticated"}, 401)
        return _json_response({"csrf_token": session.get("csrf_token", "")})

    # ------------------------------------------------------------------
    # API: Static Config (no secrets)
    # ------------------------------------------------------------------

    async def get_static_config(self, request: web.Request) -> web.Response:
        """Safe UI config — no secrets, just feature flags and limits."""
        config: DashboardConfig = request.app["dashboard_config"]
        bot = self._services.bot
        bot_username = bot.user.display_name if bot and bot.user else "Bot"
        return _json_response(
            {
                "dashboard_enabled": config.enabled,
                "dm_archive_enabled": config.dm_archive_enabled,
                "guild_archive_enabled": config.guild_archive_enabled,
                "backfill_enabled": config.backfill_enabled,
                "show_message_previews": config.show_message_previews,
                "redact_secrets": config.redact_secrets,
                "max_message_chars": config.max_message_chars,
                "page_size": config.page_size,
                "max_page_size": config.max_page_size,
                "rate_limit_sends_per_minute": config.rate_limit_sends_per_minute,
                "rate_limit_backfills_per_hour": config.rate_limit_backfills_per_hour,
                "message_retention_days": config.message_retention_days,
                "dm_retention_days": config.dm_retention_days,
                "audit_retention_days": config.audit_retention_days,
                "backfill_max_messages_per_channel": config.backfill_max_messages_per_channel,
                "backfill_max_channels_per_run": config.backfill_max_channels_per_run,
                "backfill_sleep_ms": config.backfill_sleep_ms,
                "static_cache_seconds": config.static_cache_seconds,
                "require_csrf": config.require_csrf,
                "suppress_mentions": config.suppress_mentions,
                "allow_everyone_mentions": config.allow_everyone_mentions,
                "message_page_size": config.message_page_size,
                "message_page_size_max": config.message_page_size_max,
                "bot_username": bot_username,
            }
        )

    @auth_required
    async def get_permissions(self, request: web.Request) -> web.Response:
        """Get detailed permissions for a channel."""
        channel_id_str = request.match_info.get("channel_id", "")
        try:
            channel_id = int(channel_id_str)
        except ValueError:
            return _json_response({"error": "invalid channel_id"}, 400)

        from .permissions import get_channel_permissions as gcp

        result = gcp(self._services.bot, channel_id)
        return _json_response(result)


def setup_routes(app: web.Application, services: DashboardServices) -> None:
    """Register all dashboard routes."""
    routes = DashboardRoutes(services)

    # Public
    app.router.add_get("/healthz", routes.healthz)
    app.router.add_get("/api/session", routes.session_check)
    app.router.add_get("/api/static-config", routes.get_static_config)

    # Auth
    app.router.add_post("/api/login", login_handler)
    app.router.add_post("/api/logout", logout_handler)

    # Static files
    app.router.add_get("/static/{filename}", routes.static)

    # Dashboard shell
    app.router.add_get("/", routes.index)

    # API: Overview
    app.router.add_get("/api/overview", routes.get_overview)
    app.router.add_get("/api/summary", routes.get_summary)
    app.router.add_get("/api/metrics", routes.get_metrics)

    # API: Guilds
    app.router.add_get("/api/guilds", routes.get_guilds)
    app.router.add_get("/api/guilds/{guild_id}", routes.get_guild_detail)
    app.router.add_get("/api/guilds/{guild_id}/channels", routes.get_guild_channels)

    # API: Channel Messages
    app.router.add_get("/api/channels/{channel_id}/messages", routes.get_channel_messages)

    # API: DMs
    app.router.add_get("/api/dms", routes.get_dms)
    app.router.add_get("/api/dms/{user_id_or_channel_id}/messages", routes.get_dm_thread)
    app.router.add_get("/api/dms/{user_id}", routes.get_dm_thread_by_user_id)

    # API: Message Search
    app.router.add_get("/api/messages/search", routes.search_messages)

    # API: Send (POST - new routes)
    app.router.add_post("/api/channels/{channel_id}/send", routes.send_to_channel)
    app.router.add_post("/api/dms/{user_id_or_channel_id}/send", routes.send_dm_to_channel)
    app.router.add_post("/api/messages/{message_id}/reply", routes.reply_to_message)

    # API: Send (POST - legacy routes)
    app.router.add_post("/api/dms/{user_id}/send", routes.send_dm_legacy)
    app.router.add_post("/api/guilds/{guild_id}/channels/{channel_id}/send", routes.send_guild_message_legacy)

    # API: Backfill
    app.router.add_post("/api/backfill/channel/{channel_id}", routes.backfill_channel)
    app.router.add_post("/api/backfill/guild/{guild_id}", routes.backfill_guild)
    app.router.add_post("/api/backfill/dm/{user_id_or_channel_id}", routes.backfill_dm)
    app.router.add_get("/api/backfill/jobs", routes.list_backfill_jobs)
    app.router.add_post("/api/backfill/jobs/{job_id}/cancel", routes.cancel_backfill_job)

    # API: Audit
    app.router.add_get("/api/audit", routes.get_audit)
    app.router.add_get("/api/audit/{event_id}", routes.get_audit_event)

    # API: CSRF
    app.router.add_get("/api/csrf-token", routes.get_csrf_token)

    # API: Permissions
    app.router.add_get("/api/permissions/{channel_id}", routes.get_permissions)
