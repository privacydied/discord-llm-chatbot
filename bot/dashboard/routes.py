"""API routes for the dashboard server."""

from __future__ import annotations

from pathlib import Path

from aiohttp import web

from bot.utils.logging import get_logger

from .audit_store import AuditStore
from .auth import _get_client_ip, _get_user_agent, auth_required, csrf_required, login_handler, logout_handler
from .config import DashboardConfig
from .dm_store import DMStore
from .services import DashboardServices

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


def _int_param(request: web.Request, name: str, default: int = 1) -> int:
    try:
        return int(request.query.get(name, default))
    except (ValueError, TypeError):
        return default


def _str_param(request: web.Request, name: str, default: str = "") -> str:
    return request.query.get(name, default) or default


class DashboardRoutes:
    """Request handlers for the dashboard."""

    def __init__(self, services: DashboardServices) -> None:
        self._services = services

    async def healthz(self, request: web.Request) -> web.Response:
        """Minimal health check — no auth required."""
        config: DashboardConfig = request.app["dashboard_config"]
        bot = self._services.bot
        running = bot is not None and bot.is_ready()
        return web.json_response(
            {
                "status": "ok" if running else "starting",
                "enabled": config.enabled,
            }
        )

    async def index(self, request: web.Request) -> web.StreamResponse:
        """Serve the dashboard HTML shell."""
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return web.Response(text="Dashboard HTML not found", status=500)
        return web.FileResponse(index_path)

    async def static(self, request: web.Request) -> web.StreamResponse:
        """Serve static files."""
        filename = request.match_info.get("filename", "")
        filepath = STATIC_DIR / filename
        # Prevent directory traversal
        try:
            filepath.resolve().relative_to(STATIC_DIR.resolve())
        except ValueError:
            return web.Response(text="Not found", status=404)
        if not filepath.exists():
            return web.Response(text="Not found", status=404)
        return web.FileResponse(filepath)

    @auth_required
    async def get_summary(self, request: web.Request) -> web.Response:
        summary = await self._services.get_summary()
        return web.json_response(summary)

    @auth_required
    async def get_metrics(self, request: web.Request) -> web.Response:
        """Return basic metrics summary."""
        summary = await self._services.get_summary()
        return web.json_response(
            {
                "uptime_seconds": summary.get("uptime_seconds", 0),
                "guild_count": summary.get("guild_count", 0),
                "latency_ms": summary.get("latency_ms", 0),
                "cog_count": summary.get("cog_count", 0),
            }
        )

    @auth_required
    async def get_guilds(self, request: web.Request) -> web.Response:
        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)
        search = _str_param(request, "search")

        result = await self._services.get_guilds(page=page, page_size=page_size, max_page_size=MAX_PAGE_SIZE, search=search or None)
        return web.json_response(result)

    @auth_required
    async def get_guild_detail(self, request: web.Request) -> web.Response:
        guild_id_str = request.match_info.get("guild_id", "")
        try:
            guild_id = int(guild_id_str)
        except ValueError:
            return web.json_response({"error": "invalid guild_id"}, status=400)

        bot = self._services.bot
        if bot is None:
            return web.json_response({"error": "bot not ready"}, status=503)

        guild = bot.get_guild(guild_id)
        if guild is None:
            return web.json_response({"error": "guild not found"}, status=404)

        channels = []
        try:
            for ch in guild.text_channels:
                try:
                    perms = ch.permissions_for(guild.me)
                    channels.append(
                        {
                            "id": str(ch.id),
                            "name": ch.name,
                            "type": "text",
                            "can_send": perms.send_messages,
                            "can_read": perms.read_messages,
                        }
                    )
                except Exception:
                    pass
        except Exception:
            pass

        return web.json_response(
            {
                "id": str(guild.id),
                "name": guild.name,
                "owner_id": str(guild.owner_id) if guild.owner_id else None,
                "member_count": guild.member_count,
                "channels": channels[:50],  # Limit channels returned
            }
        )

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
            return web.json_response({"error": "audit store not available"}, status=500)

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
        return web.json_response(result_data)

    @auth_required
    async def get_dms(self, request: web.Request) -> web.Response:
        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)

        dm_store: DMStore = request.app.get("dm_store")
        if dm_store is None:
            return web.json_response({"error": "DM store not available"}, status=500)

        config: DashboardConfig = request.app["dashboard_config"]
        if not config.dm_archive_enabled:
            return web.json_response({"error": "DM archive disabled"}, status=403)

        result = await dm_store.list_threads(page=page, page_size=page_size, max_page_size=MAX_PAGE_SIZE)
        return web.json_response(result)

    @auth_required
    async def get_dm_thread(self, request: web.Request) -> web.Response:
        user_id_str = request.match_info.get("user_id", "")
        try:
            user_id = int(user_id_str)
        except ValueError:
            return web.json_response({"error": "invalid user_id"}, status=400)

        dm_store: DMStore = request.app.get("dm_store")
        if dm_store is None:
            return web.json_response({"error": "DM store not available"}, status=500)

        page = _int_param(request, "page", 1)
        page_size = _int_param(request, "page_size", DEFAULT_PAGE_SIZE)

        result = await dm_store.get_thread_messages(
            channel_id=user_id,  # channel_id maps to user_id in DM store
            page=page,
            page_size=page_size,
            max_page_size=MAX_PAGE_SIZE,
        )
        return web.json_response(result)

    @auth_required
    @csrf_required
    async def send_dm(self, request: web.Request) -> web.Response:
        user_id_str = request.match_info.get("user_id", "")
        try:
            user_id = int(user_id_str)
        except ValueError:
            return web.json_response({"error": "invalid user_id"}, status=400)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json body"}, status=400)

        content = body.get("content", "").strip()
        if not content:
            return web.json_response({"error": "content is required"}, status=400)

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
        return web.json_response(result, status=status_code)

    @auth_required
    @csrf_required
    async def send_guild_message(self, request: web.Request) -> web.Response:
        guild_id_str = request.match_info.get("guild_id", "")
        channel_id_str = request.match_info.get("channel_id", "")

        try:
            guild_id = int(guild_id_str)
            channel_id = int(channel_id_str)
        except ValueError:
            return web.json_response({"error": "invalid guild_id or channel_id"}, status=400)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json body"}, status=400)

        content = body.get("content", "").strip()
        if not content:
            return web.json_response({"error": "content is required"}, status=400)

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
        return web.json_response(result, status=status_code)

    @auth_required
    async def get_csrf_token(self, request: web.Request) -> web.Response:
        """Return CSRF token for the current session."""
        session = request.get("session")
        if session is None:
            return web.json_response({"error": "not authenticated"}, status=401)
        return web.json_response({"csrf_token": session.get("csrf_token", "")})


def setup_routes(app: web.Application, services: DashboardServices) -> None:
    """Register all dashboard routes."""
    routes = DashboardRoutes(services)

    # Public
    app.router.add_get("/healthz", routes.healthz)

    # Auth
    app.router.add_post("/api/login", login_handler)
    app.router.add_post("/api/logout", logout_handler)

    # Static files
    app.router.add_get("/static/{filename}", routes.static)

    # Dashboard shell
    app.router.add_get("/", routes.index)

    # API endpoints
    app.router.add_get("/api/summary", routes.get_summary)
    app.router.add_get("/api/metrics", routes.get_metrics)
    app.router.add_get("/api/guilds", routes.get_guilds)
    app.router.add_get("/api/guilds/{guild_id}", routes.get_guild_detail)
    app.router.add_get("/api/audit", routes.get_audit)
    app.router.add_get("/api/dms", routes.get_dms)
    app.router.add_get("/api/dms/{user_id}", routes.get_dm_thread)
    app.router.add_post("/api/dms/{user_id}/send", routes.send_dm)
    app.router.add_post("/api/guilds/{guild_id}/channels/{channel_id}/send", routes.send_guild_message)
    app.router.add_get("/api/csrf-token", routes.get_csrf_token)
