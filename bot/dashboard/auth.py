"""Authentication middleware for the dashboard aiohttp server.

Supports:
- Bearer token auth via Authorization header
- Session-based auth via secure cookie
- CSRF protection for POST/PUT/DELETE endpoints
- IP hashing for audit trails
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from bot.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from .audit_store import AuditStore
    from .config import DashboardConfig

logger = get_logger(__name__)

# Session cookie settings
COOKIE_NAME = "dash_session"
COOKIE_MAX_AGE = 28800  # 8 hours
CSRF_HEADER = "X-CSRF-Token"
CSRF_FIELD = "csrf_token"


class SessionStore:
    """In-memory session store with TTL."""

    def __init__(self, session_ttl_hours: int = 8) -> None:
        self._ttl = session_ttl_hours * 3600
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, user_id: int, csrf_token: str) -> str:
        session_id = secrets.token_hex(32)
        self._sessions[session_id] = {
            "user_id": user_id,
            "csrf_token": csrf_token,
            "created_at": time.time(),
        }
        return session_id

    def get(self, session_id: str) -> dict[str, Any] | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session["created_at"] > self._ttl:
            del self._sessions[session_id]
            return None
        return session

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def cleanup(self) -> int:
        """Remove expired sessions. Returns count removed."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s["created_at"] > self._ttl]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


def _hash_ip(ip: str) -> str:
    """Hash IP for privacy in audit logs."""
    if not ip:
        return "unknown"
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


def _get_client_ip(request: web.Request) -> str:
    """Get client IP, respecting proxy headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    peername = request.remote
    return peername or "unknown"


def _get_user_agent(request: web.Request) -> str:
    ua = request.headers.get("User-Agent", "")
    return ua[:512]


def _check_bearer_auth(request: web.Request, auth_token: str) -> bool:
    """Check Bearer token in Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        return token == auth_token
    return False


def _check_session_auth(request: web.Request, session_store: SessionStore) -> dict[str, Any] | None:
    """Check session cookie. Returns session data or None."""
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie:
        return None
    return session_store.get(cookie)


def auth_required(handler: Callable) -> Callable:
    """Decorator: require authentication for a handler."""

    async def wrapper(self, request: web.Request, *args, **kwargs):
        config: DashboardConfig = request.app["dashboard_config"]
        session_store: SessionStore = request.app["session_store"]

        # Check bearer token
        if config.auth_token and _check_bearer_auth(request, config.auth_token):
            return await handler(self, request, *args, **kwargs)

        # Check session
        session = _check_session_auth(request, session_store)
        if session:
            request["session"] = session
            return await handler(self, request, *args, **kwargs)

        # Log failure
        audit_store: AuditStore = request.app.get("audit_store")
        if audit_store:
            await audit_store.record(
                event_type="dashboard.login.failure",
                result="unauthorized",
                actor_source_ip=_get_client_ip(request),
                actor_user_agent=_get_user_agent(request),
            )

        return web.json_response(
            {"error": "unauthorized"},
            status=401,
            headers={"WWW-Authenticate": 'Bearer realm="dashboard"'},
        )

    return wrapper


def csrf_required(handler: Callable) -> Callable:
    """Decorator: require CSRF token for POST/PUT/DELETE handlers."""

    async def wrapper(self, request: web.Request, *args, **kwargs):
        session = request.get("session")
        if session is None:
            # Not authenticated — let auth_required handle it
            return await handler(self, request, *args, **kwargs)

        # Check CSRF token from header or form
        csrf_token = request.headers.get(CSRF_HEADER, "")
        if not csrf_token:
            # Try form data
            try:
                form = await request.post()
                csrf_token = form.get(CSRF_FIELD, "")
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug(f"Failed to parse form data for CSRF token: {e}")

        if csrf_token != session.get("csrf_token"):
            return web.json_response({"error": "csrf_invalid"}, status=403)

        return await handler(self, request, *args, **kwargs)

    return wrapper


async def login_handler(request: web.Request) -> web.Response:
    """Handle login POST request."""
    config: DashboardConfig = request.app["dashboard_config"]
    session_store: SessionStore = request.app["session_store"]
    audit_store: AuditStore = request.app.get("audit_store")

    # Check body for auth_token
    try:
        body = await request.json()
    except (RuntimeError, ValueError, TypeError, ImportError) as e:
        logger.debug(f"Failed to parse JSON body: {e}")
        body = {}

    provided_token = body.get("auth_token", "") or request.headers.get("Authorization", "").replace("Bearer ", "").strip()

    if not provided_token or provided_token != config.auth_token:
        if audit_store:
            await audit_store.record(
                event_type="dashboard.login.failure",
                result="invalid_token",
                actor_source_ip=_get_client_ip(request),
                actor_user_agent=_get_user_agent(request),
            )
        return web.json_response({"error": "invalid_token"}, status=401)

    # Create session
    csrf_token = secrets.token_hex(32)
    # Use first owner_id as session user, or 0
    user_id = next(iter(config.owner_ids), 0)
    session_id = session_store.create(user_id, csrf_token)

    if audit_store:
        await audit_store.record(
            event_type="dashboard.login.success",
            result="success",
            actor_user_id=user_id,
            actor_source_ip=_get_client_ip(request),
            actor_user_agent=_get_user_agent(request),
        )

    response = web.json_response(
        {
            "success": True,
            "csrf_token": csrf_token,
        },
    )
    response.set_cookie(
        COOKIE_NAME,
        session_id,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="Lax",
        secure=False,  # Set to True if behind HTTPS reverse proxy
    )
    return response


async def logout_handler(request: web.Request) -> web.Response:
    """Handle logout POST request."""
    session_store: SessionStore = request.app["session_store"]
    audit_store: AuditStore = request.app.get("audit_store")

    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        session = session_store.get(cookie)
        if audit_store and session:
            await audit_store.record(
                event_type="dashboard.logout",
                result="success",
                actor_user_id=session.get("user_id"),
                actor_source_ip=_get_client_ip(request),
            )
        session_store.remove(cookie)

    response = web.json_response({"success": True})
    response.del_cookie(COOKIE_NAME)
    return response
