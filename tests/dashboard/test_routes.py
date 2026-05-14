"""Tests for dashboard routes and server."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from bot.dashboard.audit_store import AuditStore
from bot.dashboard.config import DashboardConfig
from bot.dashboard.dm_store import DMStore
from bot.dashboard.services import DashboardServices
from bot.dashboard.server import DashboardServer
from bot.dashboard.routes import setup_routes


@pytest.fixture
def config() -> DashboardConfig:
    return DashboardConfig(
        enabled=True,
        auth_token="test-token-123",
        owner_ids={12345},
        rate_limit_sends_per_minute=5,
        max_message_chars=100,
        dm_archive_enabled=True,
    )


@pytest.fixture
def audit_store(tmp_path: Path) -> AuditStore:
    return AuditStore(db_path=str(tmp_path / "test_audit.db"), retention_days=180)


@pytest.fixture
def dm_store(tmp_path: Path) -> DMStore:
    return DMStore(db_path=str(tmp_path / "test_dms.db"), retention_days=90)


@pytest.fixture
def services(config: DashboardConfig, audit_store: AuditStore, dm_store: DMStore) -> DashboardServices:
    return DashboardServices(
        bot=None,  # No real bot in tests
        config=config,
        audit_store=audit_store,
        dm_store=dm_store,
    )


@pytest.fixture
async def app(services: DashboardServices, config: DashboardConfig, audit_store: AuditStore, dm_store: DMStore) -> web.Application:
    from bot.dashboard.auth import SessionStore

    app = web.Application()
    app["dashboard_config"] = config
    app["session_store"] = SessionStore()
    app["audit_store"] = audit_store
    app["dm_store"] = dm_store
    setup_routes(app, services)
    return app


@pytest.fixture
async def client(app: web.Application):
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest.mark.asyncio
async def test_healthz_no_auth(client: TestClient) -> None:
    """Healthz endpoint requires no auth."""
    resp = await client.get("/healthz")
    assert resp.status == 200
    data = await resp.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_index_no_auth_returns_html(client: TestClient) -> None:
    """Index page serves HTML without auth."""
    resp = await client.get("/")
    assert resp.status == 200
    text = await resp.text()
    assert "Dashboard Login" in text


@pytest.mark.asyncio
async def test_summary_requires_auth(client: TestClient) -> None:
    """Summary endpoint requires authentication."""
    resp = await client.get("/api/summary")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_guilds_requires_auth(client: TestClient) -> None:
    """Guilds endpoint requires authentication."""
    resp = await client.get("/api/guilds")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_audit_requires_auth(client: TestClient) -> None:
    """Audit endpoint requires authentication."""
    resp = await client.get("/api/audit")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_login_with_valid_token(client: TestClient, config: DashboardConfig) -> None:
    """Login with valid auth token succeeds."""
    resp = await client.post("/api/login", json={"auth_token": config.auth_token})
    assert resp.status == 200
    data = await resp.json()
    assert data["success"] is True
    assert "csrf_token" in data


@pytest.mark.asyncio
async def test_login_with_invalid_token(client: TestClient) -> None:
    """Login with invalid token fails."""
    resp = await client.post("/api/login", json={"auth_token": "wrong-token"})
    assert resp.status == 401


@pytest.mark.asyncio
async def test_bearer_auth_summary(client: TestClient, config: DashboardConfig) -> None:
    """Bearer token auth works for API endpoints."""
    resp = await client.get(
        "/api/summary",
        headers={"Authorization": f"Bearer {config.auth_token}"},
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_send_dm_requires_csrf(client: TestClient, config: DashboardConfig) -> None:
    """Send DM requires CSRF token."""
    # Login first
    login_resp = await client.post("/api/login", json={"auth_token": config.auth_token})
    assert login_resp.status == 200

    # Try to send DM without CSRF
    resp = await client.post("/api/dms/12345/send", json={"content": "hello"})
    assert resp.status == 403  # CSRF failure


@pytest.mark.asyncio
async def test_send_dm_empty_content(client: TestClient, config: DashboardConfig) -> None:
    """Send DM with empty content fails."""
    resp = await client.post(
        "/api/dms/12345/send",
        json={"content": ""},
        headers={"Authorization": f"Bearer {config.auth_token}"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_send_dm_invalid_user_id(client: TestClient, config: DashboardConfig) -> None:
    """Send DM with invalid user ID fails."""
    resp = await client.post(
        "/api/dms/not-a-number/send",
        json={"content": "hello"},
        headers={"Authorization": f"Bearer {config.auth_token}"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_logout(client: TestClient, config: DashboardConfig) -> None:
    """Logout clears session."""
    # Login
    login_resp = await client.post("/api/login", json={"auth_token": config.auth_token})
    assert login_resp.status == 200

    # Logout
    logout_resp = await client.post("/api/logout")
    assert logout_resp.status == 200


@pytest.mark.asyncio
async def test_dashboard_disabled_no_server(config: DashboardConfig) -> None:
    """When dashboard is disabled, server should not start."""
    disabled_config = DashboardConfig(enabled=False)
    audit_store = AuditStore(db_path="/tmp/test_disabled_audit.db", retention_days=180)
    dm_store = DMStore(db_path="/tmp/test_disabled_dms.db", retention_days=90)
    services = DashboardServices(bot=None, config=disabled_config, audit_store=audit_store, dm_store=dm_store)
    server = DashboardServer(config=disabled_config, services=services, audit_store=audit_store, dm_store=dm_store)

    # Should not raise, just log and return
    await server.start()
    assert not server.is_running
    await server.stop()  # Should be safe to call
