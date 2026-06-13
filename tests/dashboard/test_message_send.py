"""Tests for dashboard message sending functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bot.dashboard.audit_store import AuditStore
from bot.dashboard.config import DashboardConfig
from bot.dashboard.dm_store import DMStore
from bot.dashboard.services import DashboardServices

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config() -> DashboardConfig:
    return DashboardConfig(
        enabled=True,
        auth_token="test-token",
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
        bot=None,
        config=config,
        audit_store=audit_store,
        dm_store=dm_store,
    )


@pytest.mark.asyncio
async def test_send_dm_no_bot(services: DashboardServices) -> None:
    """Send DM fails gracefully when bot is not connected."""
    result = await services.send_dm(
        target_user_id=123,
        content="hello",
        actor_id=12345,
    )
    assert result["success"] is False
    assert result["status"] == "not_ready"


@pytest.mark.asyncio
async def test_send_dm_too_long(config: DashboardConfig, audit_store: AuditStore, dm_store: DMStore) -> None:
    """Send DM with content exceeding max chars fails."""
    services = DashboardServices(
        bot=None,
        config=config,
        audit_store=audit_store,
        dm_store=dm_store,
    )
    result = await services.send_dm(
        target_user_id=123,
        content="x" * 200,
        actor_id=12345,
    )
    assert result["success"] is False
    assert result["status"] == "too_long"


@pytest.mark.asyncio
async def test_send_guild_message_no_bot(services: DashboardServices) -> None:
    """Send guild message fails gracefully when bot is not connected."""
    result = await services.send_guild_message(
        guild_id=1,
        channel_id=2,
        content="hello",
        actor_id=12345,
    )
    assert result["success"] is False
    assert result["status"] == "not_ready"


@pytest.mark.asyncio
async def test_send_guild_message_too_long(config: DashboardConfig, audit_store: AuditStore, dm_store: DMStore) -> None:
    """Send guild message with content exceeding max chars fails."""
    services = DashboardServices(
        bot=None,
        config=config,
        audit_store=audit_store,
        dm_store=dm_store,
    )
    result = await services.send_guild_message(
        guild_id=1,
        channel_id=2,
        content="x" * 200,
        actor_id=12345,
    )
    assert result["success"] is False
    assert result["status"] == "too_long"


@pytest.mark.asyncio
async def test_send_dm_records_audit(services: DashboardServices) -> None:
    """Send DM records an audit event even on failure."""
    await services.send_dm(
        target_user_id=123,
        content="hello",
        actor_id=12345,
    )

    result = await services._audit_store.query(page=1, page_size=10)
    # Should have at least one audit record
    assert result["total"] >= 1
    # Check that the event type is recorded
    event_types = [e["event_type"] for e in result["events"]]
    assert "dashboard.message.send.requested" in event_types


@pytest.mark.asyncio
async def test_send_guild_message_records_audit(services: DashboardServices) -> None:
    """Send guild message records an audit event even on failure."""
    await services.send_guild_message(
        guild_id=1,
        channel_id=2,
        content="hello",
        actor_id=12345,
    )

    result = await services._audit_store.query(page=1, page_size=10)
    event_types = [e["event_type"] for e in result["events"]]
    assert "dashboard.message.send.requested" in event_types
