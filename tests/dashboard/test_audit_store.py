"""Tests for dashboard audit store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bot.dashboard.audit_store import AuditStore, _make_preview, _truncate_ip

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def audit_store(tmp_path: Path) -> AuditStore:
    db_path = str(tmp_path / "test_audit.db")
    return AuditStore(db_path=db_path, retention_days=180)


@pytest.mark.asyncio
async def test_record_and_query(audit_store: AuditStore) -> None:
    """Test recording and querying audit events."""
    await audit_store.record(
        event_type="dashboard.login.success",
        result="success",
        actor_user_id=123,
        actor_source_ip="192.168.1.1",
    )

    result = await audit_store.query(page=1, page_size=10)
    assert result["total"] == 1
    assert len(result["events"]) == 1
    assert result["events"][0]["event_type"] == "dashboard.login.success"
    assert result["events"][0]["result"] == "success"


@pytest.mark.asyncio
async def test_query_filter_by_event_type(audit_store: AuditStore) -> None:
    """Test filtering by event type."""
    await audit_store.record(event_type="dashboard.login.success", result="success")
    await audit_store.record(event_type="dashboard.send.dm", result="success")

    result = await audit_store.query(page=1, page_size=10, event_type="dashboard.send.dm")
    assert result["total"] == 1
    assert result["events"][0]["event_type"] == "dashboard.send.dm"


@pytest.mark.asyncio
async def test_query_filter_by_result(audit_store: AuditStore) -> None:
    """Test filtering by result."""
    await audit_store.record(event_type="dashboard.send.dm", result="success")
    await audit_store.record(event_type="dashboard.send.dm", result="failed", error_code="forbidden")

    result = await audit_store.query(page=1, page_size=10, result="failed")
    assert result["total"] == 1
    assert result["events"][0]["result"] == "failed"


@pytest.mark.asyncio
async def test_pagination(audit_store: AuditStore) -> None:
    """Test pagination works correctly."""
    for _i in range(5):
        await audit_store.record(event_type="dashboard.command.invoke", result="success")

    result = await audit_store.query(page=1, page_size=2)
    assert result["total"] == 5
    assert len(result["events"]) == 2
    assert result["total_pages"] == 3
    assert result["page"] == 1

    result = await audit_store.query(page=2, page_size=2)
    assert len(result["events"]) == 2
    assert result["page"] == 2


@pytest.mark.asyncio
async def test_ip_truncation(audit_store: AuditStore) -> None:
    """Test IP addresses are hashed."""
    await audit_store.record(
        event_type="dashboard.login.success",
        result="success",
        actor_source_ip="192.168.1.100",
    )

    result = await audit_store.query(page=1, page_size=1)
    ip_hash = result["events"][0]["actor_source_ip"]
    assert ip_hash != "192.168.1.100"  # Should not be raw IP
    assert len(ip_hash) == 12  # Truncated hash


def test_make_preview_truncates() -> None:
    """Test preview truncation."""
    long_content = "x" * 500
    preview = _make_preview(long_content, max_chars=50)
    assert len(preview) <= 53  # 50 + "..."


def test_make_preview_short() -> None:
    """Test short content is not truncated."""
    preview = _make_preview("hello", max_chars=50)
    assert preview == "hello"


def test_truncate_ip() -> None:
    """Test IP truncation."""
    hashed = _truncate_ip("192.168.1.1")
    assert len(hashed) == 12
    assert hashed != "192.168.1.1"


def test_truncate_ip_empty() -> None:
    """Test empty IP."""
    assert _truncate_ip("") == ""


@pytest.mark.asyncio
async def test_content_preview_stored(audit_store: AuditStore) -> None:
    """Test content preview is stored correctly."""
    content = "Hello world this is a test message"
    await audit_store.record(
        event_type="dashboard.send.dm",
        result="success",
        content_preview=content,
    )

    result = await audit_store.query(page=1, page_size=1)
    assert result["events"][0]["content_preview"] == content


@pytest.mark.asyncio
async def test_metadata_stored(audit_store: AuditStore) -> None:
    """Test metadata is stored and retrieved."""
    meta = {"key1": "value1", "key2": 42}
    await audit_store.record(
        event_type="dashboard.send.dm",
        result="success",
        metadata=meta,
    )

    result = await audit_store.query(page=1, page_size=1)
    assert result["events"][0]["metadata"]["key1"] == "value1"
    assert result["events"][0]["metadata"]["key2"] == 42


@pytest.mark.asyncio
async def test_retention_cleanup(audit_store: AuditStore) -> None:
    """Test retention cleanup removes old records."""
    await audit_store.record(event_type="dashboard.start", result="success")

    # With 180-day retention, the record should still exist
    deleted = await audit_store.cleanup_retention()
    # New records won't be deleted
    assert deleted == 0
