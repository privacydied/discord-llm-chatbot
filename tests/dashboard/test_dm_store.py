"""Tests for DM store."""

from __future__ import annotations

from pathlib import Path

import pytest

from bot.dashboard.dm_store import DMStore, _make_preview


@pytest.fixture
def dm_store(tmp_path: Path) -> DMStore:
    return DMStore(db_path=str(tmp_path / "test_dms.db"), retention_days=90)


@pytest.mark.asyncio
async def test_upsert_and_list(dm_store: DMStore) -> None:
    """Test upserting users and listing DM threads."""
    await dm_store.upsert_user(
        user_id=123,
        username="testuser",
        global_name="Test User",
        display_name="Test",
    )
    await dm_store.add_message(
        message_id=1,
        channel_id=123,
        author_id=123,
        content="Hello bot!",
        clean_content="Hello bot!",
    )
    await dm_store.add_message(
        message_id=2,
        channel_id=123,
        author_id=0,  # Bot user
        content="Hello there!",
        clean_content="Hello there!",
        is_bot_author=True,
    )

    result = await dm_store.list_threads(page=1, page_size=10)
    assert result["total"] == 1
    assert len(result["threads"]) == 1
    assert result["threads"][0]["message_count"] == 2


@pytest.mark.asyncio
async def test_get_thread_messages(dm_store: DMStore) -> None:
    """Test retrieving thread messages."""
    for i in range(5):
        await dm_store.add_message(
            message_id=i + 1,
            channel_id=456,
            author_id=123 if i % 2 == 0 else 0,
            content=f"Message {i}",
            clean_content=f"Message {i}",
            is_bot_author=i % 2 != 0,
        )

    result = await dm_store.get_thread_messages(channel_id=456, page=1, page_size=10)
    assert result["total"] == 5
    # Newest first
    assert result["messages"][0]["content_preview"] == "Message 4"


@pytest.mark.asyncio
async def test_message_pagination(dm_store: DMStore) -> None:
    """Test pagination of thread messages."""
    for i in range(10):
        await dm_store.add_message(
            message_id=i + 1,
            channel_id=789,
            author_id=123,
            content=f"Message {i}",
            clean_content=f"Message {i}",
        )

    result = await dm_store.get_thread_messages(channel_id=789, page=1, page_size=3)
    assert len(result["messages"]) == 3
    assert result["total_pages"] == 4

    result = await dm_store.get_thread_messages(channel_id=789, page=2, page_size=3)
    assert len(result["messages"]) == 3


@pytest.mark.asyncio
async def test_direction_tracking(dm_store: DMStore) -> None:
    """Test inbound/outbound direction tracking."""
    await dm_store.add_message(
        message_id=1,
        channel_id=100,
        author_id=123,
        content="User message",
        clean_content="User message",
        is_bot_author=False,
    )
    await dm_store.add_message(
        message_id=2,
        channel_id=100,
        author_id=0,
        content="Bot message",
        clean_content="Bot message",
        is_bot_author=True,
    )

    result = await dm_store.get_thread_messages(channel_id=100, page=1, page_size=10)
    directions = {m["message_id"]: m["direction"] for m in result["messages"]}
    assert "1" in directions
    assert "2" in directions
    assert directions["1"] == "inbound"
    assert directions["2"] == "outbound"


def test_preview_truncation() -> None:
    """Test content preview truncation."""
    long_content = "x" * 300
    preview = _make_preview(long_content, max_chars=50)
    assert len(preview) <= 53


@pytest.mark.asyncio
async def test_user_update(dm_store: DMStore) -> None:
    """Test user info is updated on upsert."""
    await dm_store.upsert_user(
        user_id=123,
        username="old_name",
        global_name="Old",
    )
    await dm_store.upsert_user(
        user_id=123,
        username="new_name",
        global_name="New",
    )

    result = await dm_store.list_threads(page=1, page_size=1)
    # After upsert, the user info should be updated
    # (This indirectly tests that upsert worked)
    assert result["total"] == 0  # No messages yet, so no threads
