"""Comprehensive test suite for the dashboard stores, config, and permissions.

Covers:
- MessageStore (in-memory SQLite)
- AuditStore (in-memory SQLite)
- DMStore (in-memory SQLite)
- DashboardConfig / load_dashboard_config
- Permission functions (mock Discord objects)
- BackfillJobStore (in-memory SQLite)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from bot.dashboard.audit_store import EVENT_DASHBOARD_LOGIN_SUCCESS, AuditStore
from bot.dashboard.backfill import (
    BACKFILL_STATUS_CANCELLED,
    BACKFILL_STATUS_COMPLETED,
    BACKFILL_STATUS_QUEUED,
    BACKFILL_STATUS_RUNNING,
    BackfillJobStore,
)
from bot.dashboard.config import (
    DashboardConfig,
    load_dashboard_config,
)
from bot.dashboard.dm_store import DMStore
from bot.dashboard.message_store import MessageStore
from bot.dashboard.permissions import (
    can_read_message_history,
    can_send_messages,
    can_view_channel,
    get_channel_permissions,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def msg_store(tmp_path):
    """MessageStore backed by a temp file."""
    db = tmp_path / "test_messages.db"
    return MessageStore(str(db), retention_days=90)


@pytest.fixture
def audit_store(tmp_path):
    """AuditStore backed by a temp file."""
    db = tmp_path / "test_audit.db"
    return AuditStore(str(db), retention_days=180)


@pytest.fixture
def dm_store(tmp_path):
    """DMStore backed by a temp file."""
    db = tmp_path / "test_dm.db"
    return DMStore(str(db), retention_days=90)


@pytest.fixture
def bj_store(tmp_path):
    """BackfillJobStore backed by a temp file."""
    db = tmp_path / "test_backfill.db"
    return BackfillJobStore(str(db))


# ===================================================================
# MessageStore Tests
# ===================================================================


class TestMessageStoreInsertAndRetrieve:
    """Insert a message and retrieve it by Discord ID."""

    async def test_insert_and_retrieve(self, msg_store) -> None:
        inserted = await msg_store.insert_message(
            discord_message_id=1001,
            channel_id=2001,
            content="Hello world",
            guild_id=3001,
            channel_name="general",
            author_id=4001,
            author_username="testuser",
            author_display_name="TestUser",
            direction="inbound",
        )
        assert inserted is True

        msg = await msg_store.get_message_by_discord_id(1001)
        assert msg is not None
        assert msg["discord_message_id"] == "1001"
        assert msg["channel_id"] == "2001"
        assert msg["content"] == "Hello world"
        assert msg["author_username"] == "testuser"
        assert msg["direction"] == "inbound"


class TestMessageStoreDuplicateInsert:
    """Inserting the same discord_message_id twice succeeds (no constraint on ID column)."""

    async def test_duplicate_not_ignored(self, msg_store) -> None:
        """The messages table has no UNIQUE constraint on discord_message_id,
        so duplicate IDs are accepted. This is the current design.
        """
        inserted1 = await msg_store.insert_message(
            discord_message_id=1002,
            channel_id=2002,
            content="First",
            author_id=4002,
        )
        assert inserted1 is True

        inserted2 = await msg_store.insert_message(
            discord_message_id=1002,  # same ID — no constraint, so insert succeeds
            channel_id=2002,
            content="Second (different row_id)",
            author_id=4002,
        )
        assert inserted2 is True  # Both inserts succeed

        # Both rows exist — get_message_by_discord_id returns the first one
        msg = await msg_store.get_message_by_discord_id(1002)
        assert msg is not None
        # Since LIMIT 1 is used, it returns whichever row is found first


class TestMessageStoreGetChannelMessages:
    """Get channel messages with pagination."""

    async def test_get_channel_messages_paginated(self, msg_store) -> None:
        # Insert 5 messages in channel 2010
        for i in range(5):
            await msg_store.insert_message(
                discord_message_id=1010 + i,
                channel_id=2010,
                content=f"Message {i}",
                author_id=4010,
                created_at=f"2024-01-0{i+1}T00:00:00.000Z",
            )

        # Page 1 with page_size=2
        result = await msg_store.get_channel_messages(2010, page=1, page_size=2)
        assert result["total"] == 5
        assert result["total_pages"] == 3
        assert len(result["messages"]) == 2

        # Page 2
        result2 = await msg_store.get_channel_messages(2010, page=2, page_size=2)
        assert len(result2["messages"]) == 2

        # Page 3
        result3 = await msg_store.get_channel_messages(2010, page=3, page_size=2)
        assert len(result3["messages"]) == 1


class TestMessageStoreGetDMThreadMessages:
    """Get DM thread messages via MessageStore."""

    async def test_get_dm_thread_messages(self, msg_store) -> None:
        # Insert messages in a DM channel (no guild_id)
        for i in range(3):
            await msg_store.insert_message(
                discord_message_id=1100 + i,
                channel_id=999001,  # DM channel
                content=f"DM msg {i}",
                author_id=5001,
                created_at=f"2024-02-0{i+1}T00:00:00.000Z",
            )

        result = await msg_store.get_dm_thread_messages(999001, page=1, page_size=10)
        assert result["total"] == 3
        assert len(result["messages"]) == 3
        assert result["dm_channel_id"] == "999001"


class TestMessageStoreSearchMessages:
    """Search messages by content."""

    async def test_search_by_content(self, msg_store) -> None:
        await msg_store.insert_message(
            discord_message_id=1201,
            channel_id=2050,
            content="The quick brown fox",
            author_id=4050,
        )
        await msg_store.insert_message(
            discord_message_id=1202,
            channel_id=2050,
            content="Jumped over the lazy dog",
            author_id=4050,
        )
        await msg_store.insert_message(
            discord_message_id=1203,
            channel_id=2050,
            content="Fox in the box",
            author_id=4051,
        )

        result = await msg_store.search_messages("fox")
        assert result["total"] == 2  # matches first and third
        assert result["query"] == "fox"

        # Filter by author
        result2 = await msg_store.search_messages("fox", author_id=4051)
        assert result2["total"] == 1

        # Filter by channel
        result3 = await msg_store.search_messages("fox", channel_id=2050)
        assert result3["total"] == 2

        # No matches
        result4 = await msg_store.search_messages("xyzzy")
        assert result4["total"] == 0


class TestMessageStoreUpsertDMThread:
    """Upsert DM thread metadata."""

    async def test_upsert_dm_thread(self, msg_store) -> None:
        await msg_store.upsert_dm_thread(
            dm_channel_id=999002,
            user_id=5002,
            username="dmuser",
            display_name="DM User",
            last_message_id=1301,
        )

        threads = await msg_store.get_dm_threads()
        assert threads["total"] == 1
        t = threads["threads"][0]
        assert t["dm_channel_id"] == "999002"
        assert t["username"] == "dmuser"
        assert t["message_count"] == 1

        # Upsert again (increment message_count)
        await msg_store.upsert_dm_thread(
            dm_channel_id=999002,
            user_id=5002,
            username="dmuser",
            increment_count=True,
        )
        threads2 = await msg_store.get_dm_threads()
        assert threads2["threads"][0]["message_count"] == 2


class TestMessageStoreCleanupRetention:
    """Soft-delete old messages."""

    async def test_cleanup_retention(self, msg_store) -> None:
        # Insert one recent and one old message
        old_ts = (datetime.now(UTC) - timedelta(days=200)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        await msg_store.insert_message(
            discord_message_id=1401,
            channel_id=2060,
            content="Old message",
            author_id=4060,
            created_at=old_ts,
        )
        await msg_store.insert_message(
            discord_message_id=1402,
            channel_id=2060,
            content="Recent message",
            author_id=4060,
        )

        # retention_days=90 should delete the old one
        cleaned = await msg_store.cleanup_retention()
        assert cleaned == 1

        # Old message still exists but soft-deleted (deleted_at set)
        msg = await msg_store.get_message_by_discord_id(1401)
        assert msg is not None
        assert msg["deleted_at"] is not None

        # Recent message should still be active
        msg2 = await msg_store.get_message_by_discord_id(1402)
        assert msg2 is not None
        assert msg2["deleted_at"] is None


# ===================================================================
# AuditStore Tests
# ===================================================================


class TestAuditStoreRecordAndQuery:
    """Record events and query them back."""

    async def test_record_and_query(self, audit_store) -> None:
        aid = await audit_store.record(
            event_type=EVENT_DASHBOARD_LOGIN_SUCCESS,
            result="success",
            actor_user_id=100,
        )
        assert aid is not None
        assert isinstance(aid, str)
        assert len(aid) > 0

        result = await audit_store.query(page=1, page_size=10)
        assert result["total"] == 1
        assert len(result["events"]) == 1
        assert result["events"][0]["event_type"] == EVENT_DASHBOARD_LOGIN_SUCCESS


class TestAuditStorePagination:
    """Pagination works correctly."""

    async def test_pagination(self, audit_store) -> None:
        for i in range(10):
            await audit_store.record(
                event_type=f"test.event.{i}",
                result="success" if i % 2 == 0 else "failure",
                actor_user_id=i,
            )

        # Page 1, size 3
        r1 = await audit_store.query(page=1, page_size=3)
        assert r1["total"] == 10
        assert r1["total_pages"] == 4
        assert len(r1["events"]) == 3

        # Page 4 should have 1 event
        r4 = await audit_store.query(page=4, page_size=3)
        assert len(r4["events"]) == 1


class TestAuditStoreFilterByEventType:
    """Filter audit events by event_type."""

    async def test_filter_by_event_type(self, audit_store) -> None:
        await audit_store.record(event_type="type.a", result="success")
        await audit_store.record(event_type="type.b", result="success")
        await audit_store.record(event_type="type.a", result="success")

        r = await audit_store.query(event_type="type.a")
        assert r["total"] == 2

        r2 = await audit_store.query(event_type="type.b")
        assert r2["total"] == 1

        r3 = await audit_store.query(event_type="type.nonexistent")
        assert r3["total"] == 0


class TestAuditStoreFilterByResult:
    """Filter audit events by result field."""

    async def test_filter_by_result(self, audit_store) -> None:
        await audit_store.record(event_type="evt.1", result="success")
        await audit_store.record(event_type="evt.2", result="failure")
        await audit_store.record(event_type="evt.3", result="success")

        r = await audit_store.query(result="success")
        assert r["total"] == 2

        r2 = await audit_store.query(result="failure")
        assert r2["total"] == 1


class TestAuditStoreGetSingleEvent:
    """Fetch a single audit event by ID."""

    async def test_get_single_event(self, audit_store) -> None:
        aid = await audit_store.record(
            event_type="test.single",
            result="success",
            actor_user_id=42,
            metadata={"foo": "bar"},
        )

        event = await audit_store.get_single_event(aid)
        assert event is not None
        assert event["event_type"] == "test.single"
        assert event["result"] == "success"
        assert event["metadata"] == {"foo": "bar"}

        # Non-existent ID
        none_event = await audit_store.get_single_event("nonexistent-uuid")
        assert none_event is None


class TestAuditStoreCleanupRetention:
    """Remove events older than retention period."""

    async def test_cleanup_retention(self, audit_store) -> None:
        old_ts = (datetime.now(UTC) - timedelta(days=365)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        import uuid

        # Initialize the store first so tables exist
        await audit_store.initialize()

        # Use the internal sync method to bypass the auto-generated created_at
        old_id = str(uuid.uuid4())

        audit_store._insert_sync(
            audit_id=old_id,
            event_type="test.old",
            result="success",
            actor_user_id=None,
            actor_source_ip=None,
            actor_user_agent="",
            target_user_id=None,
            target_guild_id=None,
            target_channel_id=None,
            message_id=None,
            error_code=None,
            content_preview=None,
            content_hash=None,
            metadata={},
        )
        # Manually update created_at to be old
        conn = audit_store._connect()
        conn.execute(
            "UPDATE audit_events SET created_at = ? WHERE audit_id = ?",
            (old_ts, old_id),
        )
        conn.commit()
        conn.close()

        # Insert a recent event
        await audit_store.record(event_type="test.new", result="success")

        cleaned = await audit_store.cleanup_retention()
        assert cleaned == 1

        # Old event gone, new event remains
        r = await audit_store.query()
        assert r["total"] == 1
        assert r["events"][0]["event_type"] == "test.new"


# ===================================================================
# DMStore Tests
# ===================================================================


class TestDMStoreUpsertUser:
    """Upsert a DM user."""

    async def test_upsert_user(self, dm_store) -> None:
        await dm_store.upsert_user(
            user_id=6001,
            username="dmuser1",
            global_name="DM User One",
            display_name="User1",
        )

        # Second upsert with updated name
        await dm_store.upsert_user(
            user_id=6001,
            username="dmuser1_updated",
            global_name="DM User One Updated",
        )

        # Verify by listing threads (no messages yet, so no threads)
        # We can verify indirectly by checking that no errors occurred
        threads = await dm_store.list_threads()
        assert threads["total"] == 0


class TestDMStoreAddMessage:
    """Add messages to a DM conversation."""

    async def test_add_message(self, dm_store) -> None:
        await dm_store.upsert_user(user_id=7001, username="sender")
        await dm_store.add_message(
            message_id=8001,
            channel_id=7001,
            author_id=7001,
            content="Hello from DM",
            clean_content="Hello from DM",
            is_bot_author=False,
        )

        threads = await dm_store.list_threads()
        assert threads["total"] == 1
        assert threads["threads"][0]["channel_id"] == "7001"
        assert threads["threads"][0]["last_preview"] == "Hello from DM"
        assert threads["threads"][0]["last_direction"] == "inbound"


class TestDMStoreListThreads:
    """List DM threads with latest message info."""

    async def test_list_threads(self, dm_store) -> None:
        # Two DM conversations
        for uid, mid, content in [
            (7101, 8101, "First msg from A"),
            (7102, 8102, "First from B"),
        ]:
            await dm_store.upsert_user(user_id=uid, username=f"user{uid}")
            await dm_store.add_message(
                message_id=mid,
                channel_id=uid,
                author_id=uid,
                content=content,
                clean_content=content,
            )

        threads = await dm_store.list_threads()
        assert threads["total"] == 2

        channel_ids = {t["channel_id"] for t in threads["threads"]}
        assert channel_ids == {"7101", "7102"}


class TestDMStoreGetThreadMessages:
    """Get paginated messages for a DM thread."""

    async def test_get_thread_messages(self, dm_store) -> None:
        await dm_store.upsert_user(user_id=7201, username="chatty")
        for i in range(5):
            await dm_store.add_message(
                message_id=8200 + i,
                channel_id=7201,
                author_id=7201,
                content=f"Msg {i}",
                clean_content=f"Msg {i}",
                created_at=f"2025-01-0{i+1}T00:00:00.000Z",
            )

        r1 = await dm_store.get_thread_messages(7201, page=1, page_size=2)
        assert r1["total"] == 5
        assert r1["total_pages"] == 3
        assert len(r1["messages"]) == 2

        r2 = await dm_store.get_thread_messages(7201, page=3, page_size=2)
        assert len(r2["messages"]) == 1


# ===================================================================
# DashboardConfig Tests
# ===================================================================


class TestDashboardConfigDefaults:
    """Default values when no env vars are set."""

    def test_defaults(self) -> None:
        cfg = DashboardConfig()
        assert cfg.enabled is False
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8011
        assert cfg.auth_token is None
        assert cfg.session_secret is None
        assert cfg.owner_ids == set()
        assert cfg.rate_limit_sends_per_minute == 5
        assert cfg.max_message_chars == 1800
        assert cfg.dm_retention_days == 180
        assert cfg.audit_retention_days == 180
        assert cfg.session_ttl_hours == 8
        assert cfg.page_size == 50
        assert cfg.message_db_path == "./data/dashboard_messages.db"
        assert cfg.backfill_db_path == "./data/dashboard_backfill.db"


class TestDashboardConfigCustomEnv:
    """Custom values via environment variables."""

    def test_custom_env_values(self, monkeypatch) -> None:
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")
        monkeypatch.setenv("DASHBOARD_PORT", "9090")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "s3cret!")
        monkeypatch.setenv("DASHBOARD_OWNER_IDS", "12345,67890")
        monkeypatch.setenv("DASHBOARD_RATE_LIMIT_SENDS_PER_MINUTE", "10")
        monkeypatch.setenv("DASHBOARD_MAX_MESSAGE_CHARS", "4000")
        monkeypatch.setenv("DASHBOARD_DM_RETENTION_DAYS", "365")
        monkeypatch.setenv("DASHBOARD_MESSAGE_DB_PATH", "/custom/path/msgs.db")
        monkeypatch.setenv("DASHBOARD_BACKFILL_DB_PATH", "/custom/path/bf.db")
        monkeypatch.setenv("DASHBOARD_SESSION_SECRET", "my-secret-key")

        cfg = load_dashboard_config()
        assert cfg.enabled is True
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9090
        assert cfg.auth_token == "s3cret!"
        assert cfg.owner_ids == {12345, 67890}
        assert cfg.rate_limit_sends_per_minute == 10
        assert cfg.max_message_chars == 4000
        assert cfg.dm_retention_days == 365
        assert cfg.message_db_path == "/custom/path/msgs.db"
        assert cfg.backfill_db_path == "/custom/path/bf.db"

    def test_auth_requirement(self, monkeypatch) -> None:
        """Dashboard should be disabled if auth_token is missing when enabled."""
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        # No DASHBOARD_AUTH_TOKEN set
        monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)

        cfg = load_dashboard_config()
        assert cfg.enabled is False  # Auto-disabled due to missing auth token
        assert cfg.auth_token is None

    def test_owner_ids_merged_with_existing(self, monkeypatch) -> None:
        """DASHBOARD_OWNER_IDS should merge with OWNER_IDS."""
        monkeypatch.setenv("OWNER_IDS", "99999")
        monkeypatch.setenv("DASHBOARD_OWNER_IDS", "11111,22222")
        monkeypatch.setenv("DASHBOARD_ENABLED", "true")
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "tok")

        cfg = load_dashboard_config()
        assert cfg.owner_ids == {99999, 11111, 22222}


class TestDashboardConfigBoolParsing:
    """Boolean parsing edge cases."""

    @pytest.mark.parametrize(
        ("env_val", "expected"),
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("enabled", True),
            ("enable", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("", False),
            (None, False),
        ],
    )
    def test_bool_parsing(self, monkeypatch, env_val, expected) -> None:
        if env_val is not None:
            monkeypatch.setenv("DASHBOARD_ENABLED", env_val)
        else:
            monkeypatch.delenv("DASHBOARD_ENABLED", raising=False)
        monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "tok-for-enabled")

        cfg = load_dashboard_config()
        assert cfg.enabled is expected


# ===================================================================
# Permission Tests
# ===================================================================


def _make_mock_bot(channel=None):
    """Helper to create a mock bot with get_channel returning the given channel."""
    bot = MagicMock()
    bot.get_channel.return_value = channel
    return bot


def _make_mock_channel(
    *,
    channel_id: int = 123,
    name: str = "test-channel",
    guild_id: int = 456,
    guild_name: str = "Test Guild",
    channel_type: str = "text",
    permissions: dict | None = None,
    is_thread: bool = False,
):
    """Build a mock discord-like channel with configurable permissions."""
    # Full set of defaults — every permission must be explicitly set to avoid
    # MagicMock auto-creating truthy mock attributes for omitted keys.
    full_defaults: dict[str, bool] = {
        "read_messages": True,
        "read_message_history": True,
        "send_messages": True,
        "send_messages_in_threads": True,
        "embed_links": True,
        "attach_files": True,
        "add_reactions": True,
        "use_external_emojis": True,
        "mention_everyone": False,
        "manage_messages": False,
        "manage_channels": False,
        "administrator": False,
    }
    if permissions is not None:
        full_defaults.update(permissions)

    guild = MagicMock()
    guild.id = guild_id
    guild.name = guild_name
    guild.me = MagicMock()

    channel = MagicMock()
    channel.id = channel_id
    channel.name = name
    channel.guild = guild
    channel.type = channel_type

    if is_thread:
        channel.parent = MagicMock()
        channel.parent.id = 999
    else:
        channel.parent = None

    # permissions_for(guild.me) returns a Permission-like mock
    perm_mock = MagicMock()
    for perm_name, perm_val in full_defaults.items():
        setattr(perm_mock, perm_name, perm_val)
    channel.permissions_for.return_value = perm_mock

    return channel


class TestPermissionsCanViewChannel:
    """can_view_channel behavior."""

    def test_sufficient_permissions(self) -> None:
        channel = _make_mock_channel()
        bot = _make_mock_bot(channel=channel)
        result = can_view_channel(bot, 123)
        assert result.allowed is True
        assert result.reason == "ok"
        assert result.permissions["read_messages"] is True

    def test_without_read_messages(self) -> None:
        channel = _make_mock_channel(permissions={"read_messages": False})
        bot = _make_mock_bot(channel=channel)
        result = can_view_channel(bot, 123)
        assert result.allowed is False
        assert "read_messages" in result.reason
        assert result.permissions["read_messages"] is False

    def test_channel_not_found(self) -> None:
        bot = _make_mock_bot(channel=None)
        result = can_view_channel(bot, 999)
        assert result.allowed is False
        assert "not found" in result.reason.lower()

    def test_channel_no_guild(self) -> None:
        channel = MagicMock()
        channel.id = 123
        channel.guild = None
        bot = _make_mock_bot(channel=channel)
        result = can_view_channel(bot, 123)
        assert result.allowed is False


class TestPermissionsCanSendMessages:
    """can_send_messages behavior."""

    def test_sufficient_permissions(self) -> None:
        channel = _make_mock_channel()
        bot = _make_mock_bot(channel=channel)
        result = can_send_messages(bot, 123)
        assert result.allowed is True
        assert result.reason == "ok"

    def test_without_send_messages(self) -> None:
        channel = _make_mock_channel(
            permissions={
                "read_messages": True,
                "send_messages": False,
                "send_messages_in_threads": True,
            },
        )
        bot = _make_mock_bot(channel=channel)
        result = can_send_messages(bot, 123)
        assert result.allowed is False
        assert "send messages" in result.reason

    def test_administrator_override(self) -> None:
        channel = _make_mock_channel(
            permissions={
                "read_messages": True,
                "send_messages": False,
                "administrator": True,
            },
        )
        bot = _make_mock_bot(channel=channel)
        result = can_send_messages(bot, 123)
        assert result.allowed is True
        assert "administrator" in result.reason

    def test_thread_channel_uses_send_messages_in_threads(self) -> None:
        channel = _make_mock_channel(
            is_thread=True,
            permissions={
                "read_messages": True,
                "send_messages": False,
                "send_messages_in_threads": True,
            },
        )
        bot = _make_mock_bot(channel=channel)
        result = can_send_messages(bot, 123)
        assert result.allowed is True

    def test_thread_channel_without_permission(self) -> None:
        channel = _make_mock_channel(
            is_thread=True,
            permissions={
                "read_messages": True,
                "send_messages": True,
                "send_messages_in_threads": False,
            },
        )
        bot = _make_mock_bot(channel=channel)
        result = can_send_messages(bot, 123)
        assert result.allowed is False

    def test_channel_not_found(self) -> None:
        bot = _make_mock_bot(channel=None)
        result = can_send_messages(bot, 999)
        assert result.allowed is False


class TestPermissionsCanReadMessageHistory:
    """can_read_message_history behavior."""

    def test_sufficient_permissions(self) -> None:
        channel = _make_mock_channel()
        bot = _make_mock_bot(channel=channel)
        result = can_read_message_history(bot, 123)
        assert result.allowed is True

    def test_missing_read_message_history(self) -> None:
        channel = _make_mock_channel(
            permissions={
                "read_messages": True,
                "read_message_history": False,
            },
        )
        bot = _make_mock_bot(channel=channel)
        result = can_read_message_history(bot, 123)
        assert result.allowed is False
        assert "read_message_history" in result.reason

    def test_administrator_override(self) -> None:
        channel = _make_mock_channel(
            permissions={
                "read_messages": False,
                "read_message_history": False,
                "administrator": True,
            },
        )
        bot = _make_mock_bot(channel=channel)
        result = can_read_message_history(bot, 123)
        assert result.allowed is True


class TestPermissionsGetChannelPermissions:
    """get_channel_permissions returns expected structure."""

    def test_returns_expected_structure(self) -> None:
        channel = _make_mock_channel()
        bot = _make_mock_bot(channel=channel)

        result = get_channel_permissions(bot, 123)
        assert result["found"] is True
        assert result["channel_id"] == "123"
        assert result["channel_name"] == "test-channel"
        assert result["guild_id"] == "456"
        assert result["guild_name"] == "Test Guild"
        assert result["channel_type"] == "text"
        assert isinstance(result["permissions"], dict)
        assert isinstance(result["permission_summary"], list)
        assert "read messages" in result["permission_summary"]

    def test_channel_not_found(self) -> None:
        bot = _make_mock_bot(channel=None)
        result = get_channel_permissions(bot, 999)
        assert result["found"] is False
        assert result["channel_id"] == "999"
        assert result["permission_summary"] == []

    def test_summary_contains_granted_permissions(self) -> None:
        channel = _make_mock_channel(
            permissions={
                "read_messages": True,
                "send_messages": True,
                "embed_links": False,
                "administrator": False,
            },
        )
        bot = _make_mock_bot(channel=channel)
        result = get_channel_permissions(bot, 123)
        # Only granted permissions appear in summary
        assert "read messages" in result["permission_summary"]
        assert "send messages" in result["permission_summary"]
        assert "embed links" not in result["permission_summary"]


# ===================================================================
# BackfillJobStore Tests
# ===================================================================


class TestBackfillJobStoreCreateJob:
    """Create a backfill job."""

    async def test_create_job(self, bj_store) -> None:
        job_id = await bj_store.create_job(
            target_type="channel",
            target_id="2001",
        )
        assert job_id is not None
        assert isinstance(job_id, str)

        job = await bj_store.get_job(job_id)
        assert job is not None
        assert job["target_type"] == "channel"
        assert job["target_id"] == "2001"
        assert job["status"] == BACKFILL_STATUS_QUEUED

    async def test_create_duplicate_target_returns_existing(self, bj_store) -> None:
        jid1 = await bj_store.create_job(target_type="channel", target_id="2002")
        jid2 = await bj_store.create_job(target_type="channel", target_id="2002")
        # Should return the same job_id (INSERT OR IGNORE)
        assert jid1 == jid2


class TestBackfillJobStoreUpdateStatus:
    """Update job status with valid transitions."""

    async def test_queued_to_running(self, bj_store) -> None:
        jid = await bj_store.create_job(target_type="guild", target_id="g100")
        ok = await bj_store.update_status(jid, BACKFILL_STATUS_RUNNING)
        assert ok is True

        job = await bj_store.get_job(jid)
        assert job["status"] == BACKFILL_STATUS_RUNNING
        assert job["started_at"] is not None

    async def test_running_to_completed(self, bj_store) -> None:
        jid = await bj_store.create_job(target_type="guild", target_id="g101")
        await bj_store.update_status(jid, BACKFILL_STATUS_RUNNING)
        ok = await bj_store.update_status(
            jid,
            BACKFILL_STATUS_COMPLETED,
            messages_seen=50,
            messages_inserted=48,
        )
        assert ok is True

        job = await bj_store.get_job(jid)
        assert job["status"] == BACKFILL_STATUS_COMPLETED
        assert job["messages_seen"] == 50
        assert job["messages_inserted"] == 48
        assert job["finished_at"] is not None

    async def test_invalid_transition(self, bj_store) -> None:
        jid = await bj_store.create_job(target_type="dm", target_id="u100")
        # Cannot go directly from queued to completed
        ok = await bj_store.update_status(jid, BACKFILL_STATUS_COMPLETED)
        assert ok is False

        job = await bj_store.get_job(jid)
        assert job["status"] == BACKFILL_STATUS_QUEUED  # unchanged


class TestBackfillJobStoreListJobs:
    """List jobs with pagination and filters."""

    async def test_list_jobs(self, bj_store) -> None:
        for i in range(5):
            await bj_store.create_job(target_type="channel", target_id=f"c{i}")

        result = await bj_store.list_jobs(page=1, page_size=2)
        assert result["total"] == 5
        assert result["total_pages"] == 3
        assert len(result["jobs"]) == 2

    async def test_list_jobs_with_status_filter(self, bj_store) -> None:
        jid1 = await bj_store.create_job(target_type="channel", target_id="c100")
        jid2 = await bj_store.create_job(target_type="channel", target_id="c101")

        await bj_store.update_status(jid1, BACKFILL_STATUS_RUNNING)

        r1 = await bj_store.list_jobs(status_filter=BACKFILL_STATUS_RUNNING)
        assert r1["total"] == 1

        r2 = await bj_store.list_jobs(status_filter=BACKFILL_STATUS_QUEUED)
        assert r2["total"] == 1

    async def test_list_jobs_with_type_filter(self, bj_store) -> None:
        await bj_store.create_job(target_type="channel", target_id="c200")
        await bj_store.create_job(target_type="guild", target_id="g200")
        await bj_store.create_job(target_type="dm", target_id="u200")

        r = await bj_store.list_jobs(target_type_filter="guild")
        assert r["total"] == 1
        assert r["jobs"][0]["target_type"] == "guild"


class TestBackfillJobStoreCancelJob:
    """Cancel a running or queued job."""

    async def test_cancel_queued_job(self, bj_store) -> None:
        jid = await bj_store.create_job(target_type="channel", target_id="c300")
        ok = await bj_store.cancel_job(jid)
        assert ok is True

        job = await bj_store.get_job(jid)
        assert job["status"] == BACKFILL_STATUS_CANCELLED

    async def test_cancel_running_job(self, bj_store) -> None:
        jid = await bj_store.create_job(target_type="channel", target_id="c301")
        await bj_store.update_status(jid, BACKFILL_STATUS_RUNNING)
        ok = await bj_store.cancel_job(jid)
        assert ok is True

        job = await bj_store.get_job(jid)
        assert job["status"] == BACKFILL_STATUS_CANCELLED

    async def test_cancel_completed_job_fails(self, bj_store) -> None:
        jid = await bj_store.create_job(target_type="channel", target_id="c302")
        await bj_store.update_status(jid, BACKFILL_STATUS_RUNNING)
        await bj_store.update_status(jid, BACKFILL_STATUS_COMPLETED)
        ok = await bj_store.cancel_job(jid)
        assert ok is False


class TestBackfillJobStoreResetStale:
    """Reset stuck 'running' jobs back to 'queued'."""

    async def test_reset_stale_jobs(self, bj_store) -> None:
        jid1 = await bj_store.create_job(target_type="channel", target_id="c400")
        jid2 = await bj_store.create_job(target_type="channel", target_id="c401")

        await bj_store.update_status(jid1, BACKFILL_STATUS_RUNNING)
        await bj_store.update_status(jid2, BACKFILL_STATUS_RUNNING)

        # One completed job should NOT be reset
        jid3 = await bj_store.create_job(target_type="channel", target_id="c402")
        await bj_store.update_status(jid3, BACKFILL_STATUS_RUNNING)
        await bj_store.update_status(jid3, BACKFILL_STATUS_COMPLETED)

        count = await bj_store.reset_stale_jobs()
        assert count == 2

        j1 = await bj_store.get_job(jid1)
        assert j1["status"] == BACKFILL_STATUS_QUEUED

        j3 = await bj_store.get_job(jid3)
        assert j3["status"] == BACKFILL_STATUS_COMPLETED  # unchanged
