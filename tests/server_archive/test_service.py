from __future__ import annotations

import inspect
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Never

import pytest

import bot.server_archive.service as service_module
import bot.server_archive.sync as archive_sync
from bot.server_archive.models import (
    ArchiveChannel,
    ArchiveGuild,
    ArchiveMessage,
    ArchiveMessageBundle,
    ArchiveUser,
)
from bot.server_archive.service import ServerArchiveService
from bot.server_archive.store import ServerArchiveStore

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC).isoformat()


def make_bundle(*, message_id: str, guild_id: str, channel_id: str, content: str) -> ArchiveMessageBundle:
    guild = ArchiveGuild(guild_id=guild_id, name=f"Guild {guild_id}")
    channel = ArchiveChannel(
        channel_id=channel_id,
        guild_id=guild_id,
        name=f"channel-{channel_id}",
        type="text",
    )
    author = ArchiveUser(user_id=f"user-{guild_id}", username="alice", display_name="Alice")
    message = ArchiveMessage(
        message_id=message_id,
        guild_id=guild_id,
        channel_id=channel_id,
        thread_id=None,
        author_id=author.user_id,
        content=content,
        clean_content=content,
        created_at=FIXED_NOW,
        jump_url=f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}",
    )
    return ArchiveMessageBundle(guild=guild, channel=channel, author=author, message=message)


class FakeHistoryChannel:
    def __init__(
        self,
        guild: SimpleNamespace,
        channel_id: str,
        *,
        messages: list[SimpleNamespace] | None = None,
    ) -> None:
        self.guild = guild
        self.id = channel_id
        self.name = f"channel-{channel_id}"
        self.parent_id = None
        self._messages = list(messages or [])
        self.history_calls: list[dict] = []

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        after = kwargs.get("after")
        after_id = str(getattr(after, "id", "")) if after is not None else None

        async def iterator():
            for message in self._messages:
                if after_id is not None and int(message.id) <= int(after_id):
                    continue
                yield message

        return iterator()


class PermissionDeniedChannel(FakeHistoryChannel):
    def history(self, **kwargs):
        self.history_calls.append(kwargs)

        async def iterator():
            msg = "not allowed"
            raise PermissionError(msg)
            yield  # pragma: no cover

        return iterator()


@pytest.fixture
def archive_config(tmp_path):
    return {
        "SERVER_ARCHIVE_ENABLED": True,
        "SERVER_ARCHIVE_ENABLE": True,
        "SERVER_ARCHIVE_DB_PATH": str(tmp_path / "archive.sqlite3"),
        "SERVER_ARCHIVE_QUEUE_MAX": 2,
        "SERVER_ARCHIVE_BATCH_SIZE": 1,
        "SERVER_ARCHIVE_SEARCH_LIMIT": 10,
        "SERVER_ARCHIVE_ADMIN_ONLY": True,
        "SERVER_ARCHIVE_SYNC_ON_START": False,
        "SERVER_ARCHIVE_LIVE_TAIL": True,
        "SERVER_ARCHIVE_MAX_MESSAGE_CHARS": 8000,
        "SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES": False,
        "SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES": False,
    }


@pytest.mark.asyncio
async def test_live_tail_enqueue_ignores_dms_and_bot_messages_by_default(tmp_path, monkeypatch: pytest.MonkeyPatch, archive_config) -> None:
    monkeypatch.setattr(service_module, "load_config", lambda: archive_config)
    service = ServerArchiveService(SimpleNamespace(command_prefix="!"))
    await service.start()
    try:
        dm_message = SimpleNamespace(
            guild=None,
            channel=SimpleNamespace(id=10),
            author=SimpleNamespace(id=20, bot=False, display_name="User", name="User"),
            content="hello from a dm",
            attachments=[],
        )
        bot_message = SimpleNamespace(
            guild=SimpleNamespace(id=1),
            channel=SimpleNamespace(id=11),
            author=SimpleNamespace(id=21, bot=True, display_name="Bot", name="Bot"),
            content="hello from a bot",
            attachments=[],
        )

        assert await service.enqueue_live_message(dm_message) is False
        assert await service.enqueue_live_message(bot_message) is False
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_live_tail_enqueue_does_not_write_to_sqlite_immediately(tmp_path, monkeypatch: pytest.MonkeyPatch, archive_config) -> None:
    monkeypatch.setattr(service_module, "load_config", lambda: archive_config)
    service = ServerArchiveService(SimpleNamespace(command_prefix="!"))
    await service.start()
    try:
        called = False

        async def fail_upsert(*args, **kwargs) -> Never:
            nonlocal called
            called = True
            msg = "archive writes should not happen inline"
            raise AssertionError(msg)

        monkeypatch.setattr(service.store, "upsert_bundles", fail_upsert)

        message = SimpleNamespace(
            guild=SimpleNamespace(id=1, name="Guild"),
            channel=SimpleNamespace(id=11, name="general"),
            author=SimpleNamespace(id=21, bot=False, display_name="User", name="User"),
            content="hello",
            clean_content="hello",
            attachments=[],
            mentions=[],
            created_at=datetime.now(UTC),
            edited_at=None,
            jump_url=None,
            reference=None,
        )
        assert await service.enqueue_live_message(message) is True
        assert called is False
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_pause_blocks_live_tail_enqueue(tmp_path, monkeypatch: pytest.MonkeyPatch, archive_config) -> None:
    monkeypatch.setattr(service_module, "load_config", lambda: archive_config)
    service = ServerArchiveService(SimpleNamespace(command_prefix="!"))
    await service.start()
    try:
        service.pause()
        message = SimpleNamespace(
            guild=SimpleNamespace(id=1, name="Guild"),
            channel=SimpleNamespace(id=11, name="general"),
            author=SimpleNamespace(id=21, bot=False, display_name="User", name="User"),
            content="hello",
            clean_content="hello",
            attachments=[],
            mentions=[],
            created_at=datetime.now(UTC),
            edited_at=None,
            jump_url=None,
            reference=None,
        )
        assert await service.enqueue_live_message(message) is False
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_sync_checkpoint_is_stored_and_reused(tmp_path, archive_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module, "load_config", lambda: archive_config)
    store = ServerArchiveStore(tmp_path / "archive.sqlite3")
    await store.initialize()

    guild = SimpleNamespace(id="123", name="Guild", channels=[], threads=[])
    channel = FakeHistoryChannel(
        guild,
        "456",
        messages=[
            SimpleNamespace(
                id="100",
                guild=guild,
                channel=None,
                author=SimpleNamespace(id="200", bot=False, display_name="Alice", name="Alice"),
                content="first",
                clean_content="first",
                attachments=[],
                created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                edited_at=None,
                jump_url=None,
                reference=None,
                mentions=[],
                embeds=[],
            ),
            SimpleNamespace(
                id="101",
                guild=guild,
                channel=None,
                author=SimpleNamespace(id="200", bot=False, display_name="Alice", name="Alice"),
                content="second",
                clean_content="second",
                attachments=[],
                created_at=datetime(2026, 1, 1, 12, 1, tzinfo=UTC),
                edited_at=None,
                jump_url=None,
                reference=None,
                mentions=[],
                embeds=[],
            ),
        ],
    )
    for message in channel._messages:
        message.channel = channel

    first = await archive_sync.sync_channel_archive(store, channel)
    state = await store.get_sync_state(guild_id="123", channel_id="456")
    assert first == 2
    assert state is not None
    assert state.last_message_id == "101"
    assert channel.history_calls[0]["oldest_first"] is True
    assert channel.history_calls[0]["limit"] is None

    channel._messages.extend(
        [
            SimpleNamespace(
                id="102",
                guild=guild,
                channel=channel,
                author=SimpleNamespace(id="200", bot=False, display_name="Alice", name="Alice"),
                content="third",
                clean_content="third",
                attachments=[],
                created_at=datetime(2026, 1, 1, 12, 2, tzinfo=UTC),
                edited_at=None,
                jump_url=None,
                reference=None,
                mentions=[],
                embeds=[],
            ),
        ],
    )

    second = await archive_sync.sync_channel_archive(store, channel)
    state = await store.get_sync_state(guild_id="123", channel_id="456")
    assert second == 1
    assert state is not None
    assert state.last_message_id == "102"
    assert channel.history_calls[-1]["after"] is not None
    assert str(channel.history_calls[-1]["after"].id) == "101"


@pytest.mark.asyncio
async def test_failed_channel_sync_does_not_abort_whole_guild_sync(tmp_path, archive_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module, "load_config", lambda: archive_config)
    store = ServerArchiveStore(tmp_path / "archive.sqlite3")
    await store.initialize()

    guild = SimpleNamespace(id="999", name="Guild", channels=[], threads=[])
    good_channel = FakeHistoryChannel(
        guild,
        "111",
        messages=[
            SimpleNamespace(
                id="500",
                guild=guild,
                channel=None,
                author=SimpleNamespace(id="200", bot=False, display_name="Alice", name="Alice"),
                content="ok",
                clean_content="ok",
                attachments=[],
                created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                edited_at=None,
                jump_url=None,
                reference=None,
                mentions=[],
                embeds=[],
            ),
        ],
    )
    bad_channel = PermissionDeniedChannel(guild, "222")
    for message in good_channel._messages:
        message.channel = good_channel
    guild.channels = [good_channel, bad_channel]

    processed = await archive_sync.sync_guild_archive(store, guild)
    state = await store.get_sync_state(guild_id="999")

    assert processed == 1
    assert state is not None
    assert state.status == "complete_with_errors"
    assert good_channel.history_calls
    assert bad_channel.history_calls
    assert (await store.counts(guild_id="999"))["messages"] == 1


@pytest.mark.asyncio
async def test_archive_module_does_not_pull_curated_memory_or_chromadb() -> None:
    source = inspect.getsource(service_module)
    forbidden = [
        "build_memory_prompt_block",
        "semantic_search",
        "chromadb",
        "ChromaDB",
        "persistent_store",
    ]
    assert all(token not in source for token in forbidden)
