from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from typing import Never

import pytest

from bot.server_archive.ingestion_queue import ArchiveIngestionQueue
from bot.server_archive.models import (
    ArchiveAttachment,
    ArchiveChannel,
    ArchiveGuild,
    ArchiveMention,
    ArchiveMessage,
    ArchiveMessageBundle,
    ArchiveUser,
)
from bot.server_archive.service import ServerArchiveService
from bot.server_archive.store import ServerArchiveStore


@pytest.fixture
def bundle_factory():
    def _make(
        message_id: str,
        content: str,
        guild_id: str = "1",
        channel_id: str = "10",
        author_id: str = "20",
    ):
        guild = ArchiveGuild(guild_id=guild_id, name="guild")
        channel = ArchiveChannel(channel_id=channel_id, guild_id=guild_id, name="general", type="text")
        author = ArchiveUser(user_id=author_id, username="alice", display_name="Alice")
        message = ArchiveMessage(
            message_id=message_id,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=None,
            author_id=author_id,
            content=content,
            clean_content=content,
            created_at="2026-05-08T00:00:00+00:00",
            jump_url=f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}",
        )
        return ArchiveMessageBundle(
            guild=guild,
            channel=channel,
            author=author,
            message=message,
            attachments=(
                ArchiveAttachment(
                    attachment_id=f"a-{message_id}",
                    message_id=message_id,
                    filename="x.txt",
                    url="https://cdn.example/x.txt",
                ),
            ),
            mentions=(ArchiveMention(message_id=message_id, mentioned_user_id="99"),),
        )

    return _make


@pytest.mark.asyncio
async def test_schema_bootstrap_idempotent_and_wal(tmp_path) -> None:
    store = ServerArchiveStore(tmp_path / "archive.sqlite3")
    await store.initialize()
    await store.initialize()
    conn = sqlite3.connect(store.sqlite_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='archive_messages'").fetchone()
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='archive_messages_fts'").fetchone()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_distiller_tables_are_lazy_created_for_existing_db(tmp_path) -> None:
    db_path = tmp_path / "archive.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA user_version=1")
        conn.commit()
    finally:
        conn.close()

    store = ServerArchiveStore(db_path)
    await store.initialize()
    await store.start_distiller_run("run-1", started_at="2026-05-09T00:00:00+00:00")
    latest = await store.latest_distiller_run()
    assert latest is not None
    assert latest["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_message_attachment_and_fts_search(tmp_path, bundle_factory) -> None:
    store = ServerArchiveStore(tmp_path / "archive.sqlite3")
    await store.initialize()
    await store.upsert_bundle(bundle_factory("100", "hello archive search world"))
    await store.upsert_bundle(bundle_factory("101", "completely different text"))

    counts = await store.counts(guild_id="1")
    assert counts["messages"] == 2
    assert counts["attachments"] == 2
    assert counts["mentions"] == 2

    results = await store.search("archive search", guild_id="1", limit=5)
    assert results
    assert results[0].message_id == "100"
    assert "archive search" in (results[0].clean_content or results[0].content)


@pytest.mark.asyncio
async def test_delete_tombstone_excludes_from_search(tmp_path, bundle_factory) -> None:
    store = ServerArchiveStore(tmp_path / "archive.sqlite3")
    await store.initialize()
    await store.upsert_bundle(bundle_factory("200", "searchable tombstone message"))
    assert await store.soft_delete_message("200")
    results = await store.search("searchable", guild_id="1", limit=5)
    assert all(result.message_id != "200" for result in results)


@pytest.mark.asyncio
async def test_search_is_guild_scoped_and_limit_is_enforced(tmp_path, bundle_factory) -> None:
    store = ServerArchiveStore(tmp_path / "archive.sqlite3")
    await store.initialize()
    for idx in range(20):
        guild_id = "1" if idx < 12 else "2"
        await store.upsert_bundle(bundle_factory(str(idx), f"needle {idx}", guild_id=guild_id, channel_id=str(10 + idx)))
    results = await store.search("needle", guild_id="1", limit=50)
    assert len(results) == 10
    assert all(result.guild_id == "1" for result in results)


@pytest.mark.asyncio
async def test_queue_full_drops_writes_without_blocking(bundle_factory) -> None:
    seen = []

    async def persist(batch) -> None:
        seen.append([item.message.message_id for item in batch])
        await asyncio.sleep(0.05)

    queue = ArchiveIngestionQueue(persist, max_size=1, workers=1, batch_size=1, enabled=True)
    assert await queue.enqueue(bundle_factory("1", "one"))
    assert await queue.enqueue(bundle_factory("2", "two")) is False
    assert queue.stats.dropped == 1


@pytest.mark.asyncio
async def test_live_tail_ignores_dm_and_bot_messages(monkeypatch, tmp_path) -> None:
    cfg = {
        "SERVER_ARCHIVE_ENABLE": True,
        "SERVER_ARCHIVE_DB_PATH": str(tmp_path / "archive.sqlite3"),
        "SERVER_ARCHIVE_QUEUE_MAX": 10,
        "SERVER_ARCHIVE_BATCH_SIZE": 1,
        "SERVER_ARCHIVE_SEARCH_LIMIT": 5,
        "SERVER_ARCHIVE_ADMIN_ONLY": True,
        "SERVER_ARCHIVE_SYNC_ON_START": False,
        "SERVER_ARCHIVE_LIVE_TAIL": True,
        "SERVER_ARCHIVE_MAX_MESSAGE_CHARS": 8000,
        "SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES": False,
    }
    import bot.server_archive.service as service_module

    monkeypatch.setattr(service_module, "load_config", lambda: cfg)
    service = ServerArchiveService(bot=None)
    await service.start()
    try:
        dm_message = SimpleNamespace(
            guild=None,
            channel=SimpleNamespace(id=1),
            author=SimpleNamespace(id=2, bot=False),
            content="hello",
            attachments=[],
        )
        bot_message = SimpleNamespace(
            guild=SimpleNamespace(id=1),
            channel=SimpleNamespace(id=2),
            author=SimpleNamespace(id=3, bot=True),
            content="hello",
            attachments=[],
        )
        assert await service.enqueue_live_message(dm_message) is False
        assert await service.enqueue_live_message(bot_message) is False
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_full_sync_does_not_start_twice(monkeypatch, tmp_path) -> None:
    cfg = {
        "SERVER_ARCHIVE_ENABLE": True,
        "SERVER_ARCHIVE_DB_PATH": str(tmp_path / "archive.sqlite3"),
        "SERVER_ARCHIVE_QUEUE_MAX": 10,
        "SERVER_ARCHIVE_BATCH_SIZE": 1,
        "SERVER_ARCHIVE_SEARCH_LIMIT": 5,
        "SERVER_ARCHIVE_ADMIN_ONLY": True,
        "SERVER_ARCHIVE_SYNC_ON_START": False,
        "SERVER_ARCHIVE_LIVE_TAIL": True,
        "SERVER_ARCHIVE_MAX_MESSAGE_CHARS": 8000,
        "SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES": False,
    }
    import bot.server_archive.service as service_module

    monkeypatch.setattr(service_module, "load_config", lambda: cfg)
    service = ServerArchiveService(bot=None)
    await service.start()

    async def slow_sync(store, guild, *, force=False) -> int:
        await asyncio.sleep(0.2)
        return 1

    monkeypatch.setattr(service_module, "_sync_guild_archive", slow_sync)
    guild = SimpleNamespace(id=123, text_channels=[], threads=[])
    first = asyncio.create_task(service.sync_guild(guild))
    await asyncio.sleep(0.01)
    second = await service.sync_guild(guild)
    assert second == 0
    await first
    await service.stop()


@pytest.mark.asyncio
async def test_permission_errors_are_logged_and_skipped(tmp_path) -> None:
    store = ServerArchiveStore(tmp_path / "archive.sqlite3")
    await store.initialize()

    class PermissionErrorHistory:
        async def history(self, **kwargs) -> Never:
            msg = "forbidden"
            raise PermissionError(msg)

    guild = SimpleNamespace(id=1, text_channels=[PermissionErrorHistory()], threads=[])
    result = await __import__("bot.server_archive.sync", fromlist=["sync_guild_archive"]).sync_guild_archive(store, guild)
    assert result == 0
