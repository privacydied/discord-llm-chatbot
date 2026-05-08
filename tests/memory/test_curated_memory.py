import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.core.bot import LLMBot
from bot.memory.curator import CuratedMemoryCurator
from bot.memory.ingestion_queue import CuratedMemoryIngestionQueue
from bot.memory.persistent_store import MemoryRecord, PersistentMemoryStore
from bot.memory.service import CuratedMemoryService
from bot.memory.scoring import combined_score, recency_score


class FakeSemanticStore:
    def __init__(self, records=None):
        self.records = records or {}
        self.calls = []
        self.upserts = []
        self.deleted = []
        self.deleted_many = []
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def upsert(self, memory_id, document, metadata):
        self.upserts.append((memory_id, document, metadata))
        self.records[memory_id] = {"memory_id": memory_id, "document": document, "metadata": metadata}
        return memory_id

    async def delete(self, memory_id):
        self.deleted.append(memory_id)
        self.records.pop(memory_id, None)

    async def delete_many(self, memory_ids):
        self.deleted_many.append(list(memory_ids))
        for memory_id in memory_ids:
            self.records.pop(memory_id, None)

    async def query(self, query, top_k=6, where=None, where_document=None):
        self.calls.append({"query": query, "top_k": top_k, "where": where, "where_document": where_document})
        results = []
        for payload in self.records.values():
            metadata = payload["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            if query.lower() not in (payload["document"] or "").lower():
                continue
            results.append(
                {
                    "memory_id": payload["memory_id"],
                    "document": payload["document"],
                    "metadata": payload["metadata"],
                    "semantic_score": 0.92,
                }
            )
        return results[:top_k]


class DummyQueue:
    def __init__(self):
        self.enqueued = []

    async def enqueue(self, candidate):
        self.enqueued.append(candidate)
        return True


@pytest.fixture
async def persistent_store(tmp_path):
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    return store


@pytest.fixture
def make_record():
    def _make_record(**kwargs):
        now = datetime.now(timezone.utc).isoformat()
        payload = dict(
            memory_id="mem-1",
            user_id="user-1",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            source_message_id=None,
            context_type="user_preference",
            text="I prefer dark mode",
            summary="prefers dark mode",
            importance=0.9,
            confidence=0.95,
            created_at=now,
            updated_at=now,
            last_accessed_at=None,
            expires_at=None,
            source="explicit_memory_command",
            deleted_at=None,
            chroma_id=None,
            metadata_json="{}",
        )
        payload.update(kwargs)
        return MemoryRecord(**payload)

    return _make_record


@pytest.fixture
def service(tmp_path):
    store = PersistentMemoryStore(tmp_path / "memory.db")
    fake_semantic = FakeSemanticStore()
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()
    return svc


@pytest.mark.asyncio
async def test_schema_bootstrap_is_idempotent(tmp_path):
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    await store.initialize()

    import sqlite3

    conn = sqlite3.connect(tmp_path / "memory.db")
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='curated_memories'"
        ).fetchone()
    finally:
        conn.close()

    assert version == 1
    assert table_exists is not None


@pytest.mark.asyncio
async def test_explicit_memory_add_stores_sqlite_and_semantic(service):
    record = await service.add_explicit_memory(
        user_id="user-1",
        text="I prefer dark mode",
        guild_id="guild-1",
        channel_id="channel-1",
        source_message_id="msg-1",
        metadata={"origin": "test"},
    )

    stored = await service.store.get_memory(record.memory_id)
    assert stored is not None
    assert stored.summary == "Prefers I prefer dark mode"
    assert service.semantic_store.upserts
    memory_id, document, metadata = service.semantic_store.upserts[0]
    assert memory_id == record.memory_id
    assert document == record.summary
    assert metadata["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_delete_and_wipe_remove_from_retrieval(tmp_path):
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    now = datetime.now(timezone.utc).isoformat()
    active = MemoryRecord(
        memory_id="mem-active",
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        source_message_id=None,
        context_type="user_preference",
        text="I prefer dark mode",
        summary="prefers dark mode",
        importance=0.9,
        confidence=0.95,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        expires_at=None,
        source="explicit_memory_command",
        deleted_at=None,
        chroma_id=None,
        metadata_json="{}",
    )
    wiped = MemoryRecord(**{**active.to_dict(), "memory_id": "mem-wipe", "summary": "likes tea"})
    await store.upsert_memory(active)
    await store.upsert_memory(wiped)

    fake_semantic = FakeSemanticStore(
        {
            "mem-active": {
                "memory_id": "mem-active",
                "document": "prefers dark mode",
                "metadata": {"user_id": "user-1", "guild_id": "guild-1", "channel_id": "channel-1", "context_type": "user_preference", "created_at": now, "importance": 0.9, "confidence": 0.95},
            },
            "mem-wipe": {
                "memory_id": "mem-wipe",
                "document": "likes tea",
                "metadata": {"user_id": "user-1", "guild_id": "guild-1", "channel_id": "channel-1", "context_type": "user_preference", "created_at": now, "importance": 0.9, "confidence": 0.95},
            },
        }
    )
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()

    before = await svc.build_prompt_block(
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        query="dark mode",
        top_k=6,
        max_chars=200,
    )
    assert "prefers dark mode" in before

    await svc.delete_memory("mem-active")
    after_delete = await svc.build_prompt_block(
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        query="dark mode",
        top_k=6,
        max_chars=200,
    )
    assert "prefers dark mode" not in after_delete
    assert "mem-active" in fake_semantic.deleted

    wiped_count = await svc.wipe_user_memories("user-1")
    assert wiped_count == 1
    assert fake_semantic.deleted_many
    after_wipe = await svc.build_prompt_block(
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        query="tea",
        top_k=6,
        max_chars=200,
    )
    assert after_wipe == ""


@pytest.mark.asyncio
async def test_expired_memories_are_not_returned(tmp_path):
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    record = MemoryRecord(
        memory_id="mem-expired",
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        source_message_id=None,
        context_type="temporary_context",
        text="Temporary note",
        summary="temporary note",
        importance=0.4,
        confidence=0.8,
        created_at=past,
        updated_at=past,
        last_accessed_at=None,
        expires_at=past,
        source="inferred_curated",
        deleted_at=None,
        chroma_id=None,
        metadata_json="{}",
    )
    await store.upsert_memory(record)
    fake_semantic = FakeSemanticStore(
        {
            "mem-expired": {
                "memory_id": "mem-expired",
                "document": "temporary note",
                "metadata": {"user_id": "user-1", "guild_id": "guild-1", "channel_id": "channel-1", "context_type": "temporary_context", "created_at": past, "expires_at": past, "importance": 0.4, "confidence": 0.8},
            }
        }
    )
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()

    block = await svc.build_prompt_block(
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        query="temporary",
        top_k=6,
        max_chars=200,
    )
    assert block == ""


@pytest.mark.asyncio
async def test_retrieval_filters_by_scope(tmp_path):
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    now = datetime.now(timezone.utc).isoformat()
    rec_user = MemoryRecord(
        memory_id="mem-user",
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        source_message_id=None,
        context_type="user_preference",
        text="I prefer dark mode fact",
        summary="prefers dark mode fact",
        importance=0.9,
        confidence=0.95,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        expires_at=None,
        source="explicit_memory_command",
        deleted_at=None,
        chroma_id=None,
        metadata_json="{}",
    )
    rec_guild = MemoryRecord(**{**rec_user.to_dict(), "memory_id": "mem-guild", "user_id": "user-2", "summary": "guild fact", "text": "guild fact"})
    rec_channel = MemoryRecord(**{**rec_user.to_dict(), "memory_id": "mem-channel", "user_id": "user-2", "channel_id": "channel-9", "summary": "channel fact", "text": "channel fact"})
    await store.upsert_memory(rec_user)
    await store.upsert_memory(rec_guild)
    await store.upsert_memory(rec_channel)

    fake_semantic = FakeSemanticStore(
        {
            "mem-user": {
                "memory_id": "mem-user",
                "document": "prefers dark mode fact",
                "metadata": {
                    "user_id": "user-1",
                    "guild_id": "guild-1",
                    "channel_id": "channel-1",
                    "context_type": "user_preference",
                    "created_at": now,
                    "importance": 0.9,
                    "confidence": 0.95,
                },
            },
            "mem-guild": {
                "memory_id": "mem-guild",
                "document": "guild fact",
                "metadata": {
                    "user_id": "user-2",
                    "guild_id": "guild-1",
                    "channel_id": "channel-1",
                    "context_type": "server_fact",
                    "created_at": now,
                    "importance": 0.9,
                    "confidence": 0.95,
                },
            },
            "mem-channel": {
                "memory_id": "mem-channel",
                "document": "channel fact",
                "metadata": {
                    "user_id": "user-2",
                    "guild_id": "guild-1",
                    "channel_id": "channel-9",
                    "context_type": "server_fact",
                    "created_at": now,
                    "importance": 0.9,
                    "confidence": 0.95,
                },
            },
        }
    )
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()

    block = await svc.build_prompt_block(
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        query="fact",
        top_k=6,
        max_chars=400,
    )
    assert "prefers dark mode fact" in block
    assert "guild fact" in block
    assert "channel fact" not in block


@pytest.mark.asyncio
async def test_top_k_and_max_prompt_chars_are_enforced(tmp_path):
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    now = datetime.now(timezone.utc).isoformat()
    records = {}
    for idx in range(10):
        memory_id = f"mem-{idx}"
        record = MemoryRecord(
            memory_id=memory_id,
            user_id="user-1",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            source_message_id=None,
            context_type="project_fact",
            text=f"fact {idx}",
            summary=f"fact {idx} " + ("x" * 110),
            importance=0.9,
            confidence=0.95,
            created_at=now,
            updated_at=now,
            last_accessed_at=None,
            expires_at=None,
            source="inferred_curated",
            deleted_at=None,
            chroma_id=None,
            metadata_json="{}",
        )
        await store.upsert_memory(record)
        records[memory_id] = {
            "memory_id": memory_id,
            "document": record.summary,
            "metadata": {"user_id": "user-1", "guild_id": "guild-1", "channel_id": "channel-1", "context_type": "project_fact", "created_at": now, "importance": 0.9, "confidence": 0.95},
        }

    fake_semantic = FakeSemanticStore(records)
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()

    block = await svc.build_prompt_block(
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        query="fact",
        top_k=20,
        max_chars=180,
    )
    assert fake_semantic.calls[0]["top_k"] == 8
    assert len(block) <= 180
    assert block.startswith("Relevant long-term memory:")
    assert block.count("\n- ") <= 1


@pytest.mark.asyncio
async def test_queue_full_drops_inferred_memory_without_blocking():
    async def persist_callback(batch):
        await asyncio.sleep(0)

    queue = CuratedMemoryIngestionQueue(persist_callback, max_size=1, workers=1, batch_size=1)
    candidate = CuratedMemoryCurator().build_explicit_candidate(user_id="user-1", text="I prefer dark mode")
    assert candidate is not None
    assert await queue.enqueue(candidate) is True
    second = CuratedMemoryCurator().build_explicit_candidate(user_id="user-1", text="I prefer light mode")
    assert second is not None
    start = asyncio.get_event_loop().time()
    assert await queue.enqueue(second) is False
    assert asyncio.get_event_loop().time() - start < 0.2


@pytest.mark.asyncio
async def test_explicit_memory_command_rejects_internal_traces_and_secrets():
    curator = CuratedMemoryCurator()
    assert curator.build_explicit_candidate(user_id="u", text="my API key is sk-12345678901234567890") is None
    assert curator.build_explicit_candidate(user_id="u", text="tool trace: hidden reasoning") is None


@pytest.mark.asyncio
async def test_no_raw_transcript_is_injected_into_prompt(tmp_path):
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    now = datetime.now(timezone.utc).isoformat()
    raw_transcript = "USER RAW TRANSCRIPT: I said a lot of details that should not be injected."
    record = MemoryRecord(
        memory_id="mem-raw",
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        source_message_id=None,
        context_type="user_preference",
        text=raw_transcript,
        summary="prefers concise answers",
        importance=0.9,
        confidence=0.95,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        expires_at=None,
        source="explicit_memory_command",
        deleted_at=None,
        chroma_id=None,
        metadata_json="{}",
    )
    await store.upsert_memory(record)
    fake_semantic = FakeSemanticStore(
        {
            "mem-raw": {
                "memory_id": "mem-raw",
                "document": record.summary,
                "metadata": {"user_id": "user-1", "guild_id": "guild-1", "channel_id": "channel-1", "context_type": "user_preference", "created_at": now, "importance": 0.9, "confidence": 0.95},
            }
        }
    )
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()

    block = await svc.build_prompt_block(
        user_id="user-1",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id=None,
        query="concise",
        top_k=6,
        max_chars=400,
    )
    assert "prefers concise answers" in block
    assert raw_transcript not in block


@pytest.mark.asyncio
async def test_bot_message_handling_does_not_await_slow_chroma_writes(monkeypatch):
    bot = LLMBot.__new__(LLMBot)
    bot.context_manager = SimpleNamespace(append=lambda message: None)
    bot.enhanced_context_manager = AsyncMock()
    bot._message_is_command = AsyncMock(return_value=False)
    bot.router = SimpleNamespace(
        dispatch_message=AsyncMock(return_value=None),
        get_dispatch_metadata=MagicMock(return_value={}),
        pop_gate_denied_reason=MagicMock(return_value=None),
        clear_dispatch_metadata=MagicMock(),
    )
    bot.logger = MagicMock()
    bot.config = {"STREAMING_ENABLE": False}
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=999))
    bot._dispatch_lock = asyncio.Lock()
    bot._processed_messages = OrderedDict()
    bot._get_user_queue = lambda _user_id: SimpleNamespace(qsize=lambda: 0)
    bot.process_commands = AsyncMock()

    slow_started = asyncio.Event()
    slow_finished = asyncio.Event()

    async def slow_enqueue_inferred_memory(**kwargs):
        slow_started.set()
        await asyncio.sleep(0.2)
        slow_finished.set()
        return True

    created = []

    def fake_create_task(coro):
        created.append(coro)
        coro.close()
        return MagicMock()

    monkeypatch.setattr("bot.core.bot.enqueue_inferred_memory", slow_enqueue_inferred_memory)
    monkeypatch.setattr("bot.core.bot.asyncio.create_task", fake_create_task)

    message = SimpleNamespace(
        id=1,
        content="remember this preference",
        author=SimpleNamespace(id=123, bot=False),
        guild=SimpleNamespace(id=456),
        channel=SimpleNamespace(id=789),
        attachments=[],
    )

    await bot._process_single_message(message)

    assert created, "expected enqueue task to be scheduled"
    assert not slow_started.is_set()
    assert not slow_finished.is_set()


def test_recency_and_combined_scoring_decay_with_age():
    now = datetime.now(timezone.utc)
    fresh = now.isoformat()
    old = (now - timedelta(days=180)).isoformat()
    assert recency_score(fresh, now=now) > recency_score(old, now=now)
    assert combined_score(0.9, 0.8, fresh, now=now) > combined_score(0.9, 0.8, old, now=now)
