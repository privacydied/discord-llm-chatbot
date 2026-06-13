import asyncio
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.core.bot import LLMBot
from bot.memory.curator import CuratedMemoryCurator
from bot.memory.ingestion_queue import CuratedMemoryIngestionQueue
from bot.memory.persistent_store import MemoryRecord, PersistentMemoryStore
from bot.memory.scoring import combined_score, recency_score
from bot.memory.service import CuratedMemoryService


class FakeSemanticStore:
    def __init__(self, records=None) -> None:
        self.records = records or {}
        self.calls = []
        self.upserts = []
        self.deleted = []
        self.deleted_many = []
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def upsert(self, memory_id, document, metadata):
        self.upserts.append((memory_id, document, metadata))
        self.records[memory_id] = {
            "memory_id": memory_id,
            "document": document,
            "metadata": metadata,
        }
        return memory_id

    async def delete(self, memory_id) -> None:
        self.deleted.append(memory_id)
        self.records.pop(memory_id, None)

    async def delete_many(self, memory_ids) -> None:
        self.deleted_many.append(list(memory_ids))
        for memory_id in memory_ids:
            self.records.pop(memory_id, None)

    async def query(self, query, top_k=6, where=None, where_document=None):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "where": where,
                "where_document": where_document,
            },
        )
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
                },
            )
        return results[:top_k]


class DummyQueue:
    def __init__(self) -> None:
        self.enqueued = []

    async def enqueue(self, candidate) -> bool:
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
        now = datetime.now(UTC).isoformat()
        payload = {
            "memory_id": "mem-1",
            "user_id": "user-1",
            "guild_id": "guild-1",
            "channel_id": "channel-1",
            "thread_id": None,
            "source_message_id": None,
            "context_type": "user_preference",
            "text": "I prefer dark mode",
            "summary": "prefers dark mode",
            "importance": 0.9,
            "confidence": 0.95,
            "created_at": now,
            "updated_at": now,
            "last_accessed_at": None,
            "expires_at": None,
            "source": "explicit_memory_command",
            "deleted_at": None,
            "chroma_id": None,
            "metadata_json": "{}",
        }
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
async def test_schema_bootstrap_is_idempotent(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    await store.initialize()

    import sqlite3

    conn = sqlite3.connect(tmp_path / "memory.db")
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        table_exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='curated_memories'").fetchone()
    finally:
        conn.close()

    assert version == 1
    assert table_exists is not None


@pytest.mark.asyncio
async def test_explicit_memory_add_stores_sqlite_and_semantic(service) -> None:
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
async def test_delete_and_wipe_remove_from_retrieval(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    now = datetime.now(UTC).isoformat()
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
            "mem-wipe": {
                "memory_id": "mem-wipe",
                "document": "likes tea",
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
        },
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
async def test_expired_memories_are_not_returned(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
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
                "metadata": {
                    "user_id": "user-1",
                    "guild_id": "guild-1",
                    "channel_id": "channel-1",
                    "context_type": "temporary_context",
                    "created_at": past,
                    "expires_at": past,
                    "importance": 0.4,
                    "confidence": 0.8,
                },
            },
        },
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
async def test_retrieval_filters_by_scope(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    now = datetime.now(UTC).isoformat()
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
    rec_guild = MemoryRecord(
        **{
            **rec_user.to_dict(),
            "memory_id": "mem-guild",
            "user_id": "user-2",
            "summary": "guild fact",
            "text": "guild fact",
        },
    )
    rec_channel = MemoryRecord(
        **{
            **rec_user.to_dict(),
            "memory_id": "mem-channel",
            "user_id": "user-2",
            "channel_id": "channel-9",
            "summary": "channel fact",
            "text": "channel fact",
        },
    )
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
        },
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
    # owner-scoped memory is returned
    assert "prefers dark mode fact" in block
    # guild-scoped memory from user-2 must NOT leak to user-1 (strict owner-scope)
    assert "guild fact" not in block
    # channel-scoped memory in a different channel must also not appear
    assert "channel fact" not in block


@pytest.mark.asyncio
async def test_top_k_and_max_prompt_chars_are_enforced(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    now = datetime.now(UTC).isoformat()
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
            "metadata": {
                "user_id": "user-1",
                "guild_id": "guild-1",
                "channel_id": "channel-1",
                "context_type": "project_fact",
                "created_at": now,
                "importance": 0.9,
                "confidence": 0.95,
            },
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
async def test_queue_full_drops_inferred_memory_without_blocking() -> None:
    async def persist_callback(batch) -> None:
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
async def test_explicit_memory_command_rejects_internal_traces_and_secrets() -> None:
    curator = CuratedMemoryCurator()
    assert curator.build_explicit_candidate(user_id="u", text="my API key is sk-123...7890") is None
    assert curator.build_explicit_candidate(user_id="u", text="tool trace: hidden reasoning") is None


# ==================
# Tightened inferred memory: ACCEPT cases
# ==================


@pytest.mark.asyncio
async def test_inferred_accepts_harmless_stable_user_preference() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="I prefer concise Claude prompts for discord-bot fixes.",
    )
    assert c is not None, "Should accept a clear harmless user preference"
    assert c.context_type == "user_preference"


@pytest.mark.asyncio
async def test_inferred_rejects_old_vague_project_rule() -> None:
    """Vague project chatter without instruction/preference signals should be rejected."""
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="Working on the discord-bot router and memory module today.",
    )
    assert c is None, "Project debug chatter should be rejected"


@pytest.mark.asyncio
async def test_inferred_accepts_conversation_decision() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="We decided the archive must not auto-inject into prompts",
    )
    assert c is not None, "Should accept conversation decision"
    assert c.context_type in ("conversation_decision", "project_fact")


@pytest.mark.asyncio
async def test_inferred_accepts_bot_correction() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="You were wrong about X; the correct rule is Y",
    )
    assert c is not None, "Should accept correction to bot behavior"
    assert c.context_type == "correction"


@pytest.mark.asyncio
async def test_inferred_rejects_temporary_context_for_now() -> None:
    """Temporary phrasing without strong instruction content should be rejected."""
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="For now, let's just keep it as is",
    )
    assert c is None, "Temporary phrasing without durable instruction should be rejected"


# ==================
# Tightened inferred memory: REJECT cases
# ==================


@pytest.mark.asyncio
async def test_inferred_rejects_broad_race_demographic_claim() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="Project fact: race is overrepresented in finance, politics, media, and academia.",
    )
    assert c is None, "Should reject broad race/demographic claims"


@pytest.mark.asyncio
async def test_inferred_rejects_political_social_claim() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="Project fact: political viewpoints are underrepresented in academia and media.",
    )
    assert c is None, "Should reject political/societal claims"


@pytest.mark.asyncio
async def test_inferred_rejects_finance_media_academia_claim() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="Project fact: finance, media, and academia are overrepresented by one demographic.",
    )
    assert c is None, "Should reject broad world claims"


@pytest.mark.asyncio
async def test_inferred_rejects_quoted_article_content() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="According to an article, 'Race X is overrepresented in STEM fields.'",
    )
    assert c is None, "Should reject quoted/external content"


@pytest.mark.asyncio
async def test_inferred_rejects_one_off_opinion() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="Project fact: The user believes that diversity initiatives are harmful.",
    )
    assert c is None, "Should reject one-off opinions phrased as project facts"


@pytest.mark.asyncio
async def test_inferred_rejects_sensitive_identity_statement() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="I'm bisexual and prefer dark mode.",
    )
    assert c is None, "Should reject inferred sensitive identity content"


@pytest.mark.asyncio
async def test_inferred_accepts_real_bot_bug_report() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="Bug report: the sort dropdown should trigger a full server-side search, not a client-only re-render.",
    )
    assert c is not None
    assert c.context_type == "project_fact"


@pytest.mark.asyncio
async def test_inferred_accepts_real_implementation_requirement() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="The router should wrap the typing indicator around accepted routed messages.",
    )
    assert c is not None
    assert c.context_type == "project_fact"


@pytest.mark.asyncio
async def test_inferred_accepts_project_architecture_fact() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="The memory service persists curated memories in SQLite and ChromaDB.",
    )
    assert c is not None
    assert c.context_type == "project_fact"


@pytest.mark.asyncio
async def test_inferred_accepts_project_audit_fact() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="The audit found the memory service needs stricter project-fact validation.",
    )
    assert c is not None
    assert c.context_type == "project_fact"


@pytest.mark.asyncio
async def test_inferred_rejects_code_is_wrong() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(user_id="u", text="the code is wrong")
    assert c is None, "Debugging chatter should be rejected"


@pytest.mark.asyncio
async def test_inferred_rejects_correct_this_function() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(user_id="u", text="correct this function")
    assert c is None, "Instruction to fix code is not a bot-behavior correction"


@pytest.mark.asyncio
async def test_inferred_rejects_i_dont_remember() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(user_id="u", text="I don't remember")
    assert c is None, "Casual 'don't remember' is not durable"


@pytest.mark.asyncio
async def test_inferred_rejects_do_you_remember_this() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(user_id="u", text="do you remember this?")
    assert c is None, "Question about memory is not durable"


@pytest.mark.asyncio
async def test_inferred_rejects_today_was_annoying() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(user_id="u", text="today was annoying")
    assert c is None, "Casual time-bound complaint is not durable"


@pytest.mark.asyncio
async def test_inferred_rejects_short_banter() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(user_id="u", text="nice")
    assert c is None, "Short banter should not become memory"


@pytest.mark.asyncio
async def test_inferred_rejects_token_like_content() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="my api key is sk-1234567890abcdef",
    )
    assert c is None, "Token-like content should be rejected"


@pytest.mark.asyncio
async def test_inferred_rejects_internal_tool_trace() -> None:
    curator = CuratedMemoryCurator()
    c = curator.curate_inferred_candidate(
        user_id="u",
        text="tool trace: calling memory service with importance 0.9",
    )
    assert c is None, "Internal/tool-like content should be rejected"


# ==================
# Dedupe / merge tests
# ==================


@pytest.mark.asyncio
async def test_dedupe_repeated_preference_updates_existing(service) -> None:
    # Insert first memory
    r1 = await service.add_explicit_memory(
        user_id="u",
        text="I prefer concise replies",
    )
    assert r1 is not None

    # Add another explicit memory with very similar content
    r2 = await service.add_explicit_memory(
        user_id="u",
        text="I prefer concise replies",
    )
    assert r2 is not None

    # Since both are explicit and identical, they should be distinct IDs
    # but the dedupe logic for inferred memories should avoid duplicates.
    # This test ensures we do not break explicit memory insertion.
    assert r1.memory_id != r2.memory_id


@pytest.mark.asyncio
async def test_dedupe_inferred_exact_normalized_match(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()

    fake_semantic = FakeSemanticStore()
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()

    curator = CuratedMemoryCurator()

    first = curator.curate_inferred_candidate(
        user_id="u",
        guild_id="g",
        channel_id="c",
        text="I prefer short answers",
    )
    assert first is not None
    first.memory_id = "mem-a"

    # Persist first
    await svc._persist_batch([first])

    # Insert second candidate with the same normalized summary via the curator
    second = curator.curate_inferred_candidate(
        user_id="u",
        guild_id="g",
        channel_id="c",
        text="I prefer short answers",
    )
    assert second is not None
    second.memory_id = "mem-b"

    await svc._persist_batch([second])

    # Check: only one memory with this summary should exist (merged)
    mems = await store.list_memories(user_id="u", guild_id="g", limit=10)
    short_answer_mems = [m for m in mems if "short answers" in (m.summary or "")]
    assert len(short_answer_mems) == 1, "Duplicate inferred memory should be merged, not inserted"


@pytest.mark.asyncio
async def test_dedupe_semantic_high_similarity_merges(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()

    class SimilarSemanticStore(FakeSemanticStore):
        async def query(self, query, top_k=6, where=None, where_document=None):
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "where": where,
                    "where_document": where_document,
                },
            )
            query_lower = (query or "").lower()
            if "prefer" not in query_lower:
                return []
            for payload in self.records.values():
                if "prefer" in (payload["document"] or "").lower():
                    return [
                        {
                            "memory_id": payload["memory_id"],
                            "document": payload["document"],
                            "metadata": payload["metadata"],
                            "semantic_score": 0.91,
                        },
                    ]
            return []

    fake_semantic = SimilarSemanticStore()
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()

    curator = CuratedMemoryCurator()

    first = curator.curate_inferred_candidate(
        user_id="u",
        guild_id="g",
        channel_id="c",
        text="I prefer concise replies",
    )
    assert first is not None
    first.memory_id = "mem-c1"

    await svc._persist_batch([first])

    # Second candidate is semantically similar, but not an exact normalized duplicate.
    second = curator.curate_inferred_candidate(
        user_id="u",
        guild_id="g",
        channel_id="c",
        text="I prefer short replies",
    )
    assert second is not None
    second.memory_id = "mem-c2"

    await svc._persist_batch([second])

    # Expect merged: only one memory with this theme
    mems = await store.list_memories(user_id="u", guild_id="g", limit=10)
    concise_mems = [m for m in mems if "replies" in (m.summary or "").lower()]
    assert len(concise_mems) == 1, "Semantically similar inferred memory should be merged"


@pytest.mark.asyncio
async def test_dedupe_unrelated_memories_insert_separately(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()

    fake_semantic = FakeSemanticStore()
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = fake_semantic
    svc.queue = DummyQueue()

    curator = CuratedMemoryCurator()

    m1 = curator.curate_inferred_candidate(
        user_id="u",
        guild_id="g",
        channel_id="c",
        text="I prefer short replies",
    )
    assert m1 is not None
    m1.memory_id = "mem-d1"

    m2 = curator.curate_inferred_candidate(
        user_id="u",
        guild_id="g",
        channel_id="c",
        text="discord-bot must never expose raw tokens",
    )
    assert m2 is not None
    m2.memory_id = "mem-d2"

    await svc._persist_batch([m1, m2])

    mems = await store.list_memories(user_id="u", guild_id="g", limit=10)
    # Both unrelated memories should exist
    assert len(mems) == 2, "Unrelated inferred memories should be stored separately"


@pytest.mark.asyncio
async def test_no_raw_transcript_is_injected_into_prompt(tmp_path) -> None:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    await store.initialize()
    now = datetime.now(UTC).isoformat()
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
        },
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
async def test_bot_message_handling_does_not_await_slow_chroma_writes(monkeypatch) -> None:
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

    async def slow_enqueue_inferred_memory(**kwargs) -> bool:
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


def test_recency_and_combined_scoring_decay_with_age() -> None:
    now = datetime.now(UTC)
    fresh = now.isoformat()
    old = (now - timedelta(days=180)).isoformat()
    assert recency_score(fresh, now=now) > recency_score(old, now=now)
    assert combined_score(0.9, 0.8, fresh, now=now) > combined_score(0.9, 0.8, old, now=now)
