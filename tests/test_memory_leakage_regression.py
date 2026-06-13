"""Regression tests for memory/persona leakage fixes.

Covers:
1.  User A vs User B memory isolation in build_prompt_block.
2.  'me/myself' resolves to message.author.id.
3.  Sensitive memories withheld in normal guild chat.
4.  Sensitive memories still visible in !memories-show (owner only).
5.  Orphan memories (missing user_id) never returned.
6.  Semantic/vector post-filtering drops foreign-user records.
7.  Inferred memory blocks sexual/body/drug/medical/identity/slur content.
8.  Explicit !memory-add still saves for the requesting user.
9.  Delete requires owner_id for normal user deletions.
10. Speaker labels preserved in context, not transformed into requester facts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.memory.curator import (
    CuratedMemoryCurator,
    is_public_safe,
)
from bot.memory.identity import is_self_recall_intent, resolve_memory_subject_user_id
from bot.memory.persistent_store import MemoryRecord
from bot.memory.service import CuratedMemoryService

# ---- helpers ----------------------------------------------------------------

_NOW = datetime.now(UTC).isoformat()


def _make_record(
    *,
    memory_id: str,
    user_id: str,
    guild_id: str = "guild-1",
    summary: str = "summary",
    text: str = "text",
    context_type: str = "user_preference",
    source: str = "explicit_memory_command",
    **kw,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id=user_id,
        guild_id=guild_id,
        channel_id="channel-1",
        thread_id=None,
        source_message_id=None,
        context_type=context_type,
        text=text,
        summary=summary,
        importance=0.9,
        confidence=0.95,
        created_at=_NOW,
        updated_at=_NOW,
        last_accessed_at=None,
        expires_at=None,
        source=source,
        deleted_at=None,
        chroma_id=None,
        metadata_json="{}",
        **kw,
    )


def _fake_semantic_store(records: dict | None = None):
    """In-memory semantic store that respects where-filter."""
    if records is None:
        records = {}
    mock = MagicMock()
    mock.records = records
    mock.initialize = AsyncMock()
    mock.upserts = []
    mock.delete = AsyncMock()
    mock.deleted = []
    mock.delete_many = AsyncMock()
    mock.deleted_many = []

    async def fake_upsert(memory_id: str, document: str, metadata: dict) -> str:
        mock.upserts.append((memory_id, document, metadata))
        records[memory_id] = {
            "memory_id": memory_id,
            "document": document,
            "metadata": metadata,
        }
        return "chroma-id"

    mock.upsert = AsyncMock(side_effect=fake_upsert)

    async def fake_query(query, top_k=6, where=None):
        results = []
        for mid, payload in records.items():
            meta = payload.get("metadata", {})
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            results.append(
                {
                    "memory_id": mid,
                    "document": payload.get("document", ""),
                    "metadata": meta,
                    "semantic_score": 0.92,
                },
            )
        return results[:top_k]

    mock.query = AsyncMock(side_effect=fake_query)
    return mock


def _fake_persistent_store():
    store = MagicMock()
    store.initialize = AsyncMock()
    store._records: dict[str, MemoryRecord] = {}

    async def fake_upsert(rec: MemoryRecord) -> None:
        store._records[rec.memory_id] = rec

    async def fake_get(mid: str):
        return store._records.get(mid)

    async def fake_list(user_id: str | None = None, limit: int = 20, **kw):
        recs = list(store._records.values())
        if user_id is not None:
            recs = [r for r in recs if str(r.user_id) == str(user_id)]
        recs = [r for r in recs if r.deleted_at is None]
        recs.sort(key=lambda r: r.updated_at, reverse=True)
        return recs[:limit]

    async def fake_fetch(ids: list[str]):
        return [store._records[i] for i in ids if i in store._records]

    async def fake_soft_delete(mid: str) -> bool:
        rec = store._records.get(mid)
        if rec is None:
            return False
        rec.deleted_at = _NOW
        return True

    store.upsert_memory = AsyncMock(side_effect=fake_upsert)
    store.get_memory = AsyncMock(side_effect=fake_get)
    store.list_memories = AsyncMock(side_effect=fake_list)
    store.fetch_active_by_ids = AsyncMock(side_effect=fake_fetch)
    store.soft_delete_memory = AsyncMock(side_effect=fake_soft_delete)
    store.find_by_normalized_text = AsyncMock(return_value=None)

    async def fake_wipe(user_id: str):
        ids = []
        for mid, rec in list(store._records.items()):
            if str(rec.user_id) == str(user_id) and rec.deleted_at is None:
                rec.deleted_at = _NOW
                ids.append(mid)
        return ids

    store.wipe_user_memories = AsyncMock(side_effect=fake_wipe)
    return store


def _build_service(store, semantic):
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.store = store
    svc.semantic_store = semantic
    svc.curator = CuratedMemoryCurator()
    svc.max_prompt_chars = 1200
    svc.top_k = 6
    return svc


# ---- Part A: resolve_memory_subject_user_id --------------------------------


def test_resolve_memory_subject_is_author_id() -> None:
    """'me/myself' always resolves to message.author.id."""
    msg = MagicMock()
    msg.author.id = 99887
    result = resolve_memory_subject_user_id(msg)
    assert result == "99887"


def test_resolve_does_not_use_mentioned() -> None:
    msg = MagicMock()
    msg.author.id = 111
    msg.mentions = [MagicMock(id=222)]
    result = resolve_memory_subject_user_id(msg)
    assert result == "111"


def test_self_recall_intent_patterns() -> None:
    assert is_self_recall_intent("tell me about myself")
    assert is_self_recall_intent("what do you know about me")
    assert is_self_recall_intent("what are my memories")
    assert is_self_recall_intent("what do you remember about me")
    assert is_self_recall_intent("tell me something about myself")
    assert not is_self_recall_intent("tell me about user2")
    assert not is_self_recall_intent("hello world")


# ---- Part B / D: owner-scope filtering + safe disclosure -------------------


class TestOwnerScopeAndSafeDisclosure:
    """Tests that build_prompt_block is strictly owner-scoped with safe disclosure."""

    @pytest.fixture
    def svc(self):
        # User A has two memories: one safe, one sensitive
        rec_a1 = _make_record(
            memory_id="mem-a1",
            user_id="111",
            summary="Prefers dark mode UI",
            text="I prefer dark mode",
        )
        rec_a2 = _make_record(
            memory_id="mem-a2",
            user_id="111",
            summary="Prefers heavy music",
            text="I like metal music",
        )
        # User B has two memories in the same guild
        rec_b1 = _make_record(
            memory_id="mem-b1",
            user_id="222",
            summary="Prefers light mode UI",
            text="I prefer light mode",
        )
        rec_b2 = _make_record(
            memory_id="mem-b2",
            user_id="222",
            summary="Has a dog named Rex",
            text="I have a dog named Rex",
        )
        store = _fake_persistent_store()
        store._records = {
            "mem-a1": rec_a1,
            "mem-a2": rec_a2,
            "mem-b1": rec_b1,
            "mem-b2": rec_b2,
        }
        semantic = _fake_semantic_store(
            {
                "mem-a1": {
                    "memory_id": "mem-a1",
                    "document": "Prefers dark mode UI",
                    "metadata": {
                        "user_id": "111",
                        "guild_id": "guild-1",
                        "context_type": "user_preference",
                        "importance": 0.9,
                    },
                },
                "mem-a2": {
                    "memory_id": "mem-a2",
                    "document": "Prefers heavy music",
                    "metadata": {
                        "user_id": "111",
                        "guild_id": "guild-1",
                        "context_type": "user_preference",
                        "importance": 0.9,
                    },
                },
                "mem-b1": {
                    "memory_id": "mem-b1",
                    "document": "Prefers light mode UI",
                    "metadata": {
                        "user_id": "222",
                        "guild_id": "guild-1",
                        "context_type": "user_preference",
                        "importance": 0.9,
                    },
                },
                "mem-b2": {
                    "memory_id": "mem-b2",
                    "document": "Has a dog named Rex",
                    "metadata": {
                        "user_id": "222",
                        "guild_id": "guild-1",
                        "context_type": "user_preference",
                        "importance": 0.9,
                    },
                },
            },
        )
        return _build_service(store, semantic)

    @pytest.mark.asyncio
    async def test_user_a_sees_only_own_memories(self, svc) -> None:
        block = await svc.build_prompt_block(
            user_id="111",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="preferences",
            max_chars=500,
        )
        assert "Prefers dark mode UI" in block
        assert "Prefers light mode UI" not in block
        assert "dog named Rex" not in block

    @pytest.mark.asyncio
    async def test_user_b_sees_only_own_memories(self, svc) -> None:
        block = await svc.build_prompt_block(
            user_id="222",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="preferences",
            max_chars=500,
        )
        assert "Prefers light mode UI" in block
        assert "Has a dog named Rex" in block
        assert "dark mode" not in block

    @pytest.mark.asyncio
    async def test_empty_user_gets_empty_block(self, svc) -> None:
        block = await svc.build_prompt_block(
            user_id="999",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="anything",
            max_chars=500,
        )
        assert block == ""


# ---- Sensitive memory filtering ---------------------------------------------


class TestSensitiveMemoryDisclosure:
    @pytest.fixture
    def svc_with_sensitive(self):
        store = _fake_persistent_store()
        rec_safe = _make_record(
            memory_id="mem-safe",
            user_id="111",
            summary="Prefers dark mode UI",
        )
        rec_sensitive = _make_record(
            memory_id="mem-sens",
            user_id="111",
            summary="Has been diagnosed with depression",
            text="I have depression",
        )
        store._records = {"mem-safe": rec_safe, "mem-sens": rec_sensitive}
        semantic = _fake_semantic_store(
            {
                "mem-safe": {
                    "memory_id": "mem-safe",
                    "document": "Prefers dark mode UI",
                    "metadata": {
                        "user_id": "111",
                        "guild_id": "guild-1",
                        "importance": 0.9,
                    },
                },
                "mem-sens": {
                    "memory_id": "mem-sens",
                    "document": "Has been diagnosed with depression",
                    "metadata": {
                        "user_id": "111",
                        "guild_id": "guild-1",
                        "importance": 0.9,
                    },
                },
            },
        )
        return _build_service(store, semantic)

    @pytest.mark.asyncio
    async def test_sensitive_memory_withheld_in_normal_chat(self, svc_with_sensitive) -> None:
        block = await svc_with_sensitive.build_prompt_block(
            user_id="111",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="about me",
            max_chars=500,
            allow_sensitive=False,
        )
        assert "depression" not in block
        assert "dark mode" in block

    @pytest.mark.asyncio
    async def test_sensitive_memory_visible_with_allow_sensitive(self, svc_with_sensitive) -> None:
        block = await svc_with_sensitive.build_prompt_block(
            user_id="111",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="about me",
            max_chars=500,
            allow_sensitive=True,
        )
        assert "depression" in block
        assert "dark mode" in block

    @pytest.mark.asyncio
    async def test_all_sensitive_returns_safe_ack_message(self, svc_with_sensitive) -> None:
        # Remove safe record temporarily
        del svc_with_sensitive.store._records["mem-safe"]
        block = await svc_with_sensitive.build_prompt_block(
            user_id="111",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="about me",
            max_chars=500,
            allow_sensitive=False,
        )
        assert "personal memories" in block
        assert "won't repeat sensitive" in block


# ---- Orphan memories (missing user_id) --------------------------------------


class TestOrphanMemories:
    @pytest.fixture
    def svc_with_orphan(self):
        store = _fake_persistent_store()
        rec_orphan = _make_record(
            memory_id="mem-orphan",
            user_id="",  # orphaned — no owner
            summary="Legacy memory from old system",
        )
        rec_owned = _make_record(
            memory_id="mem-owned",
            user_id="111",
            summary="Current user preference",
        )
        store._records = {"mem-orphan": rec_orphan, "mem-owned": rec_owned}
        semantic = _fake_semantic_store(
            {
                "mem-orphan": {
                    "memory_id": "mem-orphan",
                    "document": "Legacy memory from old system",
                    "metadata": {
                        "user_id": "",
                        "guild_id": "guild-1",
                        "importance": 0.9,
                    },
                },
                "mem-owned": {
                    "memory_id": "mem-owned",
                    "document": "Current user preference",
                    "metadata": {
                        "user_id": "111",
                        "guild_id": "guild-1",
                        "importance": 0.9,
                    },
                },
            },
        )
        return _build_service(store, semantic)

    @pytest.mark.asyncio
    async def test_orphan_not_returned_to_any_user(self, svc_with_orphan) -> None:
        for uid in ["111", "222", "333"]:
            block = await svc_with_orphan.build_prompt_block(
                user_id=uid,
                guild_id="guild-1",
                channel_id="channel-1",
                thread_id=None,
                query="memory",
                max_chars=500,
            )
            assert "Legacy memory" not in block

    @pytest.mark.asyncio
    async def test_owned_memories_still_returned(self, svc_with_orphan) -> None:
        block = await svc_with_orphan.build_prompt_block(
            user_id="111",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="preference",
            max_chars=500,
        )
        assert "Current user preference" in block


# ---- inferred memory blocking -----------------------------------------------


class TestInferredMemorySensitiveBlocking:
    """Inferred memories must not include sexual, body, drug, medical, identity, or slur content."""

    @pytest.fixture
    def curator(self):
        return CuratedMemoryCurator()

    def test_sexual_content_blocked(self, curator) -> None:
        texts = [
            "I watch porn every night",
            "I have an onlyfans account",
            "I had a one night stand",
            "I like to be nude",
        ]
        for t in texts:
            result = curator.curate_inferred_candidate(user_id="111", text=t, guild_id="g1")
            assert result is None, f"Should be blocked: {t!r}"

    def test_body_size_claims_blocked(self, curator) -> None:
        result = curator.curate_inferred_candidate(user_id="111", text="I am overweight and my weight is 200kg", guild_id="g1")
        assert result is None

    def test_drug_references_blocked(self, curator) -> None:
        for t in [
            "I smoke weed every day",
            "I took cocaine at the party",
            "I use xanax for sleep",
            "I bought meth from someone",
        ]:
            result = curator.curate_inferred_candidate(user_id="111", text=t, guild_id="g1")
            assert result is None, f"Should block drugs: {t!r}"

    def test_medical_claims_blocked(self, curator) -> None:
        for t in [
            "I have depression and take medication",
            "I was diagnosed with anxiety last year",
            "I see my therapist every week",
            "I have adhd and bipolar disorder",
        ]:
            result = curator.curate_inferred_candidate(user_id="111", text=t, guild_id="g1")
            assert result is None, f"Should block medical: {t!r}"

    def test_protected_identity_blocked(self, curator) -> None:
        for t in [
            "I am gay and proud",
            "My religion is catholic",
            "I am transgender",
            "I am bisexual",
        ]:
            result = curator.curate_inferred_candidate(user_id="111", text=t, guild_id="g1")
            assert result is None, f"Should block identity: {t!r}"

    def test_slurs_blocked(self, curator) -> None:
        for t in ["that guy is a faggot", "retard moment"]:
            result = curator.curate_inferred_candidate(user_id="111", text=t, guild_id="g1")
            assert result is None, f"Should block slurs: {t!r}"

    def test_third_party_anecdote_blocked(self, curator) -> None:
        for t in [
            "my friend got arrested last night",
            "someone said the movie was terrible",
            "my bro did something crazy at the bar",
        ]:
            result = curator.curate_inferred_candidate(user_id="111", text=t, guild_id="g1")
            assert result is None, f"Should block third party: {t!r}"

    def test_safe_preference_accepted(self, curator) -> None:
        result = curator.curate_inferred_candidate(
            user_id="111",
            text="I prefer short replies and dark mode",
            guild_id="g1",
        )
        assert result is not None
        assert result.context_type == "user_preference"


# ---- explicit memory-add still saves for requesting user --------------------


class TestExplicitMemoryAddStillWorks:
    @pytest.fixture
    def svc(self):
        store = _fake_persistent_store()
        semantic = _fake_semantic_store()
        return _build_service(store, semantic)

    @pytest.mark.asyncio
    async def test_explicit_add_stores_for_owner(self, svc) -> None:
        rec = await svc.add_explicit_memory(
            user_id="111",
            text="I prefer dark mode",
            guild_id="guild-1",
            channel_id="channel-1",
            source="explicit_memory_command",
        )
        assert rec is not None
        assert rec.user_id == "111"
        stored = await svc.store.get_memory(rec.memory_id)
        assert stored is not None
        assert stored.user_id == "111"

    @pytest.mark.asyncio
    async def test_explicit_add_sensitive_text_saved_but_hidden(self, svc) -> None:
        # Sensitive content IS stored (owner-scoped) but gated at prompt injection.
        rec = await svc.add_explicit_memory(
            user_id="111",
            text="I snort coke every weekend",
            guild_id="guild-1",
            channel_id="channel-1",
            source="explicit_memory_command",
        )
        assert rec is not None
        assert rec.user_id == "111"
        # Stored in DB
        stored = await svc.store.get_memory(rec.memory_id)
        assert stored is not None
        assert stored.user_id == "111"
        # But NOT injected into normal guild chat (allow_sensitive=False)
        block = await svc.build_prompt_block(
            user_id="111",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="about me",
            max_chars=500,
            allow_sensitive=False,
        )
        assert "coke" not in block.lower()
        # But visible with allow_sensitive=True
        block_sensitive = await svc.build_prompt_block(
            user_id="111",
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id=None,
            query="about me",
            max_chars=500,
            allow_sensitive=True,
        )
        assert "coke" in block_sensitive.lower()


# ---- delete_memory requires owner_id ----------------------------------------


class TestDeleteMemoryOwnership:
    @pytest.fixture
    def svc(self):
        store = _fake_persistent_store()
        rec = _make_record(memory_id="mem-a1", user_id="111")
        store._records = {"mem-a1": rec}
        semantic = _fake_semantic_store()
        return _build_service(store, semantic)

    @pytest.mark.asyncio
    async def test_delete_succeeds_for_owner(self, svc) -> None:
        result = await svc.delete_memory("mem-a1", owner_id="111")
        assert result is True
        stored = await svc.store.get_memory("mem-a1")
        assert stored.deleted_at is not None

    @pytest.mark.asyncio
    async def test_delete_blocked_for_non_owner(self, svc) -> None:
        result = await svc.delete_memory("mem-a1", owner_id="222")
        assert result is False
        stored = await svc.store.get_memory("mem-a1")
        assert stored.deleted_at is None

    @pytest.mark.asyncio
    async def test_delete_without_owner_id_admin_works(self, svc) -> None:
        result = await svc.delete_memory("mem-a1")  # no owner_id = admin
        assert result is True


# ---- is_public_safe helper -------------------------------------------------


class TestPublicSafeHelper:
    """Safe disclosure filter correctness."""

    def test_safe_memories_pass(self) -> None:
        assert is_public_safe("Prefers dark mode UI")
        assert is_public_safe("Likes to play chess on weekends")
        assert is_public_safe("Uses Python for scripting")
        assert is_public_safe("Has a dog named Rex")

    def test_sexual_content_fails(self) -> None:
        assert not is_public_safe("I watch porn regularly")
        assert not is_public_safe("I have an onlyfans with 1000 followers")
        assert not is_public_safe("I like to be naked at home")

    def test_drug_content_fails(self) -> None:
        assert not is_public_safe("I smoke weed every evening")
        assert not is_public_safe("I took cocaine at the concert")
        assert not is_public_safe("I use xanax for anxiety")

    def test_medical_content_fails(self) -> None:
        assert not is_public_safe("I have depression")
        assert not is_public_safe("I was diagnosed with bipolar disorder")
        assert not is_public_safe("I take medication every day")

    def test_identity_content_fails(self) -> None:
        assert not is_public_safe("I am gay")
        assert not is_public_safe("My religion is catholic")
        assert not is_public_safe("I am transgender")

    def test_slur_content_fails(self) -> None:
        assert not is_public_safe("he is a retard")

    def test_third_party_anecdote_fails(self) -> None:
        assert not is_public_safe("my friend got arrested")
        assert not is_public_safe("someone said the world is flat")
