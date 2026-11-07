"""Per-user profile learning and the always-on profile prompt block."""

from __future__ import annotations

import asyncio
import json

import pytest

from bot.memory import profile_extractor as pe
from bot.memory.curator import CuratedMemoryCurator
from bot.memory.persistent_store import PersistentMemoryStore
from bot.memory.service import INSTRUCTION_HEADER, MEMORY_BLOCK_HEADER, CuratedMemoryService

USER = "111"


class FakeSemantic:
    """Vector store stand-in: stores nothing semantic, never matches a query."""

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}

    async def initialize(self) -> None:
        return None

    async def upsert(self, memory_id, document, metadata):
        self.docs[memory_id] = document
        return memory_id

    async def delete(self, memory_id):
        self.docs.pop(memory_id, None)

    async def delete_many(self, ids):
        for i in ids:
            self.docs.pop(i, None)

    async def query(self, query, top_k=6, where=None):
        return []


class FakeModel:
    def __init__(self, *replies: dict) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def __call__(self, *, prompt, system_prompt, **kw):
        self.prompts.append(prompt)
        return {"text": json.dumps(self.replies.pop(0)) if self.replies else '{"add": []}'}


@pytest.fixture
async def svc(tmp_path):
    service = CuratedMemoryService(bot=None)
    service.enabled = True
    service.store = PersistentMemoryStore(tmp_path / "m.db")
    await service.store.initialize()
    service.semantic_store = FakeSemantic()
    service.curator = CuratedMemoryCurator()
    service.max_prompt_chars = 1200
    service._started = True
    yield service
    if service.extractor is not None:
        await service.extractor.stop()


def _extractor(svc, model, **kw) -> pe.ProfileExtractor:
    svc.extractor = pe.ProfileExtractor(svc, generate=model, **kw)
    return svc.extractor


async def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


async def _profile(svc) -> dict[str, str]:
    rows = await svc.store.list_memories(user_id=USER, limit=100)
    return {r.summary: r.context_type for r in rows if r.source == pe.PROFILE_SOURCE}


def test_parse_operations_tolerates_fences_and_garbage():
    assert pe.parse_operations('```json\n{"add": [{"text": "Lives in Leeds"}]}\n```')["add"] == [{"text": "Lives in Leeds"}]
    assert pe.parse_operations("sorry, no") == {"add": [], "update": [], "remove": []}
    assert pe.parse_operations('{"add": "x", "remove": ["f1"]}') == {"add": [], "update": [], "remove": ["f1"]}


def test_clean_fact_rejects_sensitive_mentions_and_length():
    assert pe.clean_fact(" Works as a nurse in Manchester. ") == "Works as a nurse in Manchester"
    assert pe.clean_fact("Has been diagnosed with depression") is None
    assert pe.clean_fact("Is friends with <@123456>") is None
    assert pe.clean_fact("x" * 200) is None


async def test_pass_adds_global_facts_and_instructions(svc):
    model = FakeModel(
        {
            "add": [
                {"text": "Works as a nurse in Manchester", "type": "fact"},
                {"text": "Wants to be called tq", "type": "instruction"},
                {"text": "Has been diagnosed with depression", "type": "fact"},
            ]
        }
    )
    ex = _extractor(svc, model)
    ex.add_message(user_id=USER, text="i'm a nurse in manchester, call me tq", guild_id="A", message_id="m1")
    result = await ex.flush(USER)
    assert (result.added, result.rejected) == (2, 1)
    assert await _profile(svc) == {"Works as a nurse in Manchester": "user_profile", "Wants to be called tq": "recurring_instruction"}
    rows = await svc.store.list_memories(user_id=USER, limit=10)
    assert {r.guild_id for r in rows} == {None}  # one profile for every server
    assert "1. i'm a nurse in manchester, call me tq" in model.prompts[0]


async def test_pass_updates_removes_and_skips_saved_facts(svc):
    await svc.add_explicit_memory(user_id=USER, text="i support arsenal", guild_id="A")
    prompts: list[str] = []

    async def model(*, prompt, system_prompt, **kw):
        prompts.append(prompt)
        if len(prompts) == 1:
            return {"text": json.dumps({"add": [{"text": "Lives in London"}, {"text": "Plays guitar"}]})}
        ids = {line.split(": ", 1)[1]: line[2:].split(":")[0] for line in prompt.splitlines() if line.startswith("- f")}
        ops = {
            "add": [{"text": "They said: i support arsenal"}],
            "update": [{"id": ids["Lives in London"], "text": "Lives in Bristol"}],
            "remove": [ids["Plays guitar"]],
        }
        return {"text": json.dumps(ops)}

    ex = _extractor(svc, model)
    ex.add_message(user_id=USER, text="i live in london and play guitar", guild_id="A", message_id="m1")
    await ex.flush(USER)
    ex.add_message(user_id=USER, text="moved to bristol, quit guitar", guild_id="B", message_id="m2")
    result = await ex.flush(USER)
    assert "- (saved by user) They said: i support arsenal" in prompts[1]
    assert (result.added, result.updated, result.removed, result.rejected) == (0, 1, 1, 1)
    assert await _profile(svc) == {"Lives in Bristol": "user_profile"}


async def test_cap_drops_oldest_profile_facts(svc):
    model = FakeModel({"add": [{"text": f"Likes band number {i}"} for i in range(12)]})
    ex = _extractor(svc, model, max_facts=3)
    ex.add_message(user_id=USER, text="bands", guild_id=None, message_id=None)
    await ex.flush(USER)
    assert len(await _profile(svc)) == 3


async def test_model_failure_is_contained(svc):
    async def boom(**kw):
        raise RuntimeError("all providers down")

    ex = _extractor(svc, boom)
    ex.add_message(user_id=USER, text="i live in leeds", guild_id=None, message_id=None)
    assert await ex.flush(USER) is None
    assert [m.text for m in ex._buffers[USER]] == ["i live in leeds"]  # retried once
    assert await ex.flush(USER) is None
    assert not ex._buffers.get(USER)  # then dropped
    assert await _profile(svc) == {}


async def test_every_n_and_idle_trigger_a_pass(svc):
    model = FakeModel({"add": [{"text": "Lives in Leeds"}]}, {"add": [{"text": "Plays chess"}]})
    ex = _extractor(svc, model, every_n=2, idle_s=0.05)
    ex.add_message(user_id=USER, text="hello there", guild_id=None, message_id=None)
    assert model.prompts == []
    ex.add_message(user_id=USER, text="i live in leeds", guild_id=None, message_id=None)
    await _wait_for(lambda: len(model.prompts) == 1)  # every_n reached
    ex.add_message(user_id=USER, text="i play chess", guild_id=None, message_id=None)
    await _wait_for(lambda: len(model.prompts) == 2)  # idle flush
    await _wait_for(lambda: not ex._running)
    assert set(await _profile(svc)) == {"Lives in Leeds", "Plays chess"}


async def test_service_routes_messages_to_extractor_and_drops_secrets(svc):
    ex = _extractor(svc, FakeModel(), every_n=50)
    assert await svc.enqueue_inferred_memory(user_id=USER, text="i live in leeds", guild_id="A") is True
    assert await svc.enqueue_inferred_memory(user_id=USER, text="my key is sk-abcdefghijklmnopqrstuvwxyz123456", guild_id="A") is False
    assert [m.text for m in ex._buffers[USER]] == ["i live in leeds"]


async def test_prompt_block_includes_profile_from_any_server(svc):
    model = FakeModel({"add": [{"text": "Works as a nurse in Manchester"}, {"text": "Wants short answers", "type": "instruction"}]})
    ex = _extractor(svc, model)
    ex.add_message(user_id=USER, text="facts", guild_id="A", message_id=None)
    await ex.flush(USER)
    await svc.add_explicit_memory(user_id=USER, text="i have depression", guild_id="A")
    await svc.add_explicit_memory(user_id="222", text="i like jazz", guild_id="B")
    # Server B, unrelated message, no semantic match at all:
    block = await svc.build_prompt_block(user_id=USER, guild_id="B", channel_id="c", thread_id=None, query="hey whats up")
    assert MEMORY_BLOCK_HEADER in block and "- Works as a nurse in Manchester" in block
    assert INSTRUCTION_HEADER in block and "- Wants short answers" in block
    assert "depression" not in block  # never disclosed in normal chat
    assert "jazz" not in block  # other users' memories never leak
    assert await svc.build_prompt_block(user_id="999", guild_id="B", channel_id="c", thread_id=None, query="hi") == ""


async def test_instructions_survive_a_tight_char_budget(svc):
    model = FakeModel({"add": [{"text": "Wants short answers", "type": "instruction"}] + [{"text": f"Likes band number {i}"} for i in range(4)]})
    ex = _extractor(svc, model)
    ex.add_message(user_id=USER, text="facts", guild_id=None, message_id=None)
    await ex.flush(USER)
    block = await svc.build_prompt_block(user_id=USER, guild_id=None, channel_id=None, thread_id=None, query="hi", max_chars=200)
    assert "- Wants short answers" in block and len(block) <= 200


async def test_no_memory_expires_by_default(svc):
    rec = await svc.add_explicit_memory(user_id=USER, text="i support arsenal", guild_id="A")
    assert rec.expires_at is None
    ex = _extractor(svc, FakeModel({"add": [{"text": "Lives in Leeds"}]}))
    ex.add_message(user_id=USER, text="i live in leeds", guild_id=None, message_id=None)
    await ex.flush(USER)
    assert {r.expires_at for r in await svc.store.list_memories(user_id=USER, limit=10)} == {None}
    inferred = CuratedMemoryCurator().curate_inferred_candidate(user_id=USER, text="from now on call me tq please")
    assert inferred is not None and inferred.expires_at is None
