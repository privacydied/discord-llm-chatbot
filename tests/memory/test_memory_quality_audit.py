"""Regressions from the live memory audit: junk ingestion and irrelevant recall.

Every rejected text below is (a trimmed copy of) a row that was actually
stored in data/memory.db and then surfaced in unrelated replies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.memory.archive_distiller import MemoryArchiveDistiller
from bot.memory.curator import CuratedMemoryCurator, strip_user_mentions
from bot.memory.persistent_store import MemoryRecord
from bot.memory.semantic_store import cosine_from_l2
from bot.memory.service import MEMORY_BLOCK_HEADER, CuratedMemoryService, render_memory_block

BOT_ID = 1111


@pytest.fixture
def curator() -> CuratedMemoryCurator:
    return CuratedMemoryCurator(default_ttl_days=180, temp_ttl_days=14, min_importance=0.55)


@pytest.mark.parametrize(
    "text",
    [
        # generic "you always/never" chat, formerly recurring_instruction
        "im always right and you always agree with me lol",
        "you never really realize how old the court guys are until you play them",
        # motion "going forward", formerly recurring_instruction
        "even in quake you're not strafing to the side as much as going forward or backward",
        # "never"/"don't" not aimed at the bot
        "never been to london but i want to go",
        # a question is not a fact about the asker
        "from now on can you tell me what was his relationship w guenon?",
        "what was bartoks stance on hungary-romania relationship",
        "could he theoretically just delete your memory md or whatever",
        # about another member, stored under the speaker
        "theres a guy in this server his name is <@356665874041536512> always mention him in ur reply",
        # pasted bot/agent output
        "```skill loaded. from now on checked the actual state of /volume1/py/killabees```",
        "> going forward, always use the new config loader",
        # long analysis/rant
        "from now on " + "this is a very long pasted analysis of the memory table " * 6,
    ],
)
def test_inferred_junk_is_rejected(curator, text) -> None:
    assert curator.curate_inferred_candidate(user_id="1", text=text) is None


@pytest.mark.parametrize(
    "text",
    [
        "you should always reply in english to me",
        "you must never use that emoji towards me again",
        "going forward, keep replies to one paragraph",
        "from now on call me pry",
        "never call me dude again",
        "dont ping me when you reply",
        "can you always reply in english from now on",
    ],
)
def test_real_instructions_still_stored(curator, text) -> None:
    cand = curator.curate_inferred_candidate(user_id="1", text=text)
    assert cand is not None
    assert cand.context_type == "recurring_instruction"


def test_strip_user_mentions_only_removes_given_user() -> None:
    text = f"<@{BOT_ID}> <@!{BOT_ID}> from now on ping <@42>"
    assert strip_user_mentions(text, BOT_ID) == "from now on ping <@42>"
    assert strip_user_mentions(text, None) == text
    assert strip_user_mentions("", BOT_ID) == ""


def test_bot_mention_does_not_block_instruction(curator) -> None:
    text = strip_user_mentions(f"<@{BOT_ID}> you should always reply in english", BOT_ID)
    assert curator.curate_inferred_candidate(user_id="1", text=text) is not None


def test_distiller_checks_raw_content_not_clean_content() -> None:
    distiller = MemoryArchiveDistiller(bot=MagicMock(user=MagicMock(id=BOT_ID)))
    msg = {
        "author_id": "1",
        "author_bot": 0,
        # clean_content hides the mention as "@bob"; raw content keeps <@42>.
        "content": "from now on always say hi to <@42> in every reply",
        "clean_content": "from now on always say hi to @bob in every reply",
    }
    (kept,) = distiller._distill_window([msg])
    assert "<@42>" in kept["distilled_text"]
    assert distiller._curate_candidate(kept) is None


@pytest.mark.parametrize(("distance", "expected"), [(0.0, 1.0), (2.0, 0.0), (1.0, 0.5), (4.0, 0.0), (None, 0.0)])
def test_cosine_from_l2(distance, expected) -> None:
    assert cosine_from_l2(distance) == pytest.approx(expected)


def _memory_record(memory_id: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id="1",
        guild_id="g",
        channel_id=None,
        thread_id=None,
        source_message_id=None,
        context_type="user_preference",
        text=f"fact {memory_id}",
        summary=f"fact {memory_id}",
        importance=0.9,
        confidence=0.95,
        created_at="2026-10-01T00:00:00+00:00",
        updated_at="2026-10-01T00:00:00+00:00",
        last_accessed_at=None,
        expires_at=None,
        source="inferred_curated",
        deleted_at=None,
        chroma_id=None,
        metadata_json="{}",
    )


def _record(memory_id: str, summary: str) -> dict:
    return {"memory_id": memory_id, "summary": summary, "text": summary}


def test_render_memory_block_dedupes_and_frames_as_background() -> None:
    items = [{"record": _record("a", "Prefers dark mode")}, {"record": _record("b", "prefers dark mode")}]
    block = render_memory_block(items, max_chars=1000)
    assert block == f"{MEMORY_BLOCK_HEADER}\n- Prefers dark mode"
    assert render_memory_block([], max_chars=1000) == ""


async def test_low_relevance_hits_are_dropped_and_scope_is_owner_filtered() -> None:
    svc = CuratedMemoryService(bot=None)
    svc.enabled = True
    svc.min_relevance = 0.25
    hits = [
        {"memory_id": "near", "document": "d", "metadata": {"user_id": "1", "guild_id": "g"}, "semantic_score": 0.6},
        {"memory_id": "far", "document": "d", "metadata": {"user_id": "1", "guild_id": "g"}, "semantic_score": 0.15},
    ]
    svc.semantic_store = MagicMock(query=AsyncMock(return_value=hits))
    svc.store = MagicMock(
        initialize=AsyncMock(),
        fetch_active_by_ids=AsyncMock(side_effect=lambda ids: [_memory_record(i) for i in ids]),
    )
    out = await svc.semantic_search("yo", user_id="1", guild_id="g", channel_id=None, thread_id=None)
    assert [item["memory_id"] for item in out] == ["near"]
    wheres = [call.kwargs.get("where") for call in svc.semantic_store.query.await_args_list]
    assert {"$and": [{"guild_id": "g"}, {"user_id": "1"}]} in wheres
