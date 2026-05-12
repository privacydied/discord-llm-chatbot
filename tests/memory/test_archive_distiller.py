from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from bot.memory.archive_distiller import MemoryArchiveDistiller
from bot.memory.service import CuratedMemoryService
import bot.memory.service as memory_service_module
from bot.server_archive.models import (
    ArchiveChannel,
    ArchiveGuild,
    ArchiveMessage,
    ArchiveMessageBundle,
    ArchiveThread,
    ArchiveUser,
)
import bot.server_archive.service as archive_service_module
from bot.server_archive.service import ServerArchiveService


class FakeSemanticStore:
    async def initialize(self) -> None:
        return None

    async def upsert(self, memory_id, summary, metadata):
        return memory_id

    async def query(self, query_text, top_k=3, where=None):
        return []

    async def search(self, query_text, top_k=3, where=None):
        return []

    async def delete(self, memory_id):
        return None

    async def delete_many(self, ids):
        return None


@pytest_asyncio.fixture
async def distiller_env(tmp_path, monkeypatch):
    original_archive_service = archive_service_module._service
    original_memory_service = memory_service_module._memory_service
    archive_db = tmp_path / "archive.db"
    memory_db = tmp_path / "memory.db"
    chroma_dir = tmp_path / "chroma"

    archive_cfg_data = {
        "SERVER_ARCHIVE_ENABLED": True,
        "SERVER_ARCHIVE_DB_PATH": str(archive_db),
        "SERVER_ARCHIVE_SYNC_ON_START": False,
        "SERVER_ARCHIVE_LIVE_TAIL": False,
        "SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES": False,
        "SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES": False,
        "PERSISTENT_MEMORY_ENABLE": True,
    }

    memory_cfg_data = {
        "PERSISTENT_MEMORY_ENABLE": True,
        "PERSISTENT_MEMORY_SQLITE_PATH": str(memory_db),
        "PERSISTENT_MEMORY_CHROMA_PATH": str(chroma_dir),
        "PERSISTENT_MEMORY_CHROMA_COLLECTION": "curated_memories",
        "PERSISTENT_MEMORY_DEFAULT_TTL_DAYS": 180,
        "PERSISTENT_MEMORY_TEMP_TTL_DAYS": 14,
        "PERSISTENT_MEMORY_MIN_IMPORTANCE": 0.55,
        "PERSISTENT_MEMORY_MAX_PROMPT_CHARS": 1200,
        "PERSISTENT_MEMORY_QUEUE_MAX": 32,
        "PERSISTENT_MEMORY_WORKERS": 1,
    }

    distiller_cfg_data = {
        "MEMORY_DISTILLER_ENABLED": True,
        "MEMORY_DISTILLER_DRY_RUN": False,
        "MEMORY_DISTILLER_BATCH_SIZE": 200,
        "MEMORY_DISTILLER_INTERVAL_SECONDS": 900,
        "MEMORY_DISTILLER_WINDOW_MESSAGES": 25,
        "MEMORY_DISTILLER_MIN_CONFIDENCE": 0.85,
        "MEMORY_DISTILLER_MAX_MEMORIES_PER_WINDOW": 3,
        "MEMORY_DISTILLER_EXCLUDE_BOT_MESSAGES": True,
        "PERSISTENT_MEMORY_DEFAULT_TTL_DAYS": 180,
        "PERSISTENT_MEMORY_TEMP_TTL_DAYS": 14,
        "PERSISTENT_MEMORY_MIN_IMPORTANCE": 0.55,
    }

    monkeypatch.setattr(
        "bot.server_archive.service.load_config", lambda: archive_cfg_data
    )
    monkeypatch.setattr("bot.memory.service.load_config", lambda: memory_cfg_data)
    monkeypatch.setattr(
        "bot.memory.archive_distiller.load_config", lambda: distiller_cfg_data
    )

    archive_service = ServerArchiveService()
    memory_service = CuratedMemoryService()
    memory_service.semantic_store = FakeSemanticStore()
    archive_service_module._service = archive_service
    memory_service_module._memory_service = memory_service

    distiller = MemoryArchiveDistiller()
    distiller.archive_service = archive_service
    distiller.memory_service = memory_service
    try:
        yield {
            "archive_service": archive_service,
            "memory_service": memory_service,
            "distiller": distiller,
            "archive_db": archive_db,
            "memory_db": memory_db,
            "distiller_cfg": distiller_cfg_data,
        }
    finally:
        archive_service_module._service = original_archive_service
        memory_service_module._memory_service = original_memory_service


def _base_time() -> datetime:
    return datetime(2026, 5, 9, 1, 0, 0, tzinfo=timezone.utc)


LONG_USER_PREFERENCE = (
    "I prefer short replies unless I ask for detail, and I usually want the answer in one concise "
    "paragraph with bullets only when they help explain multiple steps."
)
LONG_SECOND_USER_PREFERENCE = (
    "I prefer Claude Code prompts after the summary, and I usually want them written as a short "
    "bullet list so I can copy them quickly into the tool."
)
LONG_PROJECT_RULE = (
    "For the discord-bot project, the bot should keep replies brief, lead with the answer, and avoid "
    "long explanations unless I ask for detail."
)


def _bundle(
    *,
    guild_id: str = "g1",
    channel_id: str = "c1",
    author_id: str = "u1",
    message_id: str,
    content: str,
    created_at: datetime,
    bot: bool = False,
    thread_id: str | None = None,
) -> ArchiveMessageBundle:
    guild = ArchiveGuild(guild_id=guild_id, name="Guild")
    channel = ArchiveChannel(channel_id=channel_id, guild_id=guild_id, name="general")
    author = ArchiveUser(
        user_id=author_id, username="user", display_name="User", bot=int(bot)
    )
    message = ArchiveMessage(
        message_id=message_id,
        guild_id=guild_id,
        channel_id=channel_id,
        thread_id=thread_id,
        author_id=author_id,
        content=content,
        clean_content=content,
        created_at=created_at.isoformat(),
    )
    thread = None
    if thread_id is not None:
        thread = ArchiveThread(
            thread_id=thread_id,
            guild_id=guild_id,
            parent_channel_id=channel_id,
            name="thread",
        )
    return ArchiveMessageBundle(
        guild=guild, channel=channel, author=author, message=message, thread=thread
    )


async def _insert_messages(
    archive_service: ServerArchiveService, bundles: list[ArchiveMessageBundle]
) -> None:
    for bundle in bundles:
        await archive_service.store.upsert_bundle(bundle)


@pytest.mark.asyncio
async def test_distiller_ignores_archive_when_disabled(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]
    distiller_env["distiller_cfg"]["MEMORY_DISTILLER_ENABLED"] = False

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time(),
            )
        ],
    )

    result = await distiller.run_once()
    assert result["skipped_reason"] == "disabled"
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert memories == []


@pytest.mark.asyncio
async def test_dry_run_scans_but_does_not_save_memory(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]
    distiller.set_dry_run(True)

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time(),
            )
        ],
    )

    result = await distiller.run_once()
    assert result["candidate_count"] == 1
    assert result["accepted_count"] == 1
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert memories == []


@pytest.mark.asyncio
async def test_accepted_user_preference_creates_curated_memory(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]
    distiller.set_dry_run(False)

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time(),
            )
        ],
    )

    result = await distiller.run_once()
    assert result["accepted_count"] == 1
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert len(memories) == 1
    assert memories[0].context_type == "user_preference"
    assert "short replies" in (memories[0].summary or memories[0].text).lower()


@pytest.mark.asyncio
async def test_accepted_project_rule_creates_scoped_memory(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]
    distiller.set_dry_run(False)

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_PROJECT_RULE,
                created_at=_base_time(),
            )
        ],
    )

    result = await distiller.run_once()
    assert result["accepted_count"] == 1
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert len(memories) == 1
    assert memories[0].context_type == "project_fact"
    assert memories[0].guild_id == "g1"
    assert memories[0].channel_id == "c1"


@pytest.mark.asyncio
async def test_debugging_chatter_is_rejected(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content="I'm debugging the router today and the code is broken.",
                created_at=_base_time(),
            )
        ],
    )

    result = await distiller.run_once()
    assert result["candidate_count"] == 0
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert memories == []


@pytest.mark.asyncio
async def test_secrets_are_rejected(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content="My API key is sk-12345 and don't repeat it.",
                created_at=_base_time(),
            )
        ],
    )

    result = await distiller.run_once()
    assert result["candidate_count"] == 0
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert memories == []


@pytest.mark.asyncio
async def test_bot_messages_are_ignored_by_default(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time(),
                bot=True,
            )
        ],
    )

    result = await distiller.run_once()
    assert result["candidate_count"] == 0
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert memories == []


@pytest.mark.asyncio
async def test_processed_checkpoint_is_updated(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time(),
            )
        ],
    )

    await distiller.run_once()
    state = await archive_service.store.get_distiller_state(
        guild_id="g1", channel_id="c1", thread_id=None, author_id="u1"
    )
    assert state is not None
    assert state["last_processed_message_id"] == "m1"


@pytest.mark.asyncio
async def test_restart_resumes_from_checkpoint(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]
    distiller_env["distiller_cfg"]["MEMORY_DISTILLER_BATCH_SIZE"] = 1

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time(),
            ),
            _bundle(
                message_id="m2",
                content=LONG_SECOND_USER_PREFERENCE,
                created_at=_base_time() + timedelta(minutes=31),
            ),
        ],
    )

    first = await distiller.run_once()
    assert first["accepted_count"] == 1
    state = await archive_service.store.get_distiller_state(
        guild_id="g1", channel_id="c1", thread_id=None, author_id="u1"
    )
    assert state["last_processed_message_id"] == "m1"

    second = await distiller.run_once()
    assert second["accepted_count"] == 1
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert len(memories) == 2
    assert state["last_processed_message_id"] == "m1"
    state = await archive_service.store.get_distiller_state(
        guild_id="g1", channel_id="c1", thread_id=None, author_id="u1"
    )
    assert state["last_processed_message_id"] == "m2"


@pytest.mark.asyncio
async def test_duplicate_memories_merge_instead_of_inserting_duplicates(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time(),
            ),
            _bundle(
                message_id="m2",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time() + timedelta(minutes=31),
            ),
        ],
    )

    result = await distiller.run_once()
    assert result["accepted_count"] == 2
    assert result["merged_count"] == 1
    memories = await memory_service.store.list_memories(
        user_id="u1", guild_id="g1", limit=10
    )
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_archive_results_are_never_directly_injected_into_prompt(distiller_env):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]

    raw_text = "I prefer short replies unless I ask for detail."
    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=raw_text,
                created_at=_base_time(),
            )
        ],
    )

    prompt_block = await memory_service.build_prompt_block(
        user_id="u1",
        guild_id="g1",
        channel_id="c1",
        thread_id=None,
        query=raw_text,
    )
    assert raw_text not in prompt_block
    assert prompt_block == ""
    assert (
        await memory_service.store.list_memories(user_id="u1", guild_id="g1", limit=10)
        == []
    )


@pytest.mark.asyncio
async def test_distiller_failure_does_not_break_normal_message_handling(
    distiller_env, monkeypatch
):
    distiller = distiller_env["distiller"]
    archive_service = distiller_env["archive_service"]
    memory_service = distiller_env["memory_service"]

    await _insert_messages(
        archive_service,
        [
            _bundle(
                message_id="m1",
                content=LONG_USER_PREFERENCE,
                created_at=_base_time(),
            )
        ],
    )

    def boom(_messages):
        raise RuntimeError("distiller exploded")

    monkeypatch.setattr(distiller, "_distill_window", boom)
    result = await distiller.run_once()
    assert result["error"] is not None

    record = await memory_service.add_explicit_memory(
        user_id="u1",
        text="I prefer short replies unless I ask for detail.",
        guild_id="g1",
        channel_id="c1",
        context_type="user_preference",
        source_message_id="manual",
        source="explicit_memory_command",
        metadata={"test": True},
    )
    assert record.summary
