"""Tests for the context-manager write debounce fix [PA].

ContextManager.append() and EnhancedContextManager.append_message() used to do a
full-file atomic rewrite (+fsync) of the *entire* tracked memory dict on every
single message -- one of the two biggest disk-write offenders found in the I/O
audit, since it fired on every message regardless of whether vision/RAG/TTS
were even in use. They now just set an in-memory dirty flag; a periodic task
(bot/tasks.py's context_autosave) and shutdown are responsible for the actual
flush. These tests lock in that contract:

- append()/append_message() must NOT write to disk synchronously.
- In-process reads must still see the just-appended data immediately (no
  read-consistency gap from deferring the write).
- flush_if_dirty() must actually persist, and must no-op when nothing changed.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import discord
import pytest

from bot.memory.context_manager import ContextManager
from bot.memory.enhanced_context_manager import EnhancedContextManager


class _FakeAuthor:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class _FakeGuild:
    def __init__(self, guild_id: int) -> None:
        self.id = guild_id


class _FakeMessage:
    def __init__(self, *, content: str, author_id: int = 111, channel_id: int = 222, guild_id: int | None = 333, is_dm: bool = False) -> None:
        self.id = 999
        self.author = _FakeAuthor(author_id)
        self.content = content
        self.created_at = datetime.now(UTC)
        self.guild = None if is_dm else _FakeGuild(guild_id)
        if is_dm:
            # isinstance(channel, discord.DMChannel) must hold for
            # EnhancedContextManager's context-key routing.
            self.channel = MagicMock(spec=discord.DMChannel)
            self.channel.id = channel_id
        else:
            channel = MagicMock(spec=discord.TextChannel)
            channel.id = channel_id
            channel.parent = None
            self.channel = channel


@pytest.fixture
def fake_bot() -> MagicMock:
    return MagicMock()


class TestContextManagerDebounce:
    def test_append_marks_dirty_without_writing_to_disk(self, tmp_path, fake_bot) -> None:
        filepath = tmp_path / "context.json"
        cm = ContextManager(fake_bot, filepath=str(filepath))

        msg = _FakeMessage(content="hello there")
        cm.append(msg)

        assert cm._dirty is True
        assert not filepath.exists(), "append() must not synchronously write to disk"

    def test_read_after_append_sees_data_before_flush(self, tmp_path, fake_bot) -> None:
        filepath = tmp_path / "context.json"
        cm = ContextManager(fake_bot, filepath=str(filepath))

        msg = _FakeMessage(content="in-memory read check")
        cm.append(msg)

        # get_context() reads self.memory directly -- must see the new entry
        # immediately even though nothing has been flushed to disk yet.
        history = cm.get_context(msg)
        assert len(history) == 1
        assert history[0]["content"] == "in-memory read check"

    @pytest.mark.asyncio
    async def test_flush_if_dirty_persists_and_clears_flag(self, tmp_path, fake_bot) -> None:
        filepath = tmp_path / "context.json"
        cm = ContextManager(fake_bot, filepath=str(filepath))

        msg = _FakeMessage(content="flush me")
        cm.append(msg)

        flushed = await cm.flush_if_dirty()

        assert flushed is True
        assert cm._dirty is False
        assert filepath.exists()

        import json

        data = json.loads(filepath.read_text())
        primary_key, _ = cm._get_source_keys(msg)
        assert primary_key in data

    @pytest.mark.asyncio
    async def test_flush_if_dirty_is_noop_when_clean(self, tmp_path, fake_bot) -> None:
        filepath = tmp_path / "context.json"
        cm = ContextManager(fake_bot, filepath=str(filepath))

        flushed = await cm.flush_if_dirty()

        assert flushed is False
        assert not filepath.exists()


class TestEnhancedContextManagerDebounce:
    @pytest.mark.asyncio
    async def test_append_message_marks_dirty_without_writing_to_disk(self, tmp_path, fake_bot) -> None:
        filepath = tmp_path / "enhanced_context.json"
        ecm = EnhancedContextManager(fake_bot, filepath=str(filepath))

        msg = _FakeMessage(content="hello enhanced")
        await ecm.append_message(msg, role="user")

        assert ecm._dirty is True
        assert not filepath.exists(), "append_message() must not synchronously write to disk"

    @pytest.mark.asyncio
    async def test_read_after_append_message_sees_data_before_flush(self, tmp_path, fake_bot) -> None:
        filepath = tmp_path / "enhanced_context.json"
        ecm = EnhancedContextManager(fake_bot, filepath=str(filepath))

        msg = _FakeMessage(content="in-memory enhanced read check")
        await ecm.append_message(msg, role="user")

        entries = ecm.get_context_for_user(msg)
        assert len(entries) == 1
        assert ecm._decrypt_content(entries[0].content) == "in-memory enhanced read check"

    @pytest.mark.asyncio
    async def test_flush_if_dirty_persists_and_clears_flag(self, tmp_path, fake_bot) -> None:
        filepath = tmp_path / "enhanced_context.json"
        ecm = EnhancedContextManager(fake_bot, filepath=str(filepath))

        msg = _FakeMessage(content="flush enhanced me")
        await ecm.append_message(msg, role="user")

        flushed = await ecm.flush_if_dirty()

        assert flushed is True
        assert ecm._dirty is False
        assert filepath.exists()

        import json

        data = json.loads(filepath.read_text())
        context_key = ecm._get_context_key(msg)
        assert context_key in data["messages"]

    @pytest.mark.asyncio
    async def test_flush_if_dirty_is_noop_when_clean(self, tmp_path, fake_bot) -> None:
        filepath = tmp_path / "enhanced_context.json"
        ecm = EnhancedContextManager(fake_bot, filepath=str(filepath))

        flushed = await ecm.flush_if_dirty()

        assert flushed is False
        assert not filepath.exists()

    @pytest.mark.asyncio
    async def test_dm_messages_never_persisted_even_after_flush(self, tmp_path, fake_bot) -> None:
        """Privacy: DM conversations must never hit disk, flushed or not."""
        filepath = tmp_path / "enhanced_context.json"
        ecm = EnhancedContextManager(fake_bot, filepath=str(filepath))

        msg = _FakeMessage(content="a private dm", is_dm=True)
        await ecm.append_message(msg, role="user")
        await ecm.flush_if_dirty()

        if filepath.exists():
            import json

            data = json.loads(filepath.read_text())
            assert all(not k.startswith("dm_") for k in data.get("messages", {}))
