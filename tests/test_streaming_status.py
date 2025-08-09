import asyncio
import types
from typing import Optional, List

import pytest

# Use the real BotAction dataclass
from bot.action import BotAction
from bot.core.bot import LLMBot
import discord


class FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeChannel:
    def __init__(self):
        self.sent_messages: List[FakeMessage] = []

    def typing(self):
        return FakeTyping()

    async def send(self, content=None, embeds=None, files=None, **kwargs):
        m = FakeMessage(channel=self, content=content or "", embeds=embeds or [])
        self.sent_messages.append(m)
        return m


class FakeAuthor:
    def __init__(self, id: int):
        self.id = id


class FakeGuild:
    def __init__(self, id: int):
        self.id = id


class FakeMessage:
    _id_counter = 1000

    def __init__(self, channel: FakeChannel, content: str = "", embeds: Optional[list] = None,
                 author: Optional[FakeAuthor] = None, guild: Optional[FakeGuild] = None):
        FakeMessage._id_counter += 1
        self.id = FakeMessage._id_counter
        self.channel = channel
        self.content = content
        self.embeds = embeds or []
        self.author = author or FakeAuthor(id=111)
        self.guild = guild or FakeGuild(id=222)
        self._deleted = False
        self._edits: List[tuple] = []

    async def reply(self, content=None, embeds=None, files=None, mention_author=True):
        # Simulate creating a new message in channel
        m = FakeMessage(channel=self.channel, content=content or "", embeds=embeds or [], author=self.author, guild=self.guild)
        self.channel.sent_messages.append(m)
        return m

    async def edit(self, content=None, embeds=None):
        # Record edit history and mutate state
        self._edits.append((content, embeds))
        if content is not None:
            self.content = content
        if embeds is not None:
            self.embeds = embeds
        return self

    async def delete(self):
        self._deleted = True
        return True


@pytest.mark.asyncio
async def test_streaming_lifecycle_and_final_edit(monkeypatch):
    # Configure a minimal bot instance (no network). Intents and prefix are arbitrary here.
    intents = discord.Intents.none()
    bot = LLMBot(command_prefix="!", intents=intents, config={
        "STREAMING_ENABLE": True,
        "STREAMING_EMBED_STYLE": "compact",
        "STREAMING_TICK_MS": 5,
        "STREAMING_MAX_STEPS": 3,
    })
    # Disable enhanced context manager for unit test
    bot.enhanced_context_manager = None

    # Build a fake incoming message
    ch = FakeChannel()
    incoming = FakeMessage(channel=ch, content="hello")

    # Start streaming status
    stream_ctx = await bot._start_streaming_status(incoming)
    assert stream_ctx and "message" in stream_ctx and "task" in stream_ctx
    placeholder = stream_ctx["message"]
    updater_task = stream_ctx["task"]
    assert not updater_task.done()

    # Let updater tick a bit
    await asyncio.sleep(0.03)

    # Stop streaming before final response
    await bot._stop_streaming_status(stream_ctx, final_label="✅ Done")
    # The task should be cancelled/finished
    assert updater_task.done()

    # Ensure placeholder received at least one edit during lifecycle
    assert len(placeholder._edits) >= 1

    # Execute final action by editing placeholder (no files/audio)
    action = BotAction(content="Final content", embeds=[])
    await bot._execute_action(incoming, action, target_message=placeholder)

    # Validate that the placeholder was edited (not deleted and not a new message)
    assert not placeholder._deleted
    assert placeholder.content == "Final content"
    assert len(ch.sent_messages) >= 1  # initial placeholder + no new final message


@pytest.mark.asyncio
async def test_replace_placeholder_when_files_present():
    intents = discord.Intents.none()
    bot = LLMBot(command_prefix="!", intents=intents, config={
        "STREAMING_ENABLE": True,
        "STREAMING_EMBED_STYLE": "compact",
        "STREAMING_TICK_MS": 5,
        "STREAMING_MAX_STEPS": 2,
    })
    bot.enhanced_context_manager = None

    ch = FakeChannel()
    incoming = FakeMessage(channel=ch, content="hello")

    stream_ctx = await bot._start_streaming_status(incoming)
    placeholder = stream_ctx["message"]

    # Simulate an action with a file by setting audio_path (file existence not checked in this test)
    action = BotAction(content="With file", embeds=[], audio_path="/nonexistent.ogg")

    # Monkeypatch os.path.exists and builtins.open to simulate existing audio file
    import os as _os
    import builtins as _builtins
    import io as _io
    orig_exists = _os.path.exists
    orig_open = _builtins.open
    try:
        _os.path.exists = lambda p: True
        _builtins.open = lambda p, mode='rb', *a, **k: _io.BytesIO(b"dummy-audio") if p == "/nonexistent.ogg" else orig_open(p, mode, *a, **k)
        await bot._execute_action(incoming, action, target_message=placeholder)
    finally:
        _os.path.exists = orig_exists
        _builtins.open = orig_open

    # Placeholder should be deleted and a new message sent
    assert placeholder._deleted
    # There should be at least two messages: placeholder + final new message
    assert len(ch.sent_messages) >= 2
