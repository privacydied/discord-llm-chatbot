import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List, Optional

import pytest

import discord

from bot.memory.mention_context import (
    maybe_build_mention_context,
)


# Helpers -------------------------------------------------------------

class FakeAuthor:
    def __init__(self, id: int, name: str, bot: bool = False):
        self.id = id
        self.name = name
        self.display_name = name
        self.bot = bot


class FakeMessage:
    def __init__(
        self,
        id: int,
        author: FakeAuthor,
        content: str,
        created_at: datetime,
        channel,
        guild=None,
        reference: Optional[SimpleNamespace] = None,
    ):
        self.id = id
        self.author = author
        self.content = content
        self.created_at = created_at
        self.channel = channel
        self.guild = guild
        self.reference = reference
        self.mentions = []
        self.jump_url = f"https://discordapp.com/channels/0/0/{id}"
        self.attachments = []


class FakeHistory:
    def __init__(self, items: List[FakeMessage]):
        self._items = list(items)

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class FakeChannel:
    def __init__(self, channel_id: int, items: List[FakeMessage], channel_type=None):
        self.id = channel_id
        self._items = list(items)
        self.type = channel_type

    def history(self, *, limit=None, oldest_first=False, after=None):
        items = list(self._items)
        if after is not None:
            items = [m for m in items if m.created_at > after]
        if oldest_first:
            items = sorted(items, key=lambda m: m.created_at)
        else:
            items = sorted(items, key=lambda m: m.created_at, reverse=True)
        if limit:
            items = items[:limit]
        return FakeHistory(items)

    async def fetch_message(self, mid: int):
        for m in self._items:
            if m.id == mid:
                return m
        raise discord.NotFound(SimpleNamespace(), "not found")


class FakeGuild:
    def __init__(self, gid: int, members: List[FakeAuthor]):
        self.id = gid
        self._mem_by_id = {m.id: m for m in members}
        self.members = list(members)

    def get_member(self, uid: int):
        return self._mem_by_id.get(uid)


class FakeBot:
    def __init__(self, uid: int):
        self.user = SimpleNamespace(id=uid)


def _now():
    return datetime.now(timezone.utc)


# Tests ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_thread_case_short():
    now = _now()
    bot = FakeBot(999)
    author = FakeAuthor(1, "alice")
    guild = FakeGuild(10, [author])

    # Build 6 messages in a thread
    items = []
    ch = FakeChannel(1000, items, channel_type=discord.ChannelType.public_thread)
    for i in range(6):
        m = FakeMessage(
            id=100 + i,
            author=author,
            content=f"m{i}",
            created_at=now - timedelta(minutes=5 - i),
            channel=ch,
            guild=guild,
        )
        items.append(m)
    trigger = items[-1]
    trigger.mentions = [SimpleNamespace(id=bot.user.id)]

    cfg = {
        "MEM_MENTION_CONTEXT_ENABLED": True,
        "MEM_MAX_MSGS": 40,
        "MEM_MAX_CHARS": 8000,
        "MEM_MAX_AGE_MIN": 240,
        "MEM_FETCH_TIMEOUT_S": 2,
    }

    joined, block = await maybe_build_mention_context(bot, trigger, cfg)
    assert block.source == "discord-thread"
    assert block.count == 6
    # Oldest -> newest
    assert block.items[0].text_plain == "m0"
    assert block.items[-1].text_plain == "m5"
    assert "[1/6]" in joined


@pytest.mark.asyncio
async def test_thread_case_cap_truncate():
    now = _now()
    bot = FakeBot(999)
    author = FakeAuthor(1, "alice")
    guild = FakeGuild(10, [author])

    items = []
    ch = FakeChannel(1001, items, channel_type=discord.ChannelType.public_thread)
    for i in range(120):
        m = FakeMessage(
            id=200 + i,
            author=author,
            content=f"m{i}",
            created_at=now - timedelta(minutes=120 - i),
            channel=ch,
            guild=guild,
        )
        items.append(m)
    trigger = items[-1]
    trigger.mentions = [SimpleNamespace(id=bot.user.id)]

    cfg = {
        "MEM_MENTION_CONTEXT_ENABLED": True,
        "MEM_MAX_MSGS": 40,
        "MEM_MAX_CHARS": 8000,
        "MEM_MAX_AGE_MIN": 240,
        "MEM_FETCH_TIMEOUT_S": 2,
    }

    joined, block = await maybe_build_mention_context(bot, trigger, cfg)
    assert block.count <= 40
    assert block.truncated is True


@pytest.mark.asyncio
async def test_reply_case_linear():
    now = _now()
    bot = FakeBot(999)
    a = FakeAuthor(1, "alice")
    b = FakeAuthor(2, "bob")
    guild = FakeGuild(10, [a, b])

    items = []
    ch = FakeChannel(1002, items, channel_type=None)

    root = FakeMessage(301, a, "root", now - timedelta(minutes=4), ch, guild)
    m2 = FakeMessage(302, b, "r1", now - timedelta(minutes=3), ch, guild, reference=SimpleNamespace(message_id=root.id))
    m3 = FakeMessage(303, a, "r2", now - timedelta(minutes=2), ch, guild, reference=SimpleNamespace(message_id=m2.id))
    m4 = FakeMessage(304, b, "r3", now - timedelta(minutes=1), ch, guild, reference=SimpleNamespace(message_id=m3.id))
    items.extend([root, m2, m3, m4])

    trigger = m4
    trigger.mentions = [SimpleNamespace(id=bot.user.id)]

    cfg = {
        "MEM_MENTION_CONTEXT_ENABLED": True,
        "MEM_FETCH_TIMEOUT_S": 2,
    }

    joined, block = await maybe_build_mention_context(bot, trigger, cfg)
    assert block.source == "discord-reply-chain"
    assert [it.text_plain for it in block.items] == ["root", "r1", "r2", "r3"]


@pytest.mark.asyncio
async def test_reply_case_forked_only_chain_included():
    now = _now()
    bot = FakeBot(999)
    a = FakeAuthor(1, "alice")
    b = FakeAuthor(2, "bob")
    c = FakeAuthor(3, "cara")
    guild = FakeGuild(10, [a, b, c])

    items = []
    ch = FakeChannel(1003, items)

    root = FakeMessage(401, a, "root", now - timedelta(minutes=5), ch, guild)
    other = FakeMessage(402, c, "noise", now - timedelta(minutes=4), ch, guild)
    r1 = FakeMessage(403, b, "r1", now - timedelta(minutes=3), ch, guild, reference=SimpleNamespace(message_id=root.id))
    noise_reply = FakeMessage(404, c, "noise_reply", now - timedelta(minutes=2, seconds=30), ch, guild, reference=SimpleNamespace(message_id=other.id))
    r2 = FakeMessage(405, a, "r2", now - timedelta(minutes=2), ch, guild, reference=SimpleNamespace(message_id=r1.id))
    trigger = FakeMessage(406, b, "r3", now - timedelta(minutes=1), ch, guild, reference=SimpleNamespace(message_id=r2.id))
    items.extend([root, other, r1, noise_reply, r2, trigger])
    trigger.mentions = [SimpleNamespace(id=bot.user.id)]

    cfg = {"MEM_MENTION_CONTEXT_ENABLED": True}
    joined, block = await maybe_build_mention_context(bot, trigger, cfg)
    assert [it.text_plain for it in block.items] == ["root", "r1", "r2", "r3"]


@pytest.mark.asyncio
async def test_lone_case_noop():
    now = _now()
    bot = FakeBot(999)
    a = FakeAuthor(1, "alice")
    guild = FakeGuild(10, [a])

    ch = FakeChannel(1004, [])
    msg = FakeMessage(501, a, "hi @bot", now, ch, guild)
    # Mention is not set; should not build
    cfg = {"MEM_MENTION_CONTEXT_ENABLED": True}
    res = await maybe_build_mention_context(bot, msg, cfg)
    assert res is None


@pytest.mark.asyncio
async def test_age_cap_excludes_old_except_root():
    now = _now()
    bot = FakeBot(999)
    a = FakeAuthor(1, "alice")
    guild = FakeGuild(10, [a])

    items = []
    ch = FakeChannel(1005, items)
    # Root very old
    root = FakeMessage(601, a, "root", now - timedelta(minutes=10000), ch, guild)
    r1 = FakeMessage(602, a, "r1", now - timedelta(minutes=10), ch, guild, reference=SimpleNamespace(message_id=root.id))
    trigger = FakeMessage(603, a, "r2", now - timedelta(minutes=5), ch, guild, reference=SimpleNamespace(message_id=r1.id))
    items.extend([root, r1, trigger])
    trigger.mentions = [SimpleNamespace(id=bot.user.id)]

    cfg = {"MEM_MENTION_CONTEXT_ENABLED": True, "MEM_MAX_AGE_MIN": 60}
    joined, block = await maybe_build_mention_context(bot, trigger, cfg)
    # root + r1 + r2; root allowed even if over age
    assert [it.text_plain for it in block.items] == ["root", "r1", "r2"]


@pytest.mark.asyncio
async def test_bot_filter_excludes_others_keeps_ours():
    now = _now()
    bot = FakeBot(999)
    a = FakeAuthor(1, "alice")
    other_bot = FakeAuthor(77, "otherbot", bot=True)
    our_bot_user = SimpleNamespace(id=bot.user.id)
    our_bot_author = FakeAuthor(bot.user.id, "me", bot=True)
    guild = FakeGuild(10, [a, our_bot_author])

    items = []
    ch = FakeChannel(1006, items)

    root = FakeMessage(701, a, "root", now - timedelta(minutes=3), ch, guild)
    bot_msg = FakeMessage(702, other_bot, "bot noise", now - timedelta(minutes=2), ch, guild, reference=SimpleNamespace(message_id=root.id))
    ours = FakeMessage(703, our_bot_author, "bot reply", now - timedelta(minutes=1), ch, guild, reference=SimpleNamespace(message_id=root.id))
    trigger = FakeMessage(704, a, "ok", now, ch, guild, reference=SimpleNamespace(message_id=ours.id))
    items.extend([root, bot_msg, ours, trigger])
    trigger.mentions = [our_bot_user]

    cfg = {"MEM_MENTION_CONTEXT_ENABLED": True}
    joined, block = await maybe_build_mention_context(bot, trigger, cfg)
    texts = [it.text_plain for it in block.items]
    assert "bot noise" not in texts
    assert "bot reply" in texts


@pytest.mark.asyncio
async def test_timeout_fallback_clean():
    now = _now()
    bot = FakeBot(999)
    a = FakeAuthor(1, "alice")
    guild = FakeGuild(10, [a])

    class SlowChannel(FakeChannel):
        async def fetch_message(self, mid: int):
            # Simulate long stall
            await asyncio.sleep(0.2)
            raise asyncio.TimeoutError()

    items = []
    ch = SlowChannel(1007, items)
    root = FakeMessage(801, a, "root", now - timedelta(minutes=3), ch, guild)
    trigger = FakeMessage(803, a, "ok", now, ch, guild, reference=SimpleNamespace(message_id=root.id))
    items.extend([root, trigger])
    trigger.mentions = [SimpleNamespace(id=bot.user.id)]

    cfg = {"MEM_MENTION_CONTEXT_ENABLED": True, "MEM_FETCH_TIMEOUT_S": 0.05}
    res = await maybe_build_mention_context(bot, trigger, cfg)
    assert res is None


@pytest.mark.asyncio
async def test_merge_dedup_within_block():
    # Ensure duplicates inside collected set are removed by message id
    now = _now()
    bot = FakeBot(999)
    a = FakeAuthor(1, "alice")
    guild = FakeGuild(10, [a])

    items = []
    ch = FakeChannel(1008, items, channel_type=discord.ChannelType.public_thread)

    m0 = FakeMessage(901, a, "m0", now - timedelta(minutes=2), ch, guild)
    dup_m0 = FakeMessage(901, a, "m0", now - timedelta(minutes=1, seconds=30), ch, guild)
    m1 = FakeMessage(902, a, "m1", now - timedelta(minutes=1), ch, guild)
    items.extend([m0, dup_m0, m1])
    trigger = m1
    trigger.mentions = [SimpleNamespace(id=bot.user.id)]

    cfg = {"MEM_MENTION_CONTEXT_ENABLED": True}
    joined, block = await maybe_build_mention_context(bot, trigger, cfg)
    ids = [it.id for it in block.items]
    # only two unique ids (901, 902)
    assert ids == [str(901), str(902)]
