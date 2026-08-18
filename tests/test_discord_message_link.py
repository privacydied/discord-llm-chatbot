"""Tests for Discord message-link resolution (jump URLs). [REH][SFT]"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import discord
import pytest

from bot.discord_message_link import (
    LinkResolution,
    link_budget,
    parse_message_link,
    render_message,
    requester_can_read,
    resolve_message_link,
)


GUILD_ID = 1156692144740892796
CHANNEL_ID = 1392318624508678154
MESSAGE_ID = 1539083511003349112
LINK = f"https://discord.com/channels/{GUILD_ID}/{CHANNEL_ID}/{MESSAGE_ID}"


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        LINK,
        f"https://canary.discord.com/channels/{GUILD_ID}/{CHANNEL_ID}/{MESSAGE_ID}",
        f"https://ptb.discord.com/channels/{GUILD_ID}/{CHANNEL_ID}/{MESSAGE_ID}",
        f"http://discordapp.com/channels/{GUILD_ID}/{CHANNEL_ID}/{MESSAGE_ID}",
    ],
)
def test_parse_accepts_known_link_shapes(url: str) -> None:
    ref = parse_message_link(url)
    assert ref is not None
    assert ref.guild_id == GUILD_ID
    assert ref.channel_id == CHANNEL_ID
    assert ref.message_id == MESSAGE_ID
    assert ref.is_dm is False


def test_parse_dm_link() -> None:
    ref = parse_message_link(f"https://discord.com/channels/@me/{CHANNEL_ID}/{MESSAGE_ID}")
    assert ref is not None
    assert ref.is_dm is True
    assert ref.guild_id is None


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://example.com/channels/1/2/3",
        f"https://discord.com/channels/{GUILD_ID}/{CHANNEL_ID}",  # invite/channel link, no message
        "https://discord.gg/abcdef",
        "https://cdn.discordapp.com/attachments/1/2/pic.png",
    ],
)
def test_parse_rejects_non_message_links(url: str) -> None:
    assert parse_message_link(url) is None


# --------------------------------------------------------------------------- #
# permission gate
# --------------------------------------------------------------------------- #
def _perms(view: bool = True, history: bool = True) -> SimpleNamespace:
    return SimpleNamespace(view_channel=view, read_message_history=history)


class _FakeChannel:
    def __init__(self, perms: SimpleNamespace, *, member=None, message=None) -> None:
        self._perms = perms
        self.name = "general"
        self.guild = SimpleNamespace(name="Guild", get_member=lambda _id: member)
        self._message = message

    def permissions_for(self, _member):
        return self._perms

    async def fetch_message(self, _mid):
        if isinstance(self._message, Exception):
            raise self._message
        return self._message


async def test_requester_can_read_requires_member() -> None:
    ch = _FakeChannel(_perms(), member=None)
    assert await requester_can_read(ch, None) is False
    # Plain User, absent from cache and from the members API -> denied.
    assert await requester_can_read(ch, SimpleNamespace(id=7)) is False


async def test_requester_can_read_honours_permissions() -> None:
    user = SimpleNamespace(id=7)
    member = SimpleNamespace(id=7)
    assert await requester_can_read(_FakeChannel(_perms(), member=member), user) is True
    assert await requester_can_read(_FakeChannel(_perms(view=False), member=member), user) is False
    assert await requester_can_read(_FakeChannel(_perms(history=False), member=member), user) is False


async def test_requester_can_read_falls_back_to_member_fetch() -> None:
    """Cache miss must not deny: without the members intent the cache is empty."""
    fetched = SimpleNamespace(id=7)

    async def _fetch_member(uid):
        assert uid == 7
        return fetched

    ch = _FakeChannel(_perms(), member=None)
    ch.guild = SimpleNamespace(name="Guild", get_member=lambda _id: None, fetch_member=_fetch_member)
    assert await requester_can_read(ch, SimpleNamespace(id=7)) is True


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fake_message(content: str = "hello world", **kw):
    return SimpleNamespace(
        author=SimpleNamespace(display_name="alice", name="alice"),
        channel=SimpleNamespace(name="general"),
        guild=SimpleNamespace(name="Guild"),
        created_at=None,
        clean_content=content,
        content=content,
        attachments=kw.get("attachments", []),
        embeds=kw.get("embeds", []),
        stickers=kw.get("stickers", []),
    )


def test_render_includes_author_body_and_attachments() -> None:
    att = SimpleNamespace(filename="cat.png", url="https://cdn/cat.png")
    out = render_message(_fake_message("hi there", attachments=[att]))
    assert "alice" in out
    assert "#general" in out
    assert "hi there" in out
    assert "cat.png" in out


def test_render_truncates_to_max_chars() -> None:
    out = render_message(_fake_message("x" * 5000), max_chars=200)
    assert len(out) <= 200
    assert out.endswith("[truncated]")


def test_render_handles_empty_body() -> None:
    assert "(no text content)" in render_message(_fake_message(""))


# --------------------------------------------------------------------------- #
# end-to-end resolution
# --------------------------------------------------------------------------- #
class _FakeBot:
    def __init__(self, channel) -> None:
        self._channel = channel

    def get_channel(self, _cid):
        return self._channel


async def test_resolve_returns_none_for_non_discord_url() -> None:
    assert await resolve_message_link(_FakeBot(None), "https://example.com/post/1") is None


async def test_resolve_success_wraps_untrusted_content() -> None:
    user = SimpleNamespace(id=7)
    channel = _FakeChannel(_perms(), member=SimpleNamespace(id=7), message=_fake_message("secret sauce"))
    res = await resolve_message_link(_FakeBot(channel), LINK, requester=user)
    assert isinstance(res, LinkResolution)
    assert res.ok is True
    assert "secret sauce" in res.text
    assert "UNVERIFIED EXTERNAL CONTENT" in res.text


async def test_resolve_denies_when_requester_lacks_access() -> None:
    user = SimpleNamespace(id=7)
    channel = _FakeChannel(_perms(view=False), member=SimpleNamespace(id=7), message=_fake_message("private"))
    res = await resolve_message_link(_FakeBot(channel), LINK, requester=user)
    assert res is not None and res.ok is False
    assert res.reason == "requester_no_access"
    assert "private" not in res.text


async def test_resolve_reports_dm_links() -> None:
    res = await resolve_message_link(_FakeBot(None), f"https://discord.com/channels/@me/{CHANNEL_ID}/{MESSAGE_ID}")
    assert res is not None and res.ok is False
    assert res.reason == "dm_link"


async def test_resolve_reports_unknown_channel() -> None:
    bot = SimpleNamespace(get_channel=lambda _cid: None)
    res = await resolve_message_link(bot, LINK, requester=SimpleNamespace(id=7))
    assert res is not None and res.ok is False
    assert res.reason == "channel_unavailable"


async def test_resolve_reports_deleted_message() -> None:
    err = discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "unknown message")
    channel = _FakeChannel(_perms(), member=SimpleNamespace(id=7), message=err)
    res = await resolve_message_link(_FakeBot(channel), LINK, requester=SimpleNamespace(id=7))
    assert res is not None and res.ok is False
    assert res.reason == "message_not_found"


async def test_resolve_times_out_cleanly() -> None:
    class _SlowChannel(_FakeChannel):
        async def fetch_message(self, _mid):
            await asyncio.sleep(5)

    channel = _SlowChannel(_perms(), member=SimpleNamespace(id=7))
    res = await resolve_message_link(_FakeBot(channel), LINK, requester=SimpleNamespace(id=7), timeout_s=0.05)
    assert res is not None and res.ok is False
    assert res.reason == "timeout"


# --------------------------------------------------------------------------- #
# config budget
# --------------------------------------------------------------------------- #
def test_link_budget_reads_config_and_falls_back() -> None:
    assert link_budget({"DISCORD_LINK_TIMEOUT_S": 3, "DISCORD_LINK_MAX_CHARS": 100}) == (3.0, 100)
    assert link_budget({"DISCORD_LINK_TIMEOUT_S": "nope"}) == (10.0, 4000)
    assert link_budget(None) == (10.0, 4000)
