"""ppv.* preview cards (bot/embed_fix/ppv.py) and their handler wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import discord
import pytest

from bot.embed_fix import handler as h
from bot.embed_fix import ppv

SLUG = "misfits-boxing-joey-essex-vs-dapper-laughs-30174"
PAGE = f"https://ppvs.pk/live/{SLUG}"
RECORD = {
    "id": 30174,
    "name": "Misfits Boxing: Joey Essex vs. Dapper Laughs",
    "poster": "https://api.ppv.st/assets/thumb/x-thumbnail.jpg",
    "tag": "Boxing",
    "source_tag": "DAZN",
    "start_timestamp": 1_000,
    "end_timestamp": 2_000,
    "viewers": 1483,
    "m3u8": "",
    "sources": [{"type": "iframe", "data": "https://example.invalid/embed/x"}],
}


def test_build_embed_live():
    embed = ppv.build_embed(RECORD, PAGE, now=1_500)
    assert embed.title == RECORD["name"]
    assert embed.url == PAGE
    assert embed.description == "🔴 LIVE"
    assert embed.image.url == RECORD["poster"]
    fields = {f.name: f.value for f in embed.fields}
    assert fields == {"Category": "Boxing", "Source": "DAZN", "Start": "<t:1000:F>", "Viewers": "1,483"}
    # The stream itself is never exposed.
    assert "example.invalid" not in str(embed.to_dict())


@pytest.mark.parametrize(
    ("now", "record", "status"),
    [
        (500, RECORD, "⏳ Starts <t:1000:R>"),
        (2_500, RECORD, "⏹️ Ended"),
        (1_500, {**RECORD, "end_timestamp": 0}, "🔴 LIVE"),
        (1_500, {**RECORD, "start_timestamp": 0}, "🔴 24/7"),
    ],
)
def test_build_embed_status(now, record, status):
    assert ppv.build_embed(record, PAGE, now=now).description == status


def test_build_embed_skips_bad_poster_and_clips_title():
    embed = ppv.build_embed({"name": "x" * 400, "poster": "http://insecure/x.jpg"}, PAGE, now=0)
    assert len(embed.title) == 256
    assert embed.image.url is None


class _Resp:
    def __init__(self, status, payload):
        self.status, self._payload = status, payload

    async def __aenter__(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self._payload


def _session(*responses):
    session = MagicMock()
    session.get = MagicMock(side_effect=list(responses))
    return session


async def test_fetch_stream_success_uses_slug():
    session = _session(_Resp(200, {"success": True, "data": RECORD}))
    assert await ppv.fetch_stream(session, SLUG) == RECORD
    assert session.get.call_args.args[0] == f"https://api.ppv.st/api/streams/{SLUG}"


@pytest.mark.parametrize(
    "first",
    [
        _Resp(500, {}),
        _Resp(200, {"success": False, "error": "API endpoint not found."}),
        _Resp(200, aiohttp.ClientConnectionError("down")),
        _Resp(200, ValueError("bad json")),
    ],
)
async def test_fetch_stream_falls_back_to_second_api(first):
    session = _session(first, _Resp(200, {"success": True, "data": RECORD}))
    assert await ppv.fetch_stream(session, SLUG) == RECORD
    assert session.get.call_args.args[0].startswith("https://api.ppv.cx/api/")


async def test_fetch_stream_all_fail():
    assert await ppv.fetch_stream(_session(_Resp(404, {}), _Resp(404, {})), SLUG) is None


async def test_build_cards_drops_failures_and_bad_urls(monkeypatch):
    async def fake_fetch(session, slug):
        return RECORD if slug == SLUG else None

    monkeypatch.setattr(ppv, "fetch_stream", fake_fetch)
    cards = await ppv.build_cards([PAGE, "https://ppv.st/live/missing-1", "https://evil.com/live/x"])
    assert [c.url for c in cards] == [PAGE]


# --- handler wiring ---------------------------------------------------------


def _message(content):
    msg = MagicMock()
    msg.id, msg.content = 1, content
    msg.author = SimpleNamespace(bot=False, id=2)
    msg.guild = SimpleNamespace(id=3, me=SimpleNamespace(id=4))
    perms = discord.Permissions(send_messages=True, embed_links=True, read_message_history=True, manage_messages=True)
    msg.channel = MagicMock()
    msg.channel.permissions_for = MagicMock(return_value=perms)
    msg.channel.is_nsfw = MagicMock(return_value=False)
    msg.edit = AsyncMock()
    return msg


@pytest.fixture
def reply(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(h, "safe_reply", mock)
    monkeypatch.setattr(h, "is_server_feature_enabled", lambda guild_id, feature: True)
    monkeypatch.setattr(h, "SUPPRESS_RECHECK_DELAYS_S", ())
    return mock


async def test_handler_sends_card_without_text(reply, monkeypatch):
    monkeypatch.setattr(ppv, "fetch_stream", AsyncMock(return_value=RECORD))
    fixes = await h.maybe_fix_embeds(_message(f"yo {PAGE}"), {})
    assert [f.site for f in fixes] == ["ppv"]
    args, kwargs = reply.await_args
    assert args[1] is None
    assert [e.title for e in kwargs["embeds"]] == [RECORD["name"]]


async def test_handler_mixes_links_and_cards(reply, monkeypatch):
    monkeypatch.setattr(ppv, "fetch_stream", AsyncMock(return_value=RECORD))
    await h.maybe_fix_embeds(_message(f"https://x.com/a/status/1 {PAGE}"), {})
    args, kwargs = reply.await_args
    assert args[1] == "https://vxtwitter.com/a/status/1"
    assert len(kwargs["embeds"]) == 1


async def test_handler_lookup_failure_sends_nothing(reply, monkeypatch):
    monkeypatch.setattr(ppv, "fetch_stream", AsyncMock(return_value=None))
    msg = _message(PAGE)
    assert await h.maybe_fix_embeds(msg, {}) == []
    reply.assert_not_awaited()
    msg.edit.assert_not_awaited()


async def test_handler_ppv_can_be_disabled(reply, monkeypatch):
    fetch = AsyncMock(return_value=RECORD)
    monkeypatch.setattr(ppv, "fetch_stream", fetch)
    assert await h.maybe_fix_embeds(_message(PAGE), {"EMBED_FIX_DISABLED_SITES": "ppv"}) == []
    fetch.assert_not_awaited()
