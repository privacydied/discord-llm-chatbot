"""Instagram media upload (bot/embed_fix/instagram.py + handler wiring)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.embed_fix import handler as h
from bot.embed_fix import instagram as ig

REEL = "https://www.instagram.com/reel/DTZPUbgkSzn/"
MB = 1024 * 1024


# ---- module ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "code"),
    [(REEL, "DTZPUbgkSzn"), ("https://instagram.com/p/Ab-_1", "Ab-_1"), ("https://m.instagram.com/tv/X", "X"), ("https://x.com/a", None)],
)
def test_shortcode(url, code):
    assert ig.shortcode(url) == code


def test_upload_budget_leaves_headroom_and_defaults_to_10mb():
    assert ig.upload_budget(25 * MB) < 25 * MB
    assert ig.upload_budget(None) == ig.upload_budget(10 * MB) > 9 * MB
    assert ig.upload_budget(1) == 0


@pytest.mark.parametrize(
    ("url", "ctype", "ext"),
    [
        ("https://cdn/x.mp4?a=1", "", "mp4"),
        ("https://cdn/x.jpg", "", "jpg"),
        ("https://cdn/x", "image/jpeg; charset=x", "jpg"),
        ("https://cdn/x.exe", "", "mp4"),
        ("https://cdn/x.jpg", "video/mp4", "mp4"),
    ],
)
def test_extension(url, ctype, ext):
    assert ig._extension(url, ctype) == ext


class _Content:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_chunked(self, _size):
        for chunk in self._chunks:
            yield chunk


class _Resp:
    def __init__(self, status=200, chunks=(b"x",), length=None, headers=None):
        self.status, self.content, self.content_length = status, _Content(chunks), length
        self.headers = headers or {"Content-Type": "video/mp4"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _session(*responses):
    session = MagicMock()
    session.get = MagicMock(side_effect=list(responses))
    return session


async def test_download_streams_within_cap():
    got = await ig._download(_session(_Resp(chunks=(b"ab", b"cd"))), "https://cdn/v.mp4", 10)
    assert got == (b"abcd", "video/mp4")


@pytest.mark.parametrize("resp", [_Resp(chunks=(b"abc", b"def")), _Resp(length=999), _Resp(status=403)])
async def test_download_rejects_oversize_and_errors(resp):
    assert await ig._download(_session(resp), "https://cdn/v.mp4", 5) is None


async def test_proxy_fallback_reads_location_without_following():
    session = _session(_Resp(status=302, headers={"Location": "https://cdn/p.jpg"}))
    assert await ig._resolve_via_proxy(session, "AB", "p") == ["https://cdn/p.jpg"]
    url = session.get.call_args.args[0]
    assert url == "https://www.kkinstagram.com/p/AB/"
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert await ig._resolve_via_proxy(_session(_Resp(status=200, headers={})), "AB", "p") == []


async def test_collect_prefers_ytdlp_then_proxy(monkeypatch):
    monkeypatch.setattr(ig, "_resolve_via_ytdlp", AsyncMock(return_value=[]))
    monkeypatch.setattr(ig, "_resolve_via_proxy", AsyncMock(return_value=["https://cdn/p.jpg"]))
    monkeypatch.setattr(ig, "_download", AsyncMock(return_value=(b"jpg", "image/jpeg")))
    files = await ig.fetch_media("https://www.instagram.com/p/AB", max_bytes=MB)
    assert files == [ig.MediaFile("instagram_AB.jpg", b"jpg")]
    assert ig._resolve_via_proxy.await_args.args[1:] == ("AB", "p")


async def test_collect_carousel_numbers_files_and_shares_budget(monkeypatch):
    monkeypatch.setattr(ig, "_resolve_via_ytdlp", AsyncMock(return_value=["https://cdn/1.mp4", "https://cdn/2.mp4"]))
    monkeypatch.setattr(ig, "_resolve_via_proxy", AsyncMock())
    monkeypatch.setattr(ig, "_download", AsyncMock(side_effect=[(b"1234", "video/mp4"), None]))
    files = await ig.fetch_media(REEL, max_bytes=10)
    assert [f.filename for f in files] == ["instagram_DTZPUbgkSzn_1.mp4"]
    assert ig._download.await_args_list[1].args[2] == 6  # budget minus first file
    ig._resolve_via_proxy.assert_not_awaited()


async def test_fetch_media_times_out_to_empty(monkeypatch):
    async def slow(*_a, **_k):
        import asyncio

        await asyncio.sleep(5)

    monkeypatch.setattr(ig, "_collect", slow)
    assert await ig.fetch_media(REEL, max_bytes=MB, timeout_s=0.01) == []


def test_ytdlp_urls_handles_single_and_carousel(monkeypatch):
    import yt_dlp

    infos = iter(
        [
            {"url": "https://cdn/v.mp4"},
            {"entries": [{"url": "https://cdn/1.mp4"}, {"url": "http://insecure"}, None, {"url": "https://cdn/2.mp4"}]},
        ]
    )

    class FakeYDL:
        def __init__(self, opts):
            assert opts["format"] == "b[ext=mp4]/b"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            assert download is False
            return next(infos)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    assert ig._ytdlp_urls(REEL) == ["https://cdn/v.mp4"]
    assert ig._ytdlp_urls(REEL) == ["https://cdn/1.mp4", "https://cdn/2.mp4"]


# ---- handler --------------------------------------------------------------


def _message(content, *, attach_files=True, filesize_limit=25 * MB):
    perms = discord.Permissions(send_messages=True, embed_links=True, read_message_history=True, manage_messages=True, attach_files=attach_files)
    channel = MagicMock()
    channel.permissions_for = MagicMock(return_value=perms)
    channel.is_nsfw = MagicMock(return_value=False)
    msg = MagicMock(id=1, content=content, channel=channel)
    msg.author = SimpleNamespace(bot=False, id=2)
    msg.guild = SimpleNamespace(id=3, me=SimpleNamespace(id=4), filesize_limit=filesize_limit)
    msg.edit = AsyncMock()
    return msg


@pytest.fixture
def reply(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(h, "safe_reply", mock)
    monkeypatch.setattr(h, "is_server_feature_enabled", lambda guild_id, feature: True)
    monkeypatch.setattr(h, "SUPPRESS_RECHECK_DELAYS_S", ())
    monkeypatch.setattr(h.translate, "fetch_lang", AsyncMock(return_value="en"))
    return mock


async def test_instagram_media_is_attached_instead_of_link(reply, monkeypatch):
    monkeypatch.setattr(h.instagram, "fetch_media", AsyncMock(return_value=[ig.MediaFile("instagram_AB.mp4", b"vid")]))
    msg = _message("look https://x.com/a/status/1 https://www.instagram.com/reel/AB/")
    sent = await h.maybe_fix_embeds(msg, {})
    args, kwargs = reply.await_args
    assert args[1] == "https://vxtwitter.com/a/status/1"
    (file,) = kwargs["files"]
    assert file.filename == "instagram_AB.mp4"
    assert [f.site for f in sent] == ["twitter", "instagram"]
    budget = h.instagram.fetch_media.await_args.kwargs["max_bytes"]
    assert budget == ig.upload_budget(25 * MB)
    msg.edit.assert_awaited_once_with(suppress=True)


async def test_instagram_only_post_sends_file_without_text(reply, monkeypatch):
    monkeypatch.setattr(h.instagram, "fetch_media", AsyncMock(return_value=[ig.MediaFile("instagram_AB.jpg", b"img")]))
    await h.maybe_fix_embeds(_message("https://www.instagram.com/p/AB/"), {})
    args, kwargs = reply.await_args
    assert args[1] is None
    assert len(kwargs["files"]) == 1


async def test_failed_download_falls_back_to_proxy_link(reply, monkeypatch):
    monkeypatch.setattr(h.instagram, "fetch_media", AsyncMock(return_value=[]))
    await h.maybe_fix_embeds(_message("https://www.instagram.com/reel/AB/"), {})
    args, kwargs = reply.await_args
    assert args[1] == "https://www.kkinstagram.com/reel/AB/"
    assert kwargs["files"] is None


@pytest.mark.parametrize(
    ("msg_kwargs", "config"),
    [({"attach_files": False}, {}), ({}, {"EMBED_FIX_UPLOAD_MEDIA": False})],
)
async def test_no_download_without_attach_files_or_when_disabled(reply, monkeypatch, msg_kwargs, config):
    monkeypatch.setattr(h.instagram, "fetch_media", AsyncMock(return_value=[ig.MediaFile("a.mp4", b"x")]))
    await h.maybe_fix_embeds(_message("https://www.instagram.com/reel/AB/", **msg_kwargs), config)
    h.instagram.fetch_media.assert_not_awaited()
    assert reply.await_args.args[1] == "https://www.kkinstagram.com/reel/AB/"


async def test_multiple_posts_share_budget_and_attachment_slots(reply, monkeypatch):
    monkeypatch.setattr(
        h.instagram,
        "fetch_media",
        AsyncMock(side_effect=[[ig.MediaFile("a.mp4", b"x" * 100)], [ig.MediaFile("b.mp4", b"y")]]),
    )
    msg = _message("https://www.instagram.com/reel/A/ https://www.instagram.com/reel/B/", filesize_limit=10 * MB)
    await h.maybe_fix_embeds(msg, {})
    budgets = [c.kwargs["max_bytes"] for c in h.instagram.fetch_media.await_args_list]
    assert budgets == [ig.upload_budget(10 * MB), ig.upload_budget(10 * MB) - 100]
    assert [f.filename for f in reply.await_args.kwargs["files"]] == ["a.mp4", "b.mp4"]
    assert reply.await_args.args[1] is None
