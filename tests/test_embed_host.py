"""Bot-hosted embed pages (bot/embed_host) + the embed-fix wiring that uses them."""

from __future__ import annotations

import os
import struct
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from aiohttp.test_utils import TestClient, TestServer

from bot.embed_fix import handler as h
from bot.embed_fix import hosted
from bot.embed_fix import instagram as ig
from bot.embed_host import server as srv
from bot.embed_host.store import EmbedStore, NewMedia, StoredPost, image_dimensions, mp4_dimensions

PUB = "https://embed.example"
DISCORD_UA = {"User-Agent": "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)"}


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _mp4(width: int, height: int) -> bytes:
    tkhd = _box(b"tkhd", b"\0" * 76 + struct.pack(">II", width << 16, height << 16))
    return _box(b"ftyp", b"isom") + _box(b"moov", _box(b"trak", tkhd))


PNG = b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 640, 1136) + b"\0" * 8


def _post(**kw) -> StoredPost:
    base = dict(
        token="",
        source_url="https://www.instagram.com/reel/AB/",
        site_name="Instagram",
        author_name="Tilly K",
        author_handle="tilly",
        author_url="https://www.instagram.com/tilly/",
        text='hi <b>"there"</b>',
        stats="❤️ 1K",
        published="2026-06-10T13:38:28Z",
    )
    return StoredPost(**{**base, **kw})


@pytest.fixture
def host(tmp_path):
    settings = srv.EmbedHostSettings(enabled=True, public_url=PUB, root=str(tmp_path))
    return srv.EmbedHost(settings)


# ---- store -------------------------------------------------------------------


def test_dimensions():
    assert mp4_dimensions(_mp4(720, 1280)) == (720, 1280)
    assert mp4_dimensions(b"garbage") is None
    assert image_dimensions(PNG) == (640, 1136)
    jpeg = b"\xff\xd8" + b"\xff\xe0\x00\x04ab" + b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", 829, 900) + b"\0" * 8
    assert image_dimensions(jpeg) == (900, 829)


def test_store_roundtrip_and_path_safety(tmp_path):
    store = EmbedStore(tmp_path)
    post = store.put(_post(), [NewMedia("x.mp4", _mp4(720, 1280))], NewMedia("t.png", PNG), NewMedia("a.jpg", b"\xff\xd8"))
    assert len(post.token) >= 16
    got = store.get(post.token)
    assert [(m.name, m.mime, m.width, m.height) for m in got.media] == [("1.mp4", "video/mp4", 720, 1280)]
    assert got.thumbnail.name == "thumb.png" and got.avatar.name == "avatar.jpg"
    assert store.media_path(post.token, "1.mp4").is_file()
    for bad in ("post.json", "../x.mp4", "1.MP4", ""):
        assert store.media_path(post.token, bad) is None
    for bad in ("..", "short", "a/b" * 8, ""):
        assert store.get(bad) is None


def test_store_ttl_and_prune(tmp_path):
    store = EmbedStore(tmp_path, ttl_days=1, max_total_bytes=10**6)
    old = store.put(_post(), [NewMedia("a.jpg", b"x" * 100)])
    new = store.put(_post(), [NewMedia("b.jpg", b"y" * 100)])
    newest = store.put(_post(), [NewMedia("c.jpg", b"z" * 100)])
    folder = tmp_path / old.token
    stamp = time.time() - 3 * 86400
    os.utime(folder, (stamp, stamp))
    os.utime(tmp_path / new.token, (stamp + 86400 * 2.5, stamp + 86400 * 2.5))
    assert store.prune() == 1  # TTL
    assert not folder.exists() and (tmp_path / new.token).exists()
    one_post = sum(f.stat().st_size for f in (tmp_path / newest.token).iterdir())
    store.max_total_bytes = one_post  # size cap: oldest goes first
    assert store.prune() == 1
    assert not (tmp_path / new.token).exists() and (tmp_path / newest.token).exists()
    expired = store.put(_post(created=time.time() - 2 * 86400), [NewMedia("c.jpg", b"z")])
    assert store.get(expired.token) is None


# ---- server ------------------------------------------------------------------


async def test_routes_serve_og_page_activitypub_and_media(host):
    post = host.store.put(
        _post(),
        [NewMedia("v.mp4", _mp4(720, 1280))],
        NewMedia("thumb.png", PNG),
        NewMedia("a.jpg", b"\xff\xd8avatar"),
    )
    token = post.token
    async with TestClient(TestServer(host.app())) as client:
        resp = await client.get(f"/p/{token}", headers=DISCORD_UA)
        page = await resp.text()
        assert resp.status == 200
        assert f'<meta property="og:video" content="{PUB}/m/{token}/1.mp4">' in page
        assert '<meta property="og:video:type" content="video/mp4">' in page
        assert '<meta property="og:video:width" content="720">' in page
        assert f'<meta property="og:image" content="{PUB}/m/{token}/thumb.png">' in page
        assert '<meta property="og:title" content="Tilly K (@tilly)">' in page
        assert "&lt;b&gt;&quot;there&quot;" in page  # escaped caption
        assert f'type="application/activity+json" href="{PUB}/users/{token}/statuses/{token}"' in page

        resp = await client.get(f"/p/{token}", allow_redirects=False)
        assert resp.status == 302 and resp.headers["Location"] == "https://www.instagram.com/reel/AB/"

        resp = await client.get(f"/users/{token}/statuses/{token}")
        note = await resp.json(content_type=None)
        assert resp.content_type == "application/activity+json"
        assert note["attributedTo"] == f"{PUB}/users/{token}"
        assert note["attachment"] == [{"type": "Document", "mediaType": "video/mp4", "url": f"{PUB}/m/{token}/1.mp4", "width": 720, "height": 1280}]
        assert note["content"] == "<p><b>❤️ 1K</b></p><p>hi &lt;b&gt;&quot;there&quot;&lt;/b&gt;</p>"
        assert note["published"] == "2026-06-10T13:38:28Z"

        actor = await (await client.get(f"/users/{token}")).json(content_type=None)
        assert (actor["name"], actor["preferredUsername"]) == ("Tilly K", "tilly")
        assert actor["icon"]["url"] == f"{PUB}/m/{token}/avatar.jpg"

        resp = await client.get(f"/m/{token}/1.mp4", headers={"Range": "bytes=0-3"})
        assert resp.status == 206 and resp.headers["Content-Type"] == "video/mp4"

        for path in (f"/p/{'A' * 22}", f"/m/{token}/post.json", f"/users/{token}/statuses/{'B' * 22}", "/p/.."):
            assert (await client.get(path, headers=DISCORD_UA)).status == 404

        assert await (await client.get("/healthz")).text() == host.nonce


async def test_photo_page_has_image_but_no_video(host):
    post = host.store.put(_post(text=None, avatar_url="https://cdn/avatar"), [NewMedia("p.png", PNG)])
    page = host.render_page(host.store.get(post.token))
    assert "og:video" not in page and "og:description" not in page
    assert f'<meta property="og:image" content="{PUB}/m/{post.token}/1.png">' in page
    assert host.actor_json(host.store.get(post.token))["icon"]["url"] == "https://cdn/avatar"


class _Resp:
    def __init__(self, status, text):
        self.status, self._text = status, text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._text


class _Session:
    def __init__(self, resp):
        self.resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url):
        assert url == f"{PUB}/healthz"
        return self.resp


@pytest.mark.parametrize(("status", "body", "ready"), [(200, "NONCE", True), (200, "someone-else", False), (502, "NONCE", False)])
async def test_check_public_requires_own_nonce(host, monkeypatch, status, body, ready):
    host.nonce = "NONCE"
    monkeypatch.setattr(srv.aiohttp, "ClientSession", lambda **_: _Session(_Resp(status, body)))
    assert await host.check_public() is ready
    assert host.ready is ready


async def test_check_public_rejects_non_https(tmp_path):
    host = srv.EmbedHost(srv.EmbedHostSettings(enabled=True, public_url="http://plain", root=str(tmp_path)))
    assert await host.check_public() is False


def test_settings_from_config():
    s = srv.EmbedHostSettings.from_config({"EMBED_HOST_ENABLED": True, "EMBED_HOST_PUBLIC_URL": "https://x/", "EMBED_HOST_MAX_FILE_MB": 2})
    assert (s.enabled, s.public_url, s.port, s.max_file_bytes) == (True, "https://x", 8012, 2 * 1024 * 1024)
    assert srv.EmbedHostSettings.from_config({}).enabled is False


async def test_start_disabled_returns_none():
    assert await srv.start_embed_host({}) is None
    assert srv.get_embed_host() is None


# ---- publishing ------------------------------------------------------------


async def test_publish_instagram_stores_post_and_returns_page(host, monkeypatch):
    meta = ig.PostMeta(
        username="tilly",
        display_name="TK",
        caption="cap",
        likes=31419,
        comments=0,
        published=ig.datetime(2026, 6, 10, 13, 38, 28, tzinfo=ig.UTC),
        avatar_url="https://cdn/av",
        thumbnail_url="https://cdn/th",
    )
    monkeypatch.setattr(ig, "fetch_post", AsyncMock(return_value=ig.InstagramPost("https://www.instagram.com/reel/AB/", [ig.MediaFile("instagram_AB.mp4", _mp4(720, 1280))], meta)))
    thumbs = {"https://cdn/th": ig.MediaFile("thumb.png", PNG), "https://cdn/av": ig.MediaFile("thumb.jpg", b"\xff\xd8")}
    monkeypatch.setattr(ig, "fetch_thumbnail", AsyncMock(side_effect=lambda url, **_: thumbs.get(url)))
    url = await hosted.publish_instagram(host, "https://www.instagram.com/reel/AB/")
    assert url.startswith(f"{PUB}/p/") and host.owns(url)
    stored = host.store.get(url.rsplit("/", 1)[1])
    assert (stored.author_name, stored.author_handle, stored.text, stored.stats) == ("TK", "tilly", "cap", "❤️ 31.4K")
    assert stored.published == "2026-06-10T13:38:28Z"
    assert stored.thumbnail.name == "thumb.png" and stored.avatar.name == "avatar.jpg"
    assert ig.fetch_post.await_args.kwargs["max_bytes"] == host.settings.max_file_bytes


async def test_publish_instagram_none_when_nothing_downloaded(host, monkeypatch):
    monkeypatch.setattr(ig, "fetch_post", AsyncMock(return_value=None))
    assert await hosted.publish_instagram(host, "https://www.instagram.com/reel/AB/") is None


# ---- handler -------------------------------------------------------------------


def _message(content):
    perms = discord.Permissions(send_messages=True, embed_links=True, read_message_history=True, manage_messages=True, attach_files=True)
    channel = MagicMock()
    channel.permissions_for = MagicMock(return_value=perms)
    channel.is_nsfw = MagicMock(return_value=False)
    msg = MagicMock(id=1, content=content, channel=channel)
    msg.author = SimpleNamespace(bot=False, id=2)
    msg.guild = SimpleNamespace(id=3, me=SimpleNamespace(id=4), filesize_limit=25 * 1024 * 1024)
    msg.edit = AsyncMock()
    return msg


@pytest.fixture
def reply(monkeypatch, host):
    mock = AsyncMock()
    monkeypatch.setattr(h, "safe_reply", mock)
    monkeypatch.setattr(h, "is_server_feature_enabled", lambda guild_id, feature: True)
    monkeypatch.setattr(h, "SUPPRESS_RECHECK_DELAYS_S", ())
    monkeypatch.setattr(h.translate, "fetch_lang", AsyncMock(return_value="en"))
    monkeypatch.setattr(h.instagram, "has_link_video", AsyncMock(return_value=False))
    monkeypatch.setattr(h.instagram, "fetch_post", AsyncMock(return_value=None))
    monkeypatch.setattr(h, "get_embed_host", lambda: host)
    host.ready = True
    return mock


async def test_ready_host_replaces_instagram_link_with_own_page(reply, monkeypatch):
    monkeypatch.setattr(hosted, "publish_instagram", AsyncMock(return_value=f"{PUB}/p/TOKENTOKENTOKEN123"))
    msg = _message("https://x.com/a/status/1 https://www.instagram.com/reel/AB/")
    sent = await h.maybe_fix_embeds(msg, {})
    args, kwargs = reply.await_args
    assert args[1] == f"https://vxtwitter.com/a/status/1\n{PUB}/p/TOKENTOKENTOKEN123"
    assert kwargs["files"] is None and kwargs["embeds"] is None
    h.instagram.fetch_post.assert_not_awaited()  # no upload path
    assert [f.site for f in sent] == ["twitter", "instagram"]
    msg.edit.assert_awaited_once_with(suppress=True)


@pytest.mark.parametrize("failure", [AsyncMock(return_value=None), AsyncMock(side_effect=RuntimeError("boom"))])
async def test_hosting_failure_falls_back_to_upload_path(reply, monkeypatch, failure):
    monkeypatch.setattr(hosted, "publish_instagram", failure)
    h.instagram.fetch_post.return_value = ig.InstagramPost("https://www.instagram.com/reel/AB/", [ig.MediaFile("a.mp4", b"v")])
    await h.maybe_fix_embeds(_message("https://www.instagram.com/reel/AB/"), {})
    assert [f.filename for f in reply.await_args.kwargs["files"]] == ["a.mp4"]


async def test_unready_host_is_not_used(reply, monkeypatch, host):
    host.ready = False
    monkeypatch.setattr(hosted, "publish_instagram", AsyncMock())
    await h.maybe_fix_embeds(_message("https://www.instagram.com/reel/AB/"), {})
    hosted.publish_instagram.assert_not_awaited()
    assert reply.await_args.args[1] == "https://www.oginstagram.com/reel/AB/"
