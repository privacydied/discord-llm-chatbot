"""Instagram media upload + card (bot/embed_fix/instagram.py + handler wiring)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.embed_fix import handler as h
from bot.embed_fix import instagram as ig

REEL = "https://www.instagram.com/reel/DTZPUbgkSzn/"
MB = 1024 * 1024
UPLOAD_ON = {"EMBED_FIX_UPLOAD_MEDIA": True}

# Trimmed copy of a real oginstagram reel page.
OG_REEL_HTML = """<html><head>
<meta property="og:title" content="tillykearns (@tillykearns)">
<meta property="og:image" content="https://www.oginstagram.com/offload/DZ/1?thumbnail=1&amp;sig=a">
<meta property="article:published_time" content="2026-06-10T13:37:33Z">
<meta property="og:image:alt" content="Unfortunate for whoever has to defend our protein queen">
<meta property="og:video" content="https://www.oginstagram.com/offload/DZ/1?v=2&amp;sig=b">
</head></html>"""


# ---- pure helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "code"),
    [(REEL, "DTZPUbgkSzn"), ("https://instagram.com/p/Ab-_1", "Ab-_1"), ("https://m.instagram.com/tv/X", "X"), ("https://www.instagram.com/some.user/p/Q1/", "Q1"), ("https://x.com/a", None)],
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


def test_og_post_meta_prefers_video_and_parses_author_time_caption():
    urls, meta = ig.og_post_meta(ig.parse_og(OG_REEL_HTML))
    assert urls == ["https://www.oginstagram.com/offload/DZ/1?v=2&sig=b"]  # &amp; unescaped
    assert meta.username == "tillykearns"
    assert meta.display_name == "tillykearns"
    assert meta.caption.startswith("Unfortunate")
    assert meta.published == datetime(2026, 6, 10, 13, 37, 33, tzinfo=UTC)


def test_og_post_meta_photo_and_garbage():
    urls, meta = ig.og_post_meta({"og:image": "https://og/1.jpg", "og:title": "Some Name (@some.user_1)"})
    assert urls == ["https://og/1.jpg"]
    assert (meta.display_name, meta.username) == ("Some Name", "some.user_1")
    urls, meta = ig.og_post_meta({"og:image": "http://insecure", "og:title": "nope", "article:published_time": "bad"})
    assert urls == [] and meta == ig.PostMeta()


def test_meta_merge_fills_gaps_only():
    a = ig.PostMeta(username="a", likes=5)
    b = ig.PostMeta(username="b", display_name="B", likes=9, caption="c")
    assert a.merged(b) == ig.PostMeta(username="a", display_name="B", caption="c", likes=5)


@pytest.mark.parametrize(("n", "text"), [(999, "999"), (1000, "1K"), (31419, "31.4K"), (2_500_000, "2.5M")])
def test_count(n, text):
    assert ig._count(n) == text


def _post(files, **meta):
    return ig.InstagramPost(url="https://www.instagram.com/reel/DZ/", files=files, meta=ig.PostMeta(**meta))


def test_build_card_video_post():
    when = datetime(2026, 6, 10, 13, 37, 33, tzinfo=UTC)
    card = ig.build_card(
        _post(
            [ig.MediaFile("instagram_DZ.mp4", b"v")],
            username="tillykearns",
            display_name="Tilly K",
            caption="hello",
            published=when,
            likes=31419,
            comments=12,
            avatar_url="https://og/avatar",
        )
    )
    assert card.author.name == "Tilly K (@tillykearns)"
    assert card.author.url == "https://www.instagram.com/tillykearns/"
    assert card.author.icon_url == "https://og/avatar"
    assert card.description == "hello"
    assert card.timestamp == when
    assert card.url == "https://www.instagram.com/reel/DZ/"
    assert card.footer.text == "Instagram  ❤️ 31.4K  💬 12"
    assert card.colour.value == ig.CARD_COLOUR
    assert card.image.url is None  # the video plays as the attachment


def test_build_card_photo_goes_inside_and_caption_is_capped():
    card = ig.build_card(_post([ig.MediaFile("instagram_DZ.jpg", b"i")], username="same", display_name="same", caption="x" * 2000))
    assert card.image.url == "attachment://instagram_DZ.jpg"
    assert card.author.name == "@same"
    assert len(card.description) == 600 and card.description.endswith("…")
    assert card.footer.text == "Instagram"


def test_build_card_without_metadata():
    card = ig.build_card(_post([ig.MediaFile("a.mp4", b"v")]))
    assert card.author.name is None and card.description is None and card.timestamp is None


def test_ytdlp_post_single_and_carousel(monkeypatch):
    import yt_dlp

    infos = iter(
        [
            {
                "url": "https://cdn/v.mp4",
                "channel": "tillykearns",
                "uploader": "TK",
                "timestamp": 1781098708,
                "like_count": 31419,
                "comment_count": 0,
                "description": "cap",
            },
            {"channel": "bad name!", "entries": [{"url": "https://cdn/1.mp4", "like_count": 3}, {"url": "http://insecure"}, None, {"url": "https://cdn/2.mp4"}]},
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
    urls, meta = ig._ytdlp_post(REEL)
    assert urls == ["https://cdn/v.mp4"]
    assert (meta.username, meta.display_name, meta.likes, meta.comments, meta.caption) == ("tillykearns", "TK", 31419, 0, "cap")
    assert meta.published == datetime.fromtimestamp(1781098708, tz=UTC)
    urls, meta = ig._ytdlp_post(REEL)
    assert urls == ["https://cdn/1.mp4", "https://cdn/2.mp4"]
    assert meta.username is None and meta.likes == 3


# ---- network pieces with a fake session ------------------------------------


class _Content:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_chunked(self, _size):
        for chunk in self._chunks:
            yield chunk

    async def read(self, _n):
        return b"".join(self._chunks)


class _Resp:
    def __init__(self, status=200, chunks=(b"x",), length=None, headers=None):
        self.status, self.content, self.content_length = status, _Content(chunks), length
        self.headers = headers if headers is not None else {"Content-Type": "video/mp4"}

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


async def test_kk_fallback_reads_location_without_following():
    session = _session(_Resp(status=302, headers={"Location": "https://cdn/p.jpg"}))
    assert await ig._from_kk(session, "AB", "p") == ["https://cdn/p.jpg"]
    assert session.get.call_args.args[0] == "https://www.kkinstagram.com/p/AB/"
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert await ig._from_kk(_session(_Resp(status=200, headers={})), "AB", "p") == []


async def test_og_page_and_avatar():
    session = _session(_Resp(chunks=(OG_REEL_HTML.encode(),)))
    urls, meta = await ig._from_og(session, "DZ", "reel")
    assert session.get.call_args.args[0] == "https://www.oginstagram.com/reel/DZ/"
    assert urls and meta.username == "tillykearns"
    avatar_html = b'<meta property="og:image" content="https://www.oginstagram.com/offload/@t/avatar?sig=1">'
    session = _session(_Resp(chunks=(avatar_html,)))
    assert await ig._avatar(session, "t") == "https://www.oginstagram.com/offload/@t/avatar?sig=1"
    assert session.get.call_args.args[0] == "https://www.oginstagram.com/t/"
    assert await ig._avatar(MagicMock(), "../etc") is None
    assert await ig._avatar(MagicMock(), None) is None


# ---- source order -----------------------------------------------------------


@pytest.fixture
def sources(monkeypatch):
    stubs = SimpleNamespace(
        ytdlp=AsyncMock(return_value=([], ig.PostMeta(likes=7))),
        og=AsyncMock(return_value=(["https://og/1.jpg"], ig.PostMeta(username="u", caption="c"))),
        kk=AsyncMock(return_value=["https://kk/1.jpg"]),
        avatar=AsyncMock(return_value="https://og/avatar"),
        download=AsyncMock(return_value=(b"img", "image/jpeg")),
    )
    monkeypatch.setattr(ig, "_from_ytdlp", stubs.ytdlp)
    monkeypatch.setattr(ig, "_from_og", stubs.og)
    monkeypatch.setattr(ig, "_from_kk", stubs.kk)
    monkeypatch.setattr(ig, "_avatar", stubs.avatar)
    monkeypatch.setattr(ig, "_download", stubs.download)
    return stubs


async def test_photo_post_uses_oginstagram_before_kk_and_merges_meta(sources):
    post = await ig.fetch_post("https://www.instagram.com/p/AB", max_bytes=MB)
    assert post.files == [ig.MediaFile("instagram_AB.jpg", b"img")]
    assert sources.download.await_args.args[1] == "https://og/1.jpg"
    sources.kk.assert_not_awaited()
    assert post.meta == ig.PostMeta(username="u", caption="c", likes=7, avatar_url="https://og/avatar")
    assert post.url == "https://www.instagram.com/p/AB/"


async def test_ytdlp_video_wins_over_og(sources):
    sources.ytdlp.return_value = (["https://cdn/v.mp4"], ig.PostMeta(username="y"))
    sources.download.return_value = (b"vid", "video/mp4")
    post = await ig.fetch_post(REEL, max_bytes=MB)
    assert [c.args[1] for c in sources.download.await_args_list] == ["https://cdn/v.mp4"]
    assert post.files[0].filename == "instagram_DTZPUbgkSzn.mp4"
    assert post.meta.username == "y" and post.meta.caption == "c"


async def test_kk_is_last_resort(sources):
    sources.og.return_value = ([], ig.PostMeta())
    post = await ig.fetch_post("https://www.instagram.com/p/AB", max_bytes=MB)
    assert sources.download.await_args.args[1] == "https://kk/1.jpg"
    assert post is not None


async def test_og_download_failure_falls_through_to_kk(sources):
    sources.download.side_effect = [None, (b"img", "image/jpeg")]
    post = await ig.fetch_post("https://www.instagram.com/p/AB", max_bytes=MB)
    assert [c.args[1] for c in sources.download.await_args_list] == ["https://og/1.jpg", "https://kk/1.jpg"]
    assert post.files


async def test_nothing_downloadable_returns_none(sources):
    sources.download.return_value = None
    assert await ig.fetch_post("https://www.instagram.com/p/AB", max_bytes=MB) is None
    sources.avatar.assert_not_awaited()


async def test_carousel_numbers_files_and_shares_budget(sources):
    sources.ytdlp.return_value = (["https://cdn/1.mp4", "https://cdn/2.mp4"], ig.PostMeta())
    sources.download.side_effect = [(b"1234", "video/mp4"), None]
    post = await ig.fetch_post(REEL, max_bytes=10)
    assert [f.filename for f in post.files] == ["instagram_DTZPUbgkSzn_1.mp4"]
    assert sources.download.await_args_list[1].args[2] == 6  # budget minus first file


async def test_fetch_post_times_out_to_none(monkeypatch):
    async def slow(*_a, **_k):
        await asyncio.sleep(5)

    monkeypatch.setattr(ig, "_collect", slow)
    assert await ig.fetch_post(REEL, max_bytes=MB, timeout_s=0.01) is None


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
    monkeypatch.setattr(h.instagram, "has_link_video", AsyncMock(return_value=False))
    return mock


def _ig_post(*files, username="u"):
    return ig.InstagramPost(url="https://www.instagram.com/reel/AB/", files=list(files), meta=ig.PostMeta(username=username))


async def test_instagram_media_and_card_replace_link(reply, monkeypatch):
    monkeypatch.setattr(h.instagram, "fetch_post", AsyncMock(return_value=_ig_post(ig.MediaFile("instagram_AB.mp4", b"vid"))))
    msg = _message("look https://x.com/a/status/1 https://www.instagram.com/reel/AB/")
    sent = await h.maybe_fix_embeds(msg, UPLOAD_ON)
    args, kwargs = reply.await_args
    assert args[1] == "https://vxtwitter.com/a/status/1"
    (file,) = kwargs["files"]
    assert file.filename == "instagram_AB.mp4"
    (card,) = kwargs["embeds"]
    assert card.author.name == "@u"
    assert [f.site for f in sent] == ["twitter", "instagram"]
    assert h.instagram.fetch_post.await_args.kwargs["max_bytes"] == ig.upload_budget(25 * MB)
    msg.edit.assert_awaited_once_with(suppress=True)


async def test_failed_download_falls_back_to_oginstagram_link(reply, monkeypatch):
    monkeypatch.setattr(h.instagram, "fetch_post", AsyncMock(return_value=None))
    await h.maybe_fix_embeds(_message("https://www.instagram.com/reel/AB/"), UPLOAD_ON)
    args, kwargs = reply.await_args
    assert args[1] == "https://www.oginstagram.com/reel/AB/"
    assert kwargs["files"] is None and kwargs["embeds"] is None


@pytest.mark.parametrize(("msg_kwargs", "config"), [({"attach_files": False}, UPLOAD_ON), ({}, {})])
async def test_no_download_without_attach_files_or_when_disabled(reply, monkeypatch, msg_kwargs, config):
    monkeypatch.setattr(h.instagram, "fetch_post", AsyncMock())
    await h.maybe_fix_embeds(_message("https://www.instagram.com/reel/AB/", **msg_kwargs), config)
    h.instagram.fetch_post.assert_not_awaited()
    assert reply.await_args.args[1] == "https://www.oginstagram.com/reel/AB/"


async def test_multiple_posts_share_budget_and_get_one_card_each(reply, monkeypatch):
    monkeypatch.setattr(
        h.instagram,
        "fetch_post",
        AsyncMock(side_effect=[_ig_post(ig.MediaFile("a.mp4", b"x" * 100)), _ig_post(ig.MediaFile("b.jpg", b"y"), username="v")]),
    )
    msg = _message("https://www.instagram.com/reel/A/ https://www.instagram.com/p/B/", filesize_limit=10 * MB)
    await h.maybe_fix_embeds(msg, UPLOAD_ON)
    budgets = [c.kwargs["max_bytes"] for c in h.instagram.fetch_post.await_args_list]
    assert budgets == [ig.upload_budget(10 * MB), ig.upload_budget(10 * MB) - 100]
    kwargs = reply.await_args.kwargs
    assert [f.filename for f in kwargs["files"]] == ["a.mp4", "b.jpg"]
    assert [c.author.name for c in kwargs["embeds"]] == ["@u", "@v"]
    assert kwargs["embeds"][1].image.url == "attachment://b.jpg"
    assert reply.await_args.args[1] is None


async def test_video_post_with_og_video_goes_out_as_link_for_discord_unfurl(reply, monkeypatch):
    h.instagram.has_link_video.return_value = True
    monkeypatch.setattr(h.instagram, "fetch_post", AsyncMock())
    await h.maybe_fix_embeds(_message("https://www.instagram.com/reel/AB/"), UPLOAD_ON)
    h.instagram.fetch_post.assert_not_awaited()
    args, kwargs = reply.await_args
    assert args[1] == "https://www.oginstagram.com/reel/AB/"
    assert kwargs["files"] is None and kwargs["embeds"] is None


async def test_link_video_check_skipped_when_link_host_overridden(reply, monkeypatch):
    h.instagram.has_link_video.return_value = True
    monkeypatch.setattr(h.instagram, "fetch_post", AsyncMock(return_value=_ig_post(ig.MediaFile("a.mp4", b"v"))))
    await h.maybe_fix_embeds(_message("https://www.instagram.com/reel/AB/"), {**UPLOAD_ON, "EMBED_FIX_DOMAIN_OVERRIDES": "instagram=www.kkinstagram.com"})
    h.instagram.has_link_video.assert_not_awaited()
    assert [f.filename for f in reply.await_args.kwargs["files"]] == ["a.mp4"]


@pytest.mark.parametrize(
    ("html", "expected"),
    [(OG_REEL_HTML, True), ('<meta property="og:image" content="https://og/1.jpg">', False), (None, False)],
)
async def test_has_link_video(monkeypatch, html, expected):
    get_text = AsyncMock(return_value=html)
    monkeypatch.setattr(ig, "_get_text", get_text)
    assert await ig.has_link_video("https://www.instagram.com/reels/AB/") is expected
    assert get_text.await_args.args[1] == "https://www.oginstagram.com/reel/AB/"
    assert await ig.has_link_video("https://x.com/a") is False
