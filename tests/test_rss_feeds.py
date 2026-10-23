"""RSS -> Discord posting (Zapier replacement) [CA][REH][IV]."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.rss import poller as poller_mod
from bot.rss.feeds import (
    DEFAULT_TEMPLATE,
    TRIGGER_CHANGED,
    FeedItem,
    FeedParseError,
    item_key,
    parse_feed,
    render,
    validate_template,
    zapier_template,
)
from bot.rss.poller import FeedConfig, FeedState, RssPoller, load_feed_configs


def _rss(*items: tuple[str, str]) -> bytes:
    body = "".join(
        f"<item><title>{t}</title><pubDate>Thu, 22 Oct 2026 13:01:44 GMT</pubDate><guid>{g}</guid>"
        f"<description>&lt;p&gt;Hello &lt;b&gt;there&lt;/b&gt;&lt;/p&gt;&lt;p&gt;Second&lt;/p&gt;</description>"
        f'<enclosure url="https://cdn.example/{g}.mp3" type="audio/mpeg"/><link>https://example.com/{g}</link></item>'
        for g, t in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>Pod</title>{body}</channel></rss>'.encode()


# -- parsing / formatting -----------------------------------------------------------


def test_parse_rss_items_and_html_description():
    title, items = parse_feed(_rss(("2", "Ep 2"), ("1", "Ep 1")))
    assert title == "Pod" and [i.title for i in items] == ["Ep 2", "Ep 1"]
    first = items[0]
    assert first.guid == "2" and first.enclosure_url == "https://cdn.example/2.mp3"
    assert first.description == "Hello there\nSecond"
    assert first.pub_date == "Thu, 22 Oct 2026 13:01:44 GMT"


def test_parse_atom():
    atom = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><title>Blog</title><entry><id>urn:1</id><title>Post</title>'
        b'<published>2026-10-01T00:00:00Z</published><link href="https://b.example/1"/>'
        b'<link rel="enclosure" href="https://b.example/1.mp3"/><summary>Hi</summary></entry></feed>'
    )
    title, items = parse_feed(atom)
    assert title == "Blog" and items[0].link == "https://b.example/1" and items[0].enclosure_url.endswith(".mp3")


@pytest.mark.parametrize("bad", [b"<html><body>nope</body></html>", b"not xml at all"])
def test_parse_rejects_non_feeds(bad):
    with pytest.raises(FeedParseError):
        parse_feed(bad)


def test_render_matches_zapier_template():
    item = FeedItem("Ep 1", "Thu, 22 Oct 2026", "Desc", "https://a/1.mp3", "https://a/1", "1")
    assert render(DEFAULT_TEMPLATE, item) == "**Ep 1**\u2014Thu, 22 Oct 2026\n\n```Desc```\nhttps://a/1.mp3"


def test_render_defuses_code_block_breakout_and_pings():
    item = FeedItem("@everyone ep", "d", "x``` @here", "", "", "1")
    text = render(DEFAULT_TEMPLATE, item)
    assert text.count("```") == 2 and "@everyone" not in text and "@here" not in text


def test_render_shortens_description_to_one_message():
    item = FeedItem("T", "d", "x" * 5000, "https://a/1.mp3", "", "1")
    text = render(DEFAULT_TEMPLATE, item)
    assert len(text) <= 2000 and text.endswith("```\nhttps://a/1.mp3") and "\u2026" in text


def test_item_key_trigger_modes():
    a = FeedItem("T", "d", "one", "", "", "g")
    b = FeedItem("T", "d", "two", "", "", "g")
    assert item_key(a) == item_key(b)
    assert item_key(a, TRIGGER_CHANGED) != item_key(b, TRIGGER_CHANGED)


def test_zapier_template_conversion():
    content = "**{{1__raw__title}}**\u2014{{1__raw__pubDate}}\n\n```{{1__description}}```\n{{1__raw__enclosure__@url}}"
    assert zapier_template(content) == DEFAULT_TEMPLATE
    assert zapier_template("{literal} {{3__raw__title}}") == "{{literal}} {title}"
    assert validate_template(DEFAULT_TEMPLATE) == [] and validate_template("{nope}") == ["nope"]


def test_importer_converts_export(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("imp", Path(__file__).resolve().parents[1] / "utils" / "import_zapier_rss.py")
    imp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imp)
    export = {
        "zaps": [
            {
                "title": "z",
                "nodes": {
                    "1": {"type_of": "read", "action": "new_feed", "params": {"url": "https://f/rss?auth=x", "trigger_style": "everything"}},
                    "2": {"type_of": "write", "action": "send_channel_message", "params": {"channel_id": "5", "content": "{{1__raw__title}}", "username": "Pod", "avatar_url": "https://i/a.png"}},
                },
            },
        ],
    }
    (feed,) = imp.convert(export, channel_override=99)
    assert feed == {"name": "Pod", "url": "https://f/rss?auth=x", "channel_id": 99, "username": "Pod", "avatar_url": "https://i/a.png", "template": "{title}", "trigger": "changed", "enabled": True}


def test_config_loader_skips_bad_entries(tmp_path):
    path = tmp_path / "feeds.json"
    path.write_text(
        json.dumps(
            {
                "feeds": [
                    {"url": "https://ok/rss", "channel_id": 1},
                    {"url": "file:///etc/passwd", "channel_id": 1},
                    {"url": "https://x/rss", "channel_id": 1, "template": "{bogus}"},
                    {"url": "https://off/rss", "channel_id": 1, "enabled": False},
                    {"channel_id": 1},
                ],
            },
        ),
    )
    assert [f.url for f in load_feed_configs(path)] == ["https://ok/rss"]
    assert load_feed_configs(tmp_path / "missing.json") == []


# -- poller -------------------------------------------------------------------------


class _Resp:
    """Body arrives in small chunks, like the network (a single read() is NOT the body)."""

    def __init__(self, status, body=b"", headers=None):
        self.status, self.headers = status, headers or {}

        def iter_chunked(_size):
            async def gen():
                for i in range(0, len(body), 37):
                    yield body[i : i + 37]

            return gen()

        self.content = SimpleNamespace(iter_chunked=iter_chunked)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def get(self, url, headers=None):
        self.requests.append(headers or {})
        return self.responses.pop(0)


def _poller(tmp_path, max_posts=5) -> RssPoller:
    bot = SimpleNamespace(user=SimpleNamespace(id=7))
    return RssPoller(bot, {"RSS_FEEDS_FILE": str(tmp_path / "f.json"), "RSS_STATE_FILE": str(tmp_path / "s.json"), "RSS_MAX_POSTS_PER_POLL": max_posts})


_FEED = FeedConfig(name="Pod", url="https://f/rss", channel_id=5, username="True Anon", avatar_url="https://i/a.png")


async def test_first_poll_seeds_without_posting_then_posts_only_new(tmp_path):
    poller = _poller(tmp_path)
    poller._post = AsyncMock()
    state = FeedState()
    await poller._poll_feed(_Session([_Resp(200, _rss(("2", "Ep 2"), ("1", "Ep 1")))]), _FEED, state)
    poller._post.assert_not_awaited()
    assert state.initialized and len(state.seen) == 2
    posted = await poller._poll_feed(_Session([_Resp(200, _rss(("4", "Ep 4"), ("3", "Ep 3"), ("2", "Ep 2")))]), _FEED, state)
    assert posted == 2
    assert [c.args[1].title for c in poller._post.await_args_list] == ["Ep 3", "Ep 4"]  # oldest first


async def test_failed_post_is_retried_next_poll(tmp_path):
    poller = _poller(tmp_path)
    state = FeedState(seen=[item_key(FeedItem("Ep 1", "", "", "", "", "1"))], initialized=True)
    poller._post = AsyncMock(side_effect=RuntimeError("discord down"))
    with pytest.raises(RuntimeError):
        await poller._poll_feed(_Session([_Resp(200, _rss(("2", "Ep 2"), ("1", "Ep 1")))]), _FEED, state)
    assert len(state.seen) == 1  # Ep 2 not marked seen
    poller._post = AsyncMock()
    assert await poller._poll_feed(_Session([_Resp(200, _rss(("2", "Ep 2"), ("1", "Ep 1")))]), _FEED, state) == 1


async def test_backlog_cap_posts_newest_and_skips_rest(tmp_path):
    poller = _poller(tmp_path, max_posts=2)
    poller._post = AsyncMock()
    state = FeedState(initialized=True)
    items = [(str(n), f"Ep {n}") for n in range(5, 0, -1)]
    assert await poller._poll_feed(_Session([_Resp(200, _rss(*items))]), _FEED, state) == 2
    assert [c.args[1].title for c in poller._post.await_args_list] == ["Ep 4", "Ep 5"]
    assert len(state.seen) == 5


async def test_conditional_get_and_304(tmp_path):
    poller = _poller(tmp_path)
    poller._post = AsyncMock()
    state = FeedState()
    session = _Session([_Resp(200, _rss(("1", "Ep 1")), {"ETag": '"abc"', "Last-Modified": "Wed"}), _Resp(304)])
    await poller._poll_feed(session, _FEED, state)
    assert await poller._poll_feed(session, _FEED, state) == 0
    assert session.requests[1] == {"If-None-Match": '"abc"', "If-Modified-Since": "Wed"}


async def test_post_via_webhook_with_feed_identity(tmp_path):
    poller = _poller(tmp_path)
    hook = SimpleNamespace(send=AsyncMock())
    poller._webhook_for = AsyncMock(return_value=hook)
    await poller._post(_FEED, FeedItem("Ep 1", "d", "desc", "https://a/1.mp3", "", "1"), "Pod")
    content = hook.send.await_args.args[0]
    kwargs = hook.send.await_args.kwargs
    assert content.startswith("**Ep 1**") and kwargs["username"] == "True Anon" and kwargs["avatar_url"] == "https://i/a.png"
    assert kwargs["allowed_mentions"].everyone is False and kwargs["wait"] is True


async def test_post_falls_back_to_bot_without_manage_webhooks(tmp_path, monkeypatch):
    import discord

    poller = _poller(tmp_path)
    channel = MagicMock()
    channel.webhooks = AsyncMock(side_effect=discord.Forbidden(MagicMock(status=403), "Missing Permissions"))
    poller.bot.get_channel = lambda _id: channel
    sent = AsyncMock()
    monkeypatch.setattr("bot.core.output.safe_send", sent)
    await poller._post(_FEED, FeedItem("Ep 1", "d", "desc", "", "", "1"), "Pod")
    assert sent.await_args.args[0] is channel and sent.await_args.args[1].startswith("**True Anon** \u00b7 **Ep 1**")
    assert 5 in poller._no_webhook_channels


async def test_webhook_reused_not_recreated(tmp_path):
    poller = _poller(tmp_path)
    mine = SimpleNamespace(name=poller_mod.WEBHOOK_NAME, user=SimpleNamespace(id=7))
    channel = SimpleNamespace(webhooks=AsyncMock(return_value=[SimpleNamespace(name="other", user=None), mine]), create_webhook=AsyncMock())
    poller.bot.get_channel = lambda _id: channel
    assert await poller._webhook_for(5) is mine
    assert await poller._webhook_for(5) is mine
    channel.create_webhook.assert_not_awaited()
    channel.webhooks.assert_awaited_once()


async def test_poll_once_persists_state_without_raw_url(tmp_path, monkeypatch):
    (tmp_path / "f.json").write_text(json.dumps({"feeds": [{"name": "Pod", "url": "https://f/rss?auth=SECRET", "channel_id": 5}]}))
    poller = _poller(tmp_path)

    class _FakeSession(_Session):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("aiohttp.ClientSession", lambda **kw: _FakeSession([_Resp(200, _rss(("1", "Ep 1")))]))
    assert await poller.poll_once() == 0
    saved = (tmp_path / "s.json").read_text()
    assert "SECRET" not in saved and '"initialized": true' in saved


async def test_oversized_feed_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(poller_mod, "MAX_FEED_BYTES", 100)
    with pytest.raises(RuntimeError, match="larger than"):
        await poller_mod._read_capped(_Resp(200, b"x" * 500))


# -- random_daily --------------------------------------------------------------------

from datetime import datetime as _dt  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

_RANDOM = FeedConfig(name="CumTown", url="https://c/rss", channel_id=5, mode="random_daily", post_at="12:00", timezone="Europe/London")
_LONDON = ZoneInfo("Europe/London")


def _at(hour, minute=0, day=23):
    return _dt(2026, 10, day, hour, minute, tzinfo=_LONDON)


async def test_random_daily_waits_for_post_time_then_posts_once_per_day(tmp_path):
    poller = _poller(tmp_path)
    poller._post = AsyncMock()
    state = FeedState()
    feed_xml = _rss(("3", "Ep 3"), ("2", "Ep 2"), ("1", "Ep 1"))
    assert await poller._random_daily(_Session([]), _RANDOM, state, now=_at(11, 59)) == 0  # not yet, no fetch
    assert await poller._random_daily(_Session([_Resp(200, feed_xml)]), _RANDOM, state, now=_at(12, 0)) == 1
    assert state.last_random_date == "2026-10-23" and len(state.random_used) == 1
    assert await poller._random_daily(_Session([]), _RANDOM, state, now=_at(18, 0)) == 0  # once a day
    assert await poller._random_daily(_Session([_Resp(200, feed_xml)]), _RANDOM, state, now=_at(12, 5, day=24)) == 1


async def test_random_daily_no_repeats_until_all_drawn_then_restarts(tmp_path):
    poller = _poller(tmp_path)
    poller._post = AsyncMock()
    state = FeedState()
    feed_xml = _rss(("3", "Ep 3"), ("2", "Ep 2"), ("1", "Ep 1"))
    for day in (1, 2, 3):
        await poller._random_daily(_Session([_Resp(200, feed_xml)]), _RANDOM, state, now=_at(13, day=day))
    titles = [c.args[1].title for c in poller._post.await_args_list]
    assert sorted(titles) == ["Ep 1", "Ep 2", "Ep 3"]  # each exactly once
    await poller._random_daily(_Session([_Resp(200, feed_xml)]), _RANDOM, state, now=_at(13, day=4))
    assert len(state.random_used) == 1  # new round


async def test_random_daily_ignores_etag_and_failed_post_retries(tmp_path):
    poller = _poller(tmp_path)
    poller._post = AsyncMock(side_effect=RuntimeError("discord down"))
    state = FeedState(etag='"abc"')
    session = _Session([_Resp(200, _rss(("1", "Ep 1")))])
    with pytest.raises(RuntimeError):
        await poller._random_daily(session, _RANDOM, state, now=_at(12, 30))
    assert "If-None-Match" not in session.requests[0]
    assert state.last_random_date == "" and state.random_used == []  # retried next poll


def test_random_daily_config_validation(tmp_path):
    path = tmp_path / "feeds.json"
    path.write_text(
        json.dumps(
            {
                "feeds": [
                    {"url": "https://ok/rss", "channel_id": 1, "mode": "random_daily", "post_at": "09:30"},
                    {"url": "https://a/rss", "channel_id": 1, "mode": "random_daily", "post_at": "25:00"},
                    {"url": "https://b/rss", "channel_id": 1, "mode": "random_daily", "timezone": "Mars/Base"},
                    {"url": "https://c/rss", "channel_id": 1, "mode": "weekly"},
                ],
            },
        ),
    )
    (feed,) = load_feed_configs(path)
    assert feed.mode == "random_daily" and feed.post_at == "09:30" and feed.timezone == "Europe/London"


@pytest.mark.parametrize("desc", ["", "   \n  "])
def test_render_drops_empty_description_block(desc):
    item = FeedItem("Ep 1", "Thu, 22 Oct 2026", desc, "https://a/1.mp3", "", "1")
    assert render(DEFAULT_TEMPLATE, item) == "**Ep 1**\u2014Thu, 22 Oct 2026\n\nhttps://a/1.mp3"
    multiline = "{title}\n\n```\n{description}\n```\n\n{link}"
    assert render(multiline, FeedItem("Ep 1", "", desc, "", "https://a/1", "1")) == "Ep 1\n\nhttps://a/1"


def test_render_keeps_block_when_description_present():
    item = FeedItem("Ep 1", "d", "text", "u", "", "1")
    assert "```text```" in render(DEFAULT_TEMPLATE, item)
