"""X thread unroll from X's own logged-out status page (bot/threads/x_thread_page.py)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from bot.threads import x_thread_page as xt

AUTHOR = "anagenxyz"


def _tweet(tid: str, text: str, *, handle: str = AUTHOR, reply_to: str | None = None, media: int = 0, urls=None, note: str | None = None) -> dict:
    result = {
        "__typename": "Tweet",
        "rest_id": tid,
        "core": {"user_results": {"result": {"core": {"screen_name": handle, "name": handle.title()}}}},
        "details": {"full_text": text, "created_at_ms": "1794065520000"},
        "url_entities": urls or [],
        "media_entities2": [{"type": "photo"} for _ in range(media)],
    }
    if reply_to:
        result["reply_to_results"] = {"rest_id": reply_to}
    if note:
        result["note_tweet"] = {"note_tweet_results": {"result": {"text": note}}}
    return {"tweet_results": {"result": result}}


def _item(tid: str, content: dict, display: str = "SelfThread") -> dict:
    return {"entry_id": f"m-tweet-{tid}", "item": {"content": {"__typename": "TimelineTweet", "display_type": display, **content}}}


def _page(entries: list[dict]) -> str:
    """Wrap a TweetDetail payload the way x.com serializes it into the HTML."""
    payload = {"data": {"threaded_conversation_with_injections_v2": {"instructions": [{"__typename": "TimelineAddEntries", "entries": entries}]}}}
    literal = json.dumps(json.dumps(payload)).replace("<", "\\x3C")
    return f'<html><script nonce="n">($R=>$R[1]={{kind:1,result:$R[2]={{kind:"Result.Ok",value:{literal}}}}})($R["tsr"])</script></html>'


def _single(tid: str, content: dict) -> dict:
    return {"entry_id": f"tweet-{tid}", "content": {"__typename": "TimelineTimelineItem", "content": {"__typename": "TimelineTweet", **content}}}


def _module(items: list[dict]) -> dict:
    return {"entry_id": "conversationthread-x", "content": {"__typename": "TimelineTimelineModule", "items": items}}


THREAD = [
    _tweet("1", "Story of RU58841 🧵 https://t.co/media1", media=1),
    _tweet("2", "Part two (2/4)", reply_to="1"),
    _tweet("3", "Part three, see https://t.co/abc (3/4)", reply_to="2", urls=[{"url": "https://t.co/abc", "expanded_url": "https://example.org/paper"}]),
    _tweet("4", "short", reply_to="3", note="Full long note text <b> (4/4)"),
]
OTHER_REPLY = _tweet("9", "nice thread", handle="someone", reply_to="1")


def _page_for_first() -> str:
    return _page([_single("1", THREAD[0]), _module([_item(t, c) for t, c in zip("234", THREAD[1:], strict=True)]), _module([_item("9", OTHER_REPLY, "Tweet")])])


def test_parses_serialized_page_into_posts():
    posts, more = xt.page_posts(_page_for_first())
    assert [p.id for p in posts] == ["1", "2", "3", "4", "9"] and not more
    first, _, third, fourth, other = posts
    assert first.text == "Story of RU58841 🧵" and first.media == ["photo"]  # trailing media link removed
    assert third.text == "Part three, see https://example.org/paper (3/4)"
    assert fourth.text == "Full long note text <b> (4/4)"  # long-post note wins; \x3C decoded
    assert other.handle == "someone" and not other.self_thread


def test_chain_from_first_post_skips_other_people():
    chain = xt.assemble("1", xt.page_posts(_page_for_first())[0])
    assert [p.id for p in chain] == ["1", "2", "3", "4"]


def test_mid_thread_link_includes_earlier_posts():
    page = _page([_single(t, c) for t, c in zip("123", THREAD[:3], strict=True)] + [_module([_item("4", THREAD[3])])])
    assert [p.id for p in xt.assemble("3", xt.page_posts(page)[0])] == ["1", "2", "3", "4"]


def test_status_id_from_url():
    assert xt.status_id_from_url("https://x.com/a/status/2119090498459861193?s=20") == "2119090498459861193"
    assert xt.status_id_from_url("https://twitter.com/a/statuses/123456") == "123456"
    assert xt.status_id_from_url("https://x.com/a") is None


async def test_unroll_single_post_is_none(monkeypatch):
    async def fake_fetch(status_id, *, timeout_s):
        return _page([_single("9", OTHER_REPLY)])

    monkeypatch.setattr(xt, "fetch_status_page", fake_fetch)
    assert await xt.unroll_thread("https://x.com/someone/status/9") is None


async def test_unroll_follows_show_more_cursor(monkeypatch):
    cursor = {"entry_id": "cursor-showmore", "item": {"content": {"__typename": "TimelineTimelineCursor", "cursor_type": "ShowMore"}}}
    pages = {
        "1": _page([_single("1", THREAD[0]), _module([_item("2", THREAD[1]), cursor])]),
        "2": _page([_single("2", THREAD[1]), _module([_item("3", THREAD[2]), _item("4", THREAD[3])])]),
    }
    fetched = []

    async def fake_fetch(status_id, *, timeout_s):
        fetched.append(status_id)
        return pages[status_id]

    monkeypatch.setattr(xt, "fetch_status_page", fake_fetch)
    thread = await xt.unroll_thread("https://x.com/anagenxyz/status/1")
    assert fetched == ["1", "2"]
    assert [p.id for p in thread.posts] == ["1", "2", "3", "4"] and not thread.truncated
    assert thread.url == "https://x.com/anagenxyz/status/1"


async def test_unroll_page_without_data_raises(monkeypatch):
    async def fake_fetch(status_id, *, timeout_s):
        return "<html>login</html>"

    monkeypatch.setattr(xt, "fetch_status_page", fake_fetch)
    with pytest.raises(xt.ThreadFetchError):
        await xt.unroll_thread("https://x.com/a/status/1")


def test_render_numbers_posts_and_respects_char_cap():
    posts = xt.assemble("1", xt.page_posts(_page_for_first())[0])
    thread = xt.Thread(posts=posts, url="https://x.com/anagenxyz/status/1")
    text = xt.render_thread(thread)
    assert text.startswith("X thread by @anagenxyz (Anagenxyz), 4 posts")
    assert "[1/4] Story of RU58841 🧵\n(1 attached: photo)" in text and "[4/4] Full long note text" in text
    cut = xt.render_thread(thread, max_chars=120)
    assert "more posts not shown" in cut and "[4/4]" not in cut


async def test_router_hook(monkeypatch):
    from bot.router import Router

    posts = xt.assemble("1", xt.page_posts(_page_for_first())[0])

    async def fake_unroll(url, *, timeout_s, max_posts):
        return xt.Thread(posts=posts, url="https://x.com/anagenxyz/status/1")

    router = SimpleNamespace(config={"TWITTER_UNROLL_ENABLED": True}, logger=logging.getLogger("t"), _is_twitter_status_url=lambda u: "/status/" in u)
    monkeypatch.setattr(xt, "unroll_thread", fake_unroll)
    out = await Router._maybe_unroll_x_thread(router, "https://x.com/anagenxyz/status/1")
    assert "Cover the whole thread" in out and "[4/4]" in out and "UNVERIFIED EXTERNAL CONTENT" in out

    async def failing(url, *, timeout_s, max_posts):
        raise xt.ThreadFetchError("HTTP 429")

    monkeypatch.setattr(xt, "unroll_thread", failing)
    assert await Router._maybe_unroll_x_thread(router, "https://x.com/anagenxyz/status/1") is None
    router.config = {}  # bare test configs never touch the network
    assert await Router._maybe_unroll_x_thread(router, "https://x.com/anagenxyz/status/1") is None
