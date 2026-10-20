"""Keyless search fallbacks: DDG -> camofox Google -> Google News -> Wikipedia [REH][IV]."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bot.search.types import SearchResult
from bot.tools import ToolContext
from bot.tools.builtins import web_sources as src
from bot.tools.registry import execute_tool

RSS = """<?xml version="1.0"?><rss><channel><title>"q" - Google News</title>
<item><title>New offer brings together Nintendo Switch 2 and Super Mario Galaxy 2 - Nintendo</title>
<link>https://news.google.com/rss/articles/abc?oc=5</link>
<pubDate>Mon, 06 Apr 2026 07:00:00 GMT</pubDate><source url="https://www.nintendo.com">Nintendo</source></item>
<item><title><![CDATA[Galaxy 2 review &amp; more]]></title><link>https://news.google.com/rss/articles/def</link>
<pubDate>garbage</pubDate><source url="https://x.example">IGN</source></item>
</channel></rss>"""

SERP = """- heading "q - Google Search"
- searchbox "Search" [e1]: q
- navigation:
  - link "Images" [e2]
- link "Super Mario Galaxy 2 \\"Switch 2 Edition\\"" [e3]:
  - /url: https://www.nintendo.com/us/store/products/smg2/
  - cite: nintendo.com
  - text: Play Super Mario Galaxy 2 on Nintendo Switch 2 with 4K.
- link "Maps result" [e4]:
  - /url: https://maps.google.com/x
- group:
  - text: People also ask
- link "Wiki" [e5]:
  - /url: https://en.wikipedia.org/wiki/Super_Mario_Galaxy_2
"""


def test_parse_news_rss_dates_sources_and_cdata():
    items = src.parse_news_rss(RSS, 5)
    assert items[0].title == "New offer brings together Nintendo Switch 2 and Super Mario Galaxy 2"
    assert items[0].source == "Nintendo" and items[0].published == datetime(2026, 4, 6, 7, tzinfo=UTC)
    assert items[1].title == "Galaxy 2 review & more" and items[1].published is None
    assert "Nintendo, published 06 Apr 2026" in src.render_news(items)


def test_parse_google_snapshot_keeps_organic_results_only():
    results = src.parse_google_snapshot(SERP, 5)
    assert [r.url for r in results] == [
        "https://www.nintendo.com/us/store/products/smg2/",
        "https://en.wikipedia.org/wiki/Super_Mario_Galaxy_2",
    ]
    assert results[0].title == 'Super Mario Galaxy 2 "Switch 2 Edition"'
    assert results[0].snippet.startswith("Play Super Mario Galaxy 2")


async def test_camofox_skipped_without_key(monkeypatch):
    http = AsyncMock()
    monkeypatch.setattr(src, "_http", http)
    assert await src.camofox_google("q", 5, {}) == []
    http.assert_not_awaited()


async def test_camofox_searches_and_always_closes_tab(monkeypatch):
    calls = []

    async def fake_http(method, url, total_s=8.0, **kw):
        calls.append((method, url, kw.get("headers")))
        if method == "POST":
            return SimpleNamespace(json=lambda: {"tabId": "t1"})
        if method == "GET":
            return SimpleNamespace(json=lambda: {"url": "https://www.google.com/search?q=q", "snapshot": SERP})
        return SimpleNamespace(json=lambda: {"ok": True})

    monkeypatch.setattr(src, "_http", fake_http)
    results = await src.camofox_google("q", 5, {"CAMOFOX_ACCESS_KEY": "k", "CAMOFOX_URL": "http://cf:9377/"})
    assert len(results) == 2
    assert [c[0] for c in calls] == ["POST", "GET", "DELETE"]
    assert all(c[2] == {"Authorization": "Bearer k"} for c in calls)
    assert calls[0][1] == "http://cf:9377/tabs"


async def test_camofox_captcha_page_gives_nothing(monkeypatch):
    async def fake_http(method, url, total_s=8.0, **kw):
        if method == "POST":
            return SimpleNamespace(json=lambda: {"tabId": "t1"})
        return SimpleNamespace(json=lambda: {"url": "https://www.google.com/sorry/index", "snapshot": SERP})

    monkeypatch.setattr(src, "_http", fake_http)
    assert await src.camofox_google("q", 5, {"CAMOFOX_ACCESS_KEY": "k"}) == []


# -- web_search chain ---------------------------------------------------------


def _patch_sources(monkeypatch, ddg=(), google=(), news=(), wiki=()):
    from bot.tools.builtins import web

    order = []

    def track(name, value):
        async def run(*a, **k):
            order.append(name)
            return list(value)

        return run

    monkeypatch.setattr(web, "_ddg", track("ddg", ddg))
    monkeypatch.setattr(src, "camofox_google", track("google", google))
    monkeypatch.setattr(web, "_news_results", track("news", news))
    monkeypatch.setattr(src, "wikipedia_search", track("wiki", wiki))
    return order


R = SearchResult(title="Hit", url="https://example.com", snippet="snip")


async def test_ddg_results_used_first(monkeypatch):
    order = _patch_sources(monkeypatch, ddg=[R], wiki=[R])
    result = await execute_tool("web_search", {"query": "q"}, ToolContext(config={}))
    assert order == ["ddg"] and "web search (duckduckgo)" in result.content


async def test_empty_ddg_falls_through_to_news_then_wikipedia(monkeypatch):
    order = _patch_sources(monkeypatch, wiki=[R])
    result = await execute_tool("web_search", {"query": "q"}, ToolContext(config={}))
    assert order == ["ddg", "google", "news", "wiki"]
    assert result.ok and "web search (wikipedia)" in result.content and "Hit -- https://example.com" in result.content


async def test_all_sources_empty_says_so(monkeypatch):
    _patch_sources(monkeypatch)
    result = await execute_tool("web_search", {"query": "q"}, ToolContext(config={}))
    assert result.ok and "from any search source" in result.content


# -- search_news / lookup_wikipedia --------------------------------------------


async def test_search_news_uses_google_news_first(monkeypatch):
    items = src.parse_news_rss(RSS, 5)
    gnews = AsyncMock(return_value=items)
    guardian = AsyncMock()
    monkeypatch.setattr(src, "google_news", gnews)
    monkeypatch.setattr("bot.news.headlines.fetch_headlines", guardian)
    result = await execute_tool("search_news", {"topic": "galaxy 2 switch 2", "days": 3}, ToolContext(config={}))
    assert "published 06 Apr 2026" in result.content
    assert gnews.await_args.kwargs["days"] == 3
    guardian.assert_not_awaited()


async def test_search_news_widens_window_then_falls_back_to_guardian(monkeypatch):
    gnews = AsyncMock(return_value=[])
    monkeypatch.setattr(src, "google_news", gnews)
    monkeypatch.setattr("bot.news.headlines.fetch_headlines", AsyncMock(return_value=[]))
    result = await execute_tool("search_news", {"topic": "obscure"}, ToolContext(config={}))
    assert gnews.await_count == 2 and gnews.await_args_list[1].kwargs.get("days") is None
    assert "No news articles found" in result.content


async def test_lookup_wikipedia(monkeypatch):
    article = src.WikiArticle(title="Super Mario Galaxy 2", url="https://en.wikipedia.org/wiki/SMG2", text="A 2010 game.")
    fetch = AsyncMock(return_value=article)
    monkeypatch.setattr(src, "wikipedia_article", fetch)
    result = await execute_tool("lookup_wikipedia", {"topic": "smg2", "full": True}, ToolContext(config={}))
    assert result.ok and "Super Mario Galaxy 2\nA 2010 game." in result.content
    assert fetch.await_args.kwargs["full"] is True
    monkeypatch.setattr(src, "wikipedia_article", AsyncMock(return_value=None))
    assert "No Wikipedia article" in (await execute_tool("lookup_wikipedia", {"topic": "zzz"}, ToolContext(config={}))).content
