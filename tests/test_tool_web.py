"""Tests for the web_search, search_news and read_webpage tools [IV][REH][SFT]."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from bot.search.types import SearchResult
from bot.tools import ToolContext
from bot.tools.builtins import web
from bot.tools.registry import execute_tool


def _ctx() -> ToolContext:
    return ToolContext(config={"SEARCH_SAFE": "off", "GUARDIAN_API_KEY": "k"})


def _provider(results):
    provider = MagicMock()
    provider.search = AsyncMock(return_value=results)
    return provider


async def test_web_search_formats_results_as_untrusted(monkeypatch):
    results = [SearchResult(title="Super Mario Galaxy 2 on Switch", url="https://nintendo.com/x", snippet="out now")]
    provider = _provider(results)
    monkeypatch.setattr("bot.search.factory.get_search_provider", lambda: provider)
    result = await execute_tool("web_search", {"query": "  mario   galaxy 2 switch ", "max_results": 99}, _ctx())
    assert result.ok
    assert "1. Super Mario Galaxy 2 on Switch -- https://nintendo.com/x" in result.content
    assert "out now" in result.content
    assert "web search: mario galaxy 2 switch" in result.content
    params = provider.search.await_args.args[0]
    assert params.query == "mario galaxy 2 switch"
    assert params.max_results == web.MAX_RESULTS


async def test_web_search_requires_query():
    result = await execute_tool("web_search", {"query": "   "}, _ctx())
    assert not result.ok


async def test_web_search_no_results_is_not_an_error(monkeypatch):
    monkeypatch.setattr("bot.search.factory.get_search_provider", lambda: _provider([]))
    result = await execute_tool("web_search", {"query": "nothing"}, _ctx())
    assert result.ok and "No results" in result.content


async def test_search_news_uses_guardian(monkeypatch):
    from bot.news.headlines import Headline

    fetch = AsyncMock(return_value=[Headline(title="Big story", url="https://g.com/a", published="2026-10-18T10:00Z")])
    monkeypatch.setattr("bot.news.headlines.fetch_headlines", fetch)
    result = await execute_tool("search_news", {"topic": "election", "days": 400}, _ctx())
    assert result.ok and "Big story" in result.content and "(2026-10-18)" in result.content
    assert fetch.await_args.kwargs["days"] == web.MAX_NEWS_DAYS


async def test_search_news_empty(monkeypatch):
    monkeypatch.setattr("bot.news.headlines.fetch_headlines", AsyncMock(return_value=[]))
    result = await execute_tool("search_news", {}, _ctx())
    assert result.ok and "web_search" in result.content


async def test_read_webpage_rejects_non_http():
    for url in ("file:///etc/passwd", "ftp://x", "", "javascript:alert(1)"):
        result = await execute_tool("read_webpage", {"url": url}, _ctx())
        assert not result.ok, url


async def test_read_webpage_truncates_and_wraps(monkeypatch):
    monkeypatch.setattr("bot.web.process_url", AsyncMock(return_value={"text": "x" * 10_000, "error": None}))
    result = await execute_tool("read_webpage", {"url": "https://example.com/a"}, _ctx())
    assert result.ok
    assert "[truncated]" in result.content and "https://example.com/a" in result.content


async def test_read_webpage_reports_fetch_failure(monkeypatch):
    monkeypatch.setattr("bot.web.process_url", AsyncMock(return_value={"text": None, "error": "blocked"}))
    result = await execute_tool("read_webpage", {"url": "https://10.0.0.1/"}, _ctx())
    assert not result.ok and "blocked" in (result.error or "")


def test_tool_directive_tells_model_to_search_before_disagreeing():
    from bot.tools.inference import TOOL_OUTPUT_DIRECTIVE

    assert "out of date" in TOOL_OUTPUT_DIRECTIVE
    assert "never argue against a user from memory" in TOOL_OUTPUT_DIRECTIVE
