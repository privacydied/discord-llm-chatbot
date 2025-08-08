import asyncio
import types
import builtins
import pytest

from bot.router import Router
from bot.search.types import SearchResult, SafeSearch


class FakeProvider:
    async def search(self, params):
        # Return deterministic results for testing
        return [
            SearchResult(title="Result One", url="https://example.com/1", snippet="First snippet"),
            SearchResult(title="Result Two", url="https://example.com/2", snippet="Second snippet"),
        ]


class FakeContextManager:
    async def get_context_string(self, message):
        return ""


class FakeBot:
    def __init__(self):
        self.config = {
            "SEARCH_PROVIDER": "ddg",
            "SEARCH_MAX_RESULTS": 3,
            "SEARCH_SAFE": "moderate",
            "SEARCH_LOCALE": None,
            "DDG_TIMEOUT_MS": 2000,
        }
        self.tts_manager = None
        self.loop = asyncio.get_event_loop()
        self.user = types.SimpleNamespace(id=1234567890)
        self.context_manager = FakeContextManager()


class FakeMessage:
    def __init__(self, content: str):
        self.id = 42
        self.content = content
        self.mentions = []


@pytest.mark.asyncio
async def test_extract_inline_search_queries():
    bot = FakeBot()
    r = Router(bot)
    text = "Please [search(hello world)] and also [search(another query)] thanks."
    matches = r._extract_inline_search_queries(text)
    assert len(matches) == 2
    assert matches[0][1] == "hello world"
    assert matches[1][1] == "another query"


@pytest.mark.asyncio
async def test_resolve_inline_searches_replaces_with_results(monkeypatch):
    # Monkeypatch the provider factory to return our fake provider
    import bot.router as router_mod
    monkeypatch.setattr(router_mod, "get_search_provider", lambda: FakeProvider())

    bot = FakeBot()
    r = Router(bot)

    msg = FakeMessage("Summary: [search(test query)] End.")
    new_text = await r._resolve_inline_searches(msg.content, msg)

    assert "Search: `test query`" in new_text
    assert "https://example.com/1" in new_text
    assert "https://example.com/2" in new_text


@pytest.mark.asyncio
async def test_resolve_inline_searches_no_directives_returns_same_text():
    bot = FakeBot()
    r = Router(bot)
    msg = FakeMessage("No directives here.")
    new_text = await r._resolve_inline_searches(msg.content, msg)
    assert new_text == msg.content
