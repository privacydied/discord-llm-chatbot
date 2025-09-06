import asyncio
import types
import builtins
import pytest

from bot.router import Router
from bot.search.types import SearchResult, SafeSearch, SearchCategory


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

class FakeMetrics:
    def __init__(self):
        self.calls = []
    def increment(self, name, labels=None, value=1):
        self.calls.append(("increment", name, labels or {}, value))
    def inc(self, name, value=1, labels=None):
        self.calls.append(("inc", name, labels or {}, value))

class CapturingProvider:
    def __init__(self, results=None):
        self.calls = []
        self._results = results or [
            SearchResult(title="Result One", url="https://example.com/1", snippet="First snippet"),
            SearchResult(title="Result Two", url="https://example.com/2", snippet="Second snippet"),
        ]
    async def search(self, params):
        self.calls.append(params)
        return list(self._results)


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
async def test_extract_inline_search_queries_with_category_and_commas():
    bot = FakeBot()
    r = Router(bot)
    text = (
        "A [search(what, is, love, images)] "
        "B [search(latest ai, NEWS)] "
        "C [search(simple, video)] "
        "D [search(a, b, unknowncat)]"
    )
    matches = r._extract_inline_search_queries(text)
    assert len(matches) == 4

    # 1) images
    (_, _), q1, c1 = matches[0]
    assert q1 == "what, is, love"
    assert c1 == SearchCategory.IMAGES

    # 2) NEWS (case-insensitive)
    (_, _), q2, c2 = matches[1]
    assert q2 == "latest ai"
    assert c2 == SearchCategory.NEWS

    # 3) video (singular synonym)
    (_, _), q3, c3 = matches[2]
    assert q3 == "simple"
    assert c3 == SearchCategory.VIDEOS

    # 4) unknown category -> treated as part of query, category None (back-compat)
    (_, _), q4, c4 = matches[3]
    assert q4 == "a, b, unknowncat"
    assert c4 is None


@pytest.mark.asyncio
async def test_resolve_inline_searches_passes_category_to_provider(monkeypatch):
    import bot.router as router_mod

    provider = CapturingProvider()
    monkeypatch.setattr(router_mod, "get_search_provider", lambda: provider)

    bot = FakeBot()
    r = Router(bot)

    # With explicit category
    msg1 = FakeMessage("[search(test query, images)]")
    await r._resolve_inline_searches(msg1.content, msg1)
    assert len(provider.calls) >= 1
    p1 = provider.calls[-1]
    assert getattr(p1, "category") == SearchCategory.IMAGES

    # Without category -> defaults to TEXT in SearchQueryParams
    msg2 = FakeMessage("[search(another query)]")
    await r._resolve_inline_searches(msg2.content, msg2)
    p2 = provider.calls[-1]
    assert getattr(p2, "category") == SearchCategory.TEXT


@pytest.mark.asyncio
async def test_inline_search_metrics_labels_include_category_success(monkeypatch):
    import bot.router as router_mod

    metrics = FakeMetrics()
    provider = CapturingProvider()
    monkeypatch.setattr(router_mod, "get_search_provider", lambda: provider)

    bot = FakeBot()
    bot.metrics = metrics
    r = Router(bot)

    msg = FakeMessage("Run: [search(kung fu, videos)]")
    await r._resolve_inline_searches(msg.content, msg)

    names = [n for (_, n, _, _) in metrics.calls]
    assert "inline_search.start" in names
    assert "inline_search.success" in names

    # Find labels for start and success
    start_labels = next(lbl for (kind, n, lbl, _v) in metrics.calls if n == "inline_search.start")
    success_labels = next(lbl for (kind, n, lbl, _v) in metrics.calls if n == "inline_search.success")

    assert start_labels.get("category") == "videos"
    assert success_labels.get("category") == "videos"
    assert start_labels.get("provider") == bot.config["SEARCH_PROVIDER"]
    assert success_labels.get("provider") == bot.config["SEARCH_PROVIDER"]


@pytest.mark.asyncio
async def test_inline_search_metrics_labels_include_category_error(monkeypatch):
    import bot.router as router_mod

    class ErroringProvider:
        async def search(self, params):
            raise RuntimeError("boom")

    metrics = FakeMetrics()
    monkeypatch.setattr(router_mod, "get_search_provider", lambda: ErroringProvider())

    bot = FakeBot()
    bot.metrics = metrics
    r = Router(bot)

    msg = FakeMessage("[search(thing, images)]")
    await r._resolve_inline_searches(msg.content, msg)

    names = [n for (_, n, _, _) in metrics.calls]
    assert "inline_search.start" in names
    assert "inline_search.error" in names

    error_labels = next(lbl for (kind, n, lbl, _v) in metrics.calls if n == "inline_search.error")
    assert error_labels.get("category") == "images"
    assert error_labels.get("provider") == bot.config["SEARCH_PROVIDER"]


@pytest.mark.asyncio
async def test_resolve_inline_searches_no_directives_returns_same_text():
    bot = FakeBot()
    r = Router(bot)
    msg = FakeMessage("No directives here.")
    new_text = await r._resolve_inline_searches(msg.content, msg)
    assert new_text == msg.content
