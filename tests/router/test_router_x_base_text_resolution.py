import pytest

from bot.router import Router, XApiClient


class DummyBot:
    def __init__(self):
        self.config = {
            "X_API_ENABLED": True,
            "X_API_BEARER_TOKEN": "test",
            "X_SYNDICATION_ENABLED": True,
        }
        self.tts_manager = None
        self.loop = None


@pytest.mark.asyncio
async def test_resolve_x_base_text_prefers_api(monkeypatch):
    router = Router(DummyBot())
    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda _u: "1"))

    class _DummyX:
        async def get_tweet_by_id(self, _tweet_id):
            return {"data": {"text": "api text"}}

    async def _get_client(_self):
        return _DummyX()

    async def _get_syn(_self, _tweet_id):
        raise AssertionError("syndication should not be used when API succeeds")

    monkeypatch.setattr(Router, "_get_x_api_client", _get_client)
    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _get_syn)
    monkeypatch.setattr(Router, "_format_x_tweet_result", lambda _s, _a, _u: "api-ok")

    out = await router._resolve_x_base_text_for_url("https://x.com/user/status/1")
    assert out == "api-ok"


@pytest.mark.asyncio
async def test_resolve_x_base_text_falls_back_to_syndication(monkeypatch):
    router = Router(DummyBot())
    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda _u: "1"))

    class _DummyX:
        async def get_tweet_by_id(self, _tweet_id):
            raise RuntimeError("api down")

    async def _get_client(_self):
        return _DummyX()

    async def _get_syn(_self, _tweet_id):
        return {"text": "syn text"}

    monkeypatch.setattr(Router, "_get_x_api_client", _get_client)
    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _get_syn)
    monkeypatch.setattr(Router, "_format_syndication_result", lambda _s, _a, _u: "syn-ok")

    out = await router._resolve_x_base_text_for_url("https://x.com/user/status/1")
    assert out == "syn-ok"


@pytest.mark.asyncio
async def test_resolve_x_base_text_returns_none_without_tweet_id(monkeypatch):
    router = Router(DummyBot())
    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda _u: None))

    out = await router._resolve_x_base_text_for_url("https://example.com/no-status")
    assert out is None


@pytest.mark.asyncio
async def test_resolve_x_base_text_respects_syndication_disabled(monkeypatch):
    bot = DummyBot()
    bot.config["X_SYNDICATION_ENABLED"] = False
    router = Router(bot)
    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda _u: "1"))

    class _DummyX:
        async def get_tweet_by_id(self, _tweet_id):
            raise RuntimeError("api down")

    async def _get_client(_self):
        return _DummyX()

    monkeypatch.setattr(Router, "_get_x_api_client", _get_client)

    out = await router._resolve_x_base_text_for_url("https://x.com/user/status/1")
    assert out is None
