import pytest
from unittest.mock import AsyncMock

from bot.router import Router, XApiClient
from bot.modality import InputItem


class DummyBot:
    def __init__(self):
        # Minimal config enabling X API path
        self.config = {
            "X_API_ENABLED": True,
            "X_API_BEARER_TOKEN": "test",
            "X_API_REQUIRE_API_FOR_TWITTER": False,
            "X_API_ALLOW_FALLBACK_ON_5XX": True,
        }
        self.tts_manager = None
        self.loop = None


@pytest.mark.asyncio
async def test_x_api_routes_video_to_stt(monkeypatch):
    bot = DummyBot()
    router = Router(bot)

    # Force tweet id extraction success regardless of URL format
    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    # Dummy X API client returning video media
    class _DummyX:
        async def get_tweet_by_id(self, _id):
            return {
                "data": {"text": "video post", "author_id": "u1"},
                "includes": {
                    "users": [{"id": "u1", "username": "user"}],
                    "media": [
                        {"type": "video", "media_key": "m1"},
                    ],
                },
            }

    async def _get_client(_self):
        return _DummyX()

    monkeypatch.setattr(Router, "_get_x_api_client", _get_client)

    # Patch STT ingest
    import bot.router as router_mod
    stt_mock = AsyncMock(return_value={"transcription": "hello world"})
    monkeypatch.setattr(router_mod, "hear_infer_from_url", stt_mock)

    item = InputItem(source_type="url", payload="https://twitter.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert "Video/audio content" in res
    assert "hello world" in res
    stt_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_x_api_photo_only_formats_text(monkeypatch):
    bot = DummyBot()
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    class _DummyX:
        async def get_tweet_by_id(self, _id):
            return {
                "data": {"text": "photo post", "author_id": "u1"},
                "includes": {
                    "users": [{"id": "u1", "username": "user"}],
                    "media": [
                        {"type": "photo", "media_key": "m1"},
                        {"type": "photo", "media_key": "m2"},
                    ],
                },
            }

    async def _get_client(_self):
        return _DummyX()

    monkeypatch.setattr(Router, "_get_x_api_client", _get_client)

    item = InputItem(source_type="url", payload="https://x.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert "Photos: 2" in res
    assert "x.com" in res or "twitter.com" in res


@pytest.mark.asyncio
async def test_x_api_text_only_formats_default(monkeypatch):
    bot = DummyBot()
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    class _DummyX:
        async def get_tweet_by_id(self, _id):
            return {
                "data": {"text": "plain post", "author_id": "u1"},
                "includes": {
                    "users": [{"id": "u1", "username": "user"}],
                    "media": [],
                },
            }

    async def _get_client(_self):
        return _DummyX()

    monkeypatch.setattr(Router, "_get_x_api_client", _get_client)

    item = InputItem(source_type="url", payload="https://twitter.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    # Should contain the URL and text body formatted
    assert "twitter.com" in res or "x.com" in res
    assert "plain post" in res
