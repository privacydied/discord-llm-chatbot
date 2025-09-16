import json
import pytest
from unittest.mock import AsyncMock
from urllib.parse import quote

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

    item = InputItem(
        source_type="url", payload="https://twitter.com/user/status/1", order_index=0
    )
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

    item = InputItem(
        source_type="url", payload="https://x.com/user/status/1", order_index=0
    )
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

    item = InputItem(
        source_type="url", payload="https://twitter.com/user/status/1", order_index=0
    )
    res = await router._handle_general_url(item)

    # Should contain the URL and text body formatted
    assert "twitter.com" in res or "x.com" in res
    assert "plain post" in res


@pytest.mark.asyncio
async def test_x_api_photo_only_routes_to_vl_when_enabled(monkeypatch):
    bot = DummyBot()
    # Enable photo->VL routing
    bot.config["X_API_ROUTE_PHOTOS_TO_VL"] = True
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    class _DummyX:
        async def get_tweet_by_id(self, _id):
            return {
                "data": {"text": "photo post", "author_id": "u1"},
                "includes": {
                    "users": [{"id": "u1", "username": "user"}],
                    "media": [
                        {
                            "type": "photo",
                            "media_key": "m1",
                            "url": "https://example.com/p1.jpg",
                        },
                        {
                            "type": "photo",
                            "media_key": "m2",
                            "url": "https://example.com/p2.jpg",
                        },
                    ],
                },
            }

    async def _get_client(_self):
        return _DummyX()

    monkeypatch.setattr(Router, "_get_x_api_client", _get_client)

    # Avoid real network/vision by mocking the helper
    async def _fake_vl(self, image_url: str, *, prompt=None, model_override=None):
        return f"desc for {image_url.split('/')[-1]}"

    monkeypatch.setattr(Router, "_vl_describe_image_from_url", _fake_vl, raising=True)

    item = InputItem(
        source_type="url", payload="https://twitter.com/user/status/1", order_index=0
    )
    res = await router._handle_general_url(item)

    assert "Photos analyzed: 2/2" in res
    assert "📷 Photo 1/2" in res and "📷 Photo 2/2" in res
    assert "desc for p1.jpg" in res and "desc for p2.jpg" in res


@pytest.mark.asyncio
async def test_resolve_x_media_unwraps_fx_proxy(monkeypatch):
    bot = DummyBot()
    router = Router(bot)

    target_url = "https://video.twimg.com/amplify_video/123/vid/720x1280/sample.mp4"
    wrapped_url = f"https://api.fxtwitter.com/2/go?url={quote(target_url, safe='')}"

    class DummyResp:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data
            self.text = json.dumps(data)

        def json(self):
            return self._data

    class DummyHttp:
        async def get(self, url, config=None, headers=None):
            if "api.vxtwitter.com" in url:
                return DummyResp(404, {})
            if "api.fxtwitter.com" in url:
                data = {
                    "tweet": {
                        "media": {
                            "videos": [
                                {
                                    "variants": [
                                        {
                                            "url": wrapped_url,
                                            "content_type": "video/mp4",
                                            "bitrate": 832000,
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
                return DummyResp(200, data)
            return DummyResp(404, {})

    async def fake_http_client():
        return DummyHttp()

    monkeypatch.setattr("bot.router.get_http_client", fake_http_client)

    resolved = await router._resolve_x_media(["https://x.com/user/status/1234567890"])

    assert resolved["kind"] == "video"
    assert resolved["url"] == target_url
