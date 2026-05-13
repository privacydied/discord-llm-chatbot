import json
import pytest
from unittest.mock import AsyncMock
from urllib.parse import quote

from bot.router import Router, XApiClient
from bot.modality import InputItem
from bot.exceptions import InferenceError


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

    item = InputItem(source_type="url", payload="https://twitter.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert "Photos analyzed: 2/2" in res
    assert "📷 Photo 1/2" in res and "📷 Photo 2/2" in res
    assert "desc for p1.jpg" in res and "desc for p2.jpg" in res


@pytest.mark.asyncio
async def test_sparse_syndication_defers_to_api_video_stt(monkeypatch):
    bot = DummyBot()
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    async def _fake_syn(_self, _tweet_id):
        # Sparse syndication payload: no media metadata, only text/user.
        return {"text": "caption only", "user": {"screen_name": "u"}}

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _fake_syn)

    class _DummyX:
        async def get_tweet_by_id(self, _id):
            return {
                "data": {"text": "video post", "author_id": "u1"},
                "includes": {
                    "users": [{"id": "u1", "username": "user"}],
                    "media": [{"type": "video", "media_key": "m1"}],
                },
            }

    async def _get_client(_self):
        return _DummyX()

    monkeypatch.setattr(Router, "_get_x_api_client", _get_client)

    import bot.router as router_mod

    stt_mock = AsyncMock(return_value={"transcription": "hello world"})
    monkeypatch.setattr(router_mod, "hear_infer_from_url", stt_mock)

    def _fmt(_self, base_text, url, stt_res):
        return f"STT:{(stt_res or {}).get('transcription', '')}"

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    item = InputItem(source_type="url", payload="https://x.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert res == "STT:hello world"
    stt_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_video_url_inference_error_degrades_to_caption_only_with_resolved_base_text(
    monkeypatch,
):
    bot = DummyBot()
    router = Router(bot)

    import bot.router as router_mod

    stt_mock = AsyncMock(side_effect=InferenceError("boom"))
    monkeypatch.setattr(router_mod, "hear_infer_from_url", stt_mock)

    async def _fake_resolve_base(_self, _url):
        return "base text"

    def _fmt(_self, base_text, url, stt_res):
        return f"FMT:{base_text}|{(stt_res or {}).get('transcription', '')}"

    monkeypatch.setattr(Router, "_resolve_x_base_text_for_url", _fake_resolve_base)
    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    item = InputItem(source_type="url", payload="https://x.com/user/status/1", order_index=0)
    res = await router._handle_video_url(item)

    assert res == "FMT:base text|"
    stt_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_sparse_syndication_without_api_uses_direct_media_probe_for_stt(
    monkeypatch,
):
    bot = DummyBot()
    bot.config["X_API_ENABLED"] = False
    bot.system_prompts = {"vl_prompt": None}
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    async def _fake_syn(_self, _tweet_id):
        return {"text": "caption only", "user": {"screen_name": "u"}}

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _fake_syn)

    async def _fake_get_client(_self):
        return None

    monkeypatch.setattr(Router, "_get_x_api_client", _fake_get_client)

    async def _fake_resolve(self, urls, frontend_hints=None, primary_hints=None):
        return {"kind": "video", "url": urls[0]}

    monkeypatch.setattr(Router, "_resolve_x_media", _fake_resolve)

    import bot.router as router_mod

    stt_mock = AsyncMock(return_value={"transcription": "hello world"})
    monkeypatch.setattr(router_mod, "hear_infer_from_url", stt_mock)

    def _fmt(_self, base_text, url, stt_res):
        return f"STT:{(stt_res or {}).get('transcription', '')}"

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    item = InputItem(source_type="url", payload="https://x.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert res == "STT:hello world"
    stt_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_sparse_syndication_without_api_uses_direct_media_probe_for_images(
    monkeypatch,
):
    bot = DummyBot()
    bot.config["X_API_ENABLED"] = False
    bot.system_prompts = {"vl_prompt": None}
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    async def _fake_syn(_self, _tweet_id):
        return {"text": "caption only", "user": {"screen_name": "u"}}

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _fake_syn)

    async def _fake_get_client(_self):
        return None

    monkeypatch.setattr(Router, "_get_x_api_client", _fake_get_client)

    async def _fake_resolve(self, urls, frontend_hints=None, primary_hints=None):
        return {
            "kind": "image",
            "images": ["https://pbs.twimg.com/media/test123.jpg?name=orig"],
            "url": "https://pbs.twimg.com/media/test123.jpg?name=orig",
        }

    monkeypatch.setattr(Router, "_resolve_x_media", _fake_resolve)
    import bot.syndication.handler as syn_handler_mod

    async def _fake_syn_handler(syn_data, url, vl_handler, vl_prompt=None, reply_style="ack+thoughts"):
        assert len((syn_data or {}).get("photos") or []) == 1
        return "VL_OK_SPARSE_IMAGE"

    monkeypatch.setattr(
        syn_handler_mod,
        "handle_twitter_syndication_to_vl",
        _fake_syn_handler,
        raising=True,
    )

    item = InputItem(source_type="url", payload="https://x.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert res == "VL_OK_SPARSE_IMAGE"


@pytest.mark.asyncio
async def test_sparse_syndication_unknown_forces_stt_before_text_fallback(monkeypatch):
    bot = DummyBot()
    bot.config["X_API_ENABLED"] = False
    bot.system_prompts = {"vl_prompt": None}
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    async def _fake_syn(_self, _tweet_id):
        return {"text": "caption only", "user": {"screen_name": "u"}}

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _fake_syn)

    async def _fake_get_client(_self):
        return None

    monkeypatch.setattr(Router, "_get_x_api_client", _fake_get_client)

    async def _fake_resolve(self, urls, frontend_hints=None, primary_hints=None):
        return {"kind": "unknown", "images": [], "url": None}

    monkeypatch.setattr(Router, "_resolve_x_media", _fake_resolve)

    import bot.router as router_mod

    stt_mock = AsyncMock(return_value={"transcription": "forced hello world"})
    monkeypatch.setattr(router_mod, "hear_infer_from_url", stt_mock)

    def _fmt(_self, base_text, url, stt_res):
        return f"STT:{(stt_res or {}).get('transcription', '')}"

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    item = InputItem(source_type="url", payload="https://x.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert res == "STT:forced hello world"
    stt_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_sparse_syndication_tco_article_resolves_to_text(monkeypatch):
    bot = DummyBot()
    bot.config["X_API_ENABLED"] = False
    bot.system_prompts = {"vl_prompt": None}
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    async def _fake_syn(_self, _tweet_id):
        return {"text": "https://t.co/Zq03pbrEgu", "user": {"screen_name": "u"}}

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _fake_syn)

    async def _fake_get_client(_self):
        return None

    monkeypatch.setattr(Router, "_get_x_api_client", _fake_get_client)

    article_mock = AsyncMock(
        return_value={
            "id": "2016825738041630720",
            "title": "The TESTOSTERONE Kabbalah",
            "preview_text": "They control everything.",
            "content": {"blocks": [{"text": "Cellular energy production and metabolism."}]},
        }
    )
    monkeypatch.setattr(Router, "_fetch_x_article_from_fxtwitter", article_mock, raising=True)

    resolve_mock = AsyncMock(return_value={"kind": "unknown", "images": [], "url": None})
    monkeypatch.setattr(Router, "_resolve_x_media", resolve_mock, raising=True)

    import bot.router as router_mod

    stt_mock = AsyncMock(return_value={"transcription": "should not run"})
    monkeypatch.setattr(router_mod, "hear_infer_from_url", stt_mock)

    def _fmt(_self, base_text, url, stt_res):
        return base_text

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    item = InputItem(source_type="url", payload="https://x.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert "The TESTOSTERONE Kabbalah" in res
    assert "They control everything." in res
    assert "Cellular energy production and metabolism." in res
    assert "https://t.co/" not in res
    article_mock.assert_awaited_once()
    resolve_mock.assert_not_awaited()
    stt_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_x_article_from_fxtwitter_parses_article_payload(monkeypatch):
    bot = DummyBot()
    router = Router(bot)

    class DummyResp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "tweet": {
                    "article": {
                        "id": "2016825738041630720",
                        "title": "The TESTOSTERONE Kabbalah",
                        "preview_text": "They control everything.",
                        "content": {
                            "blocks": [
                                {"type": "header-two", "text": "Section A"},
                                {"type": "unstyled", "text": "Section B"},
                            ]
                        },
                    }
                }
            }

    class DummyHttp:
        @staticmethod
        async def get(url, config=None, headers=None):
            return DummyResp()

    async def fake_http_client():
        return DummyHttp()

    monkeypatch.setattr("bot.router.get_http_client", fake_http_client)

    article = await router._fetch_x_article_from_fxtwitter("1")

    assert article is not None
    assert article["id"] == "2016825738041630720"
    assert article["url"] == "https://x.com/i/article/2016825738041630720"
    assert article["title"] == "The TESTOSTERONE Kabbalah"
    assert article["preview_text"] == "They control everything."
    assert article["content"]["blocks"][0]["text"] == "Section A"
    assert article["content"]["blocks"][1]["text"] == "Section B"


@pytest.mark.asyncio
async def test_hydrate_syndication_article_merges_full_article_blocks(monkeypatch):
    bot = DummyBot()
    router = Router(bot)

    article_mock = AsyncMock(
        return_value={
            "id": "2016825738041630720",
            "title": "The TESTOSTERONE Kabbalah",
            "preview_text": "They control everything.",
            "content": {
                "blocks": [
                    {
                        "type": "unstyled",
                        "text": "Cellular energy production and metabolism.",
                    },
                    {
                        "type": "unstyled",
                        "text": "Hormonal signaling under chronic stress.",
                    },
                ]
            },
        }
    )
    monkeypatch.setattr(Router, "_fetch_x_article_from_fxtwitter", article_mock, raising=True)

    syn = {
        "text": "https://t.co/Zq03pbrEgu",
        "article": {
            "id": "2016825738041630720",
            "title": "The TESTOSTERONE Kabbalah",
            "preview_text": "They control everything.",
        },
    }
    hydrated = await router._hydrate_syndication_article_if_needed("1", syn)
    text = router._extract_syndication_text(hydrated)

    assert "Cellular energy production and metabolism." in text
    assert "Hormonal signaling under chronic stress." in text
    assert "https://t.co/" not in text
    article_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_sparse_image_probe_passes_hydrated_article_text_to_vl(monkeypatch):
    bot = DummyBot()
    bot.config["X_API_ENABLED"] = False
    bot.system_prompts = {"vl_prompt": None}
    router = Router(bot)

    monkeypatch.setattr(XApiClient, "extract_tweet_id", staticmethod(lambda u: "1"))

    async def _fake_syn(_self, _tweet_id):
        return {
            "text": "https://t.co/Zq03pbrEgu",
            "article": {
                "id": "2016825738041630720",
                "title": "The TESTOSTERONE Kabbalah",
                "preview_text": "They control everything.",
            },
            "user": {"screen_name": "u"},
            "news_action_type": "article",
        }

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _fake_syn)

    async def _fake_get_client(_self):
        return None

    monkeypatch.setattr(Router, "_get_x_api_client", _fake_get_client)

    article_mock = AsyncMock(
        return_value={
            "id": "2016825738041630720",
            "title": "The TESTOSTERONE Kabbalah",
            "preview_text": "They control everything.",
            "content": {"blocks": [{"text": "Cellular energy production and metabolism."}]},
        }
    )
    monkeypatch.setattr(Router, "_fetch_x_article_from_fxtwitter", article_mock, raising=True)

    async def _fake_resolve(self, urls, frontend_hints=None, primary_hints=None):
        return {
            "kind": "image",
            "images": ["https://pbs.twimg.com/media/test123.jpg?name=orig"],
            "url": "https://pbs.twimg.com/media/test123.jpg?name=orig",
        }

    monkeypatch.setattr(Router, "_resolve_x_media", _fake_resolve)

    captured = {}
    import bot.syndication.handler as syn_handler_mod

    async def _fake_syn_handler(syn_data, url, vl_handler, vl_prompt=None, reply_style="ack+thoughts"):
        captured["text"] = (syn_data or {}).get("text", "")
        return "VL_OK_SPARSE_IMAGE_ARTICLE"

    monkeypatch.setattr(
        syn_handler_mod,
        "handle_twitter_syndication_to_vl",
        _fake_syn_handler,
        raising=True,
    )

    item = InputItem(source_type="url", payload="https://x.com/user/status/1", order_index=0)
    res = await router._handle_general_url(item)

    assert res == "VL_OK_SPARSE_IMAGE_ARTICLE"
    assert "Cellular energy production and metabolism." in captured.get("text", "")
    assert "https://t.co/" not in captured.get("text", "")
    article_mock.assert_awaited_once()


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
