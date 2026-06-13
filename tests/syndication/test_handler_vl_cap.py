from types import SimpleNamespace

import pytest

from bot.syndication.handler import handle_twitter_syndication_to_vl


class _FakeResp:
    def __init__(self, status=200, payload=b"img") -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def read(self):
        return self._payload


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, _url):
        return _FakeResp(status=200)


@pytest.mark.asyncio
async def test_syndication_vl_caps_to_one_image_by_default(monkeypatch) -> None:
    # Keep global cap high to prove X-specific cap defaults to 1.
    monkeypatch.setenv("VL_MAX_IMAGES", "4")
    monkeypatch.delenv("X_SYNDICATION_VL_MAX_IMAGES", raising=False)
    monkeypatch.setattr("aiohttp.ClientSession", _FakeSession)

    captured = {}

    async def _fake_unified(temp_paths, user_caption, intent):
        captured["count"] = len(temp_paths)
        captured["caption"] = user_caption
        captured["intent"] = intent
        return SimpleNamespace(content="OK")

    tweet_json = {
        "text": "tweet caption",
        "photos": [
            {"url": "https://pbs.twimg.com/media/a.jpg?name=orig"},
            {"url": "https://pbs.twimg.com/media/b.jpg?name=orig"},
            {"url": "https://pbs.twimg.com/media/c.jpg?name=orig"},
        ],
    }

    result = await handle_twitter_syndication_to_vl(
        tweet_json=tweet_json,
        url="https://x.com/user/status/1",
        unified_vl_pipeline_func=_fake_unified,
    )

    assert result == "OK"
    assert captured["count"] == 1
    assert captured["caption"] == "tweet caption"
    assert captured["intent"] == "Tweet analysis"


@pytest.mark.asyncio
async def test_syndication_vl_cap_respects_x_env_override(monkeypatch) -> None:
    monkeypatch.setenv("VL_MAX_IMAGES", "4")
    monkeypatch.setenv("X_SYNDICATION_VL_MAX_IMAGES", "2")
    monkeypatch.setattr("aiohttp.ClientSession", _FakeSession)

    captured = {}

    async def _fake_unified(temp_paths, user_caption, intent):
        captured["count"] = len(temp_paths)
        return SimpleNamespace(content="OK")

    tweet_json = {
        "text": "tweet caption",
        "photos": [
            {"url": "https://pbs.twimg.com/media/a.jpg?name=orig"},
            {"url": "https://pbs.twimg.com/media/b.jpg?name=orig"},
            {"url": "https://pbs.twimg.com/media/c.jpg?name=orig"},
        ],
    }

    result = await handle_twitter_syndication_to_vl(
        tweet_json=tweet_json,
        url="https://x.com/user/status/1",
        unified_vl_pipeline_func=_fake_unified,
    )

    assert result == "OK"
    assert captured["count"] == 2
