"""fxtwitter media hydration for sparse syndication payloads. [REH]

Regression for 2026-07-22: syndication CDN returned {text, user} only for a
tweet with 2 photos -> photos=0 -> text flow -> bot claimed it "can't open
x/twitter links". fxtwitter still exposes the media; hydration merges it in.
"""

from __future__ import annotations

import logging


from bot.router import Router
from bot.router_components.x_routing import syndication_media_hint_keys


class _StubRouter:
    """Minimal surface for Router._hydrate_media_from_fxtwitter."""

    logger = logging.getLogger("test.fx_hydration")

    def _build_x_syn_quick_request_config(self):
        return None

    def _extract_fxtwitter_tweet_node(self, payload):
        return Router._extract_fxtwitter_tweet_node(self, payload)


class _FakeResp:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, resp: _FakeResp) -> None:
        self._resp = resp
        self.calls: list[str] = []

    async def get(self, url, **kwargs):
        self.calls.append(url)
        return self._resp


def _patch_client(monkeypatch, resp: _FakeResp) -> _FakeClient:
    client = _FakeClient(resp)

    async def fake_get_http_client():
        return client

    monkeypatch.setattr("bot.router.get_http_client", fake_get_http_client)
    return client


FX_PAYLOAD = {
    "code": 200,
    "tweet": {
        "text": "The TOP 10 MOST EXPENSIVE transfers",
        "media": {
            "photos": [
                {"url": "https://pbs.twimg.com/media/AAA.jpg?name=orig"},
                {"url": "https://pbs.twimg.com/media/BBB.jpg?name=orig"},
            ],
            "videos": [],
        },
    },
}


class TestFxMediaHydration:
    async def test_sparse_payload_gains_photos(self, monkeypatch) -> None:
        client = _patch_client(monkeypatch, _FakeResp(200, FX_PAYLOAD))
        sparse = {"text": "some caption", "user": {"screen_name": "x"}}

        out = await Router._hydrate_media_from_fxtwitter(_StubRouter(), "123", sparse)

        assert len(out["photos"]) == 2
        assert out["photos"][0]["url"].startswith("https://pbs.twimg.com/")
        assert client.calls == ["https://api.fxtwitter.com/status/123"]

    async def test_payload_with_media_hints_untouched(self, monkeypatch) -> None:
        client = _patch_client(monkeypatch, _FakeResp(200, FX_PAYLOAD))
        assert any(k in ("photos",) for k in syndication_media_hint_keys())
        existing = {"text": "t", "photos": [{"url": "https://pbs.twimg.com/orig.jpg"}]}

        out = await Router._hydrate_media_from_fxtwitter(_StubRouter(), "123", existing)

        assert out is existing  # no fetch, no mutation
        assert client.calls == []

    async def test_video_mapped_to_syndication_shape(self, monkeypatch) -> None:
        from bot.syndication.extract import syndication_has_video

        payload = {"code": 200, "tweet": {"text": "clip", "media": {"photos": [], "videos": [{"url": "https://video.twimg.com/v.mp4", "format": "video/mp4"}]}}}
        _patch_client(monkeypatch, _FakeResp(200, payload))

        out = await Router._hydrate_media_from_fxtwitter(_StubRouter(), "9", {"text": "clip", "user": {}})

        assert syndication_has_video(out)

    async def test_fx_error_returns_original(self, monkeypatch) -> None:
        _patch_client(monkeypatch, _FakeResp(404, {}))
        sparse = {"text": "t", "user": {}}
        out = await Router._hydrate_media_from_fxtwitter(_StubRouter(), "123", sparse)
        assert out is sparse

    async def test_never_raises(self, monkeypatch) -> None:
        async def broken_client():
            raise RuntimeError("network down")

        monkeypatch.setattr("bot.router.get_http_client", broken_client)
        sparse = {"text": "t", "user": {}}
        out = await Router._hydrate_media_from_fxtwitter(_StubRouter(), "123", sparse)
        assert out is sparse
