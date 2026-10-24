"""check_ai_generated: tweet/IG/attachment media -> Sightengine genai scores. [CA][REH]"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiohttp import web

from bot import ai_detect
from bot.tools import ToolContext, get_registry
from bot.tools.builtins.ai_check import check_ai_generated

CREDS = {"SIGHTENGINE_API_USER": "u", "SIGHTENGINE_API_SECRET": "s"}


def test_registered():
    assert "check_ai_generated" in {s["function"]["name"] for s in get_registry().schemas()}


@pytest.mark.parametrize(
    "url",
    [
        "https://x.com/someone/status/1585341984679469056",
        "https://vxtwitter.com/someone/status/1585341984679469056",
        "https://fixupx.com/someone/status/1585341984679469056?s=20",
        "https://www.instagram.com/p/CzW4fBzvTMO/",
        "https://www.oginstagram.com/wafflesweekly/reel/CzW4fBzvTMO/",
    ],
)
def test_finds_post_links(url):
    assert ai_detect.find_post_urls(f"is this ai {url}") == [url]


def test_ignores_other_links():
    assert ai_detect.find_post_urls("https://example.com/a.jpg https://youtube.com/watch?v=x") == []


def test_verdict_bands():
    assert ai_detect.verdict(0.97) == "very likely AI-generated"
    assert ai_detect.verdict(0.6) == "possibly AI-generated"
    assert ai_detect.verdict(0.3) == "probably not AI-generated"
    assert ai_detect.verdict(0.02) == "very likely real (not AI)"


@pytest.fixture
async def fake_apis(unused_tcp_port, monkeypatch):
    seen: list[dict] = []

    async def status(req):
        return web.json_response(
            {
                "code": 200,
                "tweet": {
                    "media": {
                        "all": [
                            {"type": "photo", "url": "https://pbs.twimg.com/media/a.jpg"},
                            {"type": "video", "url": "https://video.twimg.com/v.mp4", "thumbnail_url": "https://pbs.twimg.com/thumb.jpg"},
                        ]
                    }
                },
            }
        )

    async def check(req):
        params = dict(req.query)
        seen.append(params)
        if params.get("api_user") != "u":
            return web.json_response({"status": "failure", "error": {"type": "credentials_error", "message": "Incorrect API user or secret"}})
        score = 0.97 if params.get("url", "").endswith("a.jpg") else 0.04
        return web.json_response({"status": "success", "request": {"operations": 5}, "type": {"ai_generated": score}})

    app = web.Application()
    app.router.add_get("/status/{tid}", status)
    app.router.add_get("/check.json", check)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    base = f"http://127.0.0.1:{unused_tcp_port}"
    monkeypatch.setattr(ai_detect, "FXTWITTER_STATUS_URL", base + "/status/{tweet_id}")
    monkeypatch.setattr(ai_detect, "SIGHTENGINE_URL", base + "/check.json")
    yield seen
    await runner.cleanup()


def _msg(content="", reference=None, attachments=()):
    return SimpleNamespace(id=1, content=content, reference=reference, attachments=list(attachments), embeds=[], channel=None)


async def test_tweet_in_replied_message_is_scored(fake_apis):
    replied = _msg("look https://x.com/a/status/1585341984679469056")
    ask = _msg("<@9> is this AI?", reference=SimpleNamespace(resolved=None, message_id=2))
    ask.channel = SimpleNamespace(fetch_message=lambda mid: _coro(replied))
    result = await check_ai_generated(ToolContext(message=ask, config=CREDS), {})
    assert result.ok
    assert "photo 1: 97% AI probability -> very likely AI-generated" in result.content
    assert "video 1 (thumbnail frame): 4%" in result.content
    assert "not proof" in result.content
    assert {p["models"] for p in fake_apis} == {"genai"}


async def test_attachment_scored_when_no_link(fake_apis):
    att = SimpleNamespace(content_type="image/png", filename="a.jpg", url="https://cdn.discordapp.com/attachments/1/2/a.jpg")
    result = await check_ai_generated(ToolContext(message=_msg("is this ai", attachments=[att]), config=CREDS), {})
    assert result.ok and "97%" in result.content


async def test_not_configured_and_nothing_to_check(fake_apis, monkeypatch):
    for key in ("SIGHTENGINE_API_USER", "SIGHTENGINE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)  # independent of the live .env
    result = await check_ai_generated(ToolContext(message=_msg("is this ai?"), config={}), {})
    assert not result.ok and "isn't set up" in result.error
    result = await check_ai_generated(ToolContext(message=_msg("is this ai?"), config=CREDS), {})
    assert not result.ok and "nothing to check" in result.error


async def test_detector_errors_are_reported(fake_apis):
    bad = {"SIGHTENGINE_API_USER": "wrong", "SIGHTENGINE_API_SECRET": "s"}
    result = await check_ai_generated(ToolContext(message=_msg("https://x.com/a/status/1585341984679469056"), config=bad), {})
    assert not result.ok and "Incorrect API user or secret" in result.error


async def _coro(value):
    return value
