import asyncio
from types import SimpleNamespace

import logging
import pytest

from bot.router import Router
from bot.modality import InputItem


class DummyBot:
    def __init__(self, config: dict):
        self.config = config
        self.tts_manager = None
        # Provide a minimal loop attribute if accessed
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = None
        # Optional attributes used elsewhere but not needed here
        self.vision_orchestrator = None


@pytest.mark.asyncio
async def test_unroll_logging_ok(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    # Force status URL checks to pass and avoid X API path
    monkeypatch.setattr(Router, "_is_twitter_status_url", staticmethod(lambda u: True))
    monkeypatch.setattr(Router, "_is_twitter_url", staticmethod(lambda u: False))

    # Stub out network-heavy generic URL processing
    async def fake_process_url(url: str):
        return {"text": "Generic URL processed"}

    import bot.router as router_mod

    monkeypatch.setattr(router_mod, "process_url", fake_process_url)

    # Return a successful unroll context
    async def fake_unroll(url: str, *, timeout_s: float, max_tweets: int, max_chars: int):
        ctx = SimpleNamespace(
            joined_text="Thread joined text", tweet_count=3, canonical_url=url
        )
        return ctx, None

    monkeypatch.setattr(router_mod, "unroll_author_thread", fake_unroll)

    bot = DummyBot({"TWITTER_UNROLL_ENABLED": True})
    r = Router(bot)

    item = InputItem(source_type="url", payload="https://x.com/a/status/1", order_index=0)
    result = await r._handle_general_url(item)

    # Ensure we returned the joined thread text
    assert result == "Thread joined text"

    # Verify log breadcrumbs: start and ok
    events = [getattr(rec, "event", None) for rec in caplog.records]
    assert "unroll_start" in events
    assert "unroll_ok" in events


@pytest.mark.asyncio
async def test_unroll_logging_fallback(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    monkeypatch.setattr(Router, "_is_twitter_status_url", staticmethod(lambda u: True))
    monkeypatch.setattr(Router, "_is_twitter_url", staticmethod(lambda u: False))

    async def fake_process_url(url: str):
        return {"text": "Generic URL processed"}

    import bot.router as router_mod

    monkeypatch.setattr(router_mod, "process_url", fake_process_url)

    # Return a fallback with reason
    async def fake_unroll(url: str, *, timeout_s: float, max_tweets: int, max_chars: int):
        return None, "dom_mismatch"

    monkeypatch.setattr(router_mod, "unroll_author_thread", fake_unroll)

    bot = DummyBot({"TWITTER_UNROLL_ENABLED": True})
    r = Router(bot)

    item = InputItem(source_type="url", payload="https://x.com/a/status/1", order_index=0)
    _ = await r._handle_general_url(item)

    # Verify log breadcrumbs: start and fallback with reason present
    fallback_recs = [rec for rec in caplog.records if getattr(rec, "event", None) == "unroll_fallback"]
    assert any(getattr(rec, "event", None) == "unroll_start" for rec in caplog.records)
    assert fallback_recs, "Expected an unroll_fallback log record"
    assert any(
        isinstance(getattr(rec, "detail", None), dict)
        and getattr(rec, "detail", {}).get("reason") == "dom_mismatch"
        for rec in fallback_recs
    )

