"""Tier D: camofox + Turnstile solver wiring for web extraction."""

from __future__ import annotations

import os

import pytest

from bot import camofox_solver as cs
from bot.web_extraction_service import ExtractionResult, WebExtractionService


def test_disabled_without_key(monkeypatch):
    monkeypatch.setenv("WEBEX_ENABLE_TIER_D", "1")
    monkeypatch.delenv("CAMOFOX_ACCESS_KEY", raising=False)
    assert cs.enabled() is False


def test_disabled_by_env_flag(monkeypatch):
    monkeypatch.setenv("CAMOFOX_ACCESS_KEY", "k")
    monkeypatch.setenv("WEBEX_ENABLE_TIER_D", "0")
    assert cs.enabled() is False
    monkeypatch.setenv("WEBEX_ENABLE_TIER_D", "1")
    assert cs.enabled() is True


def test_cfx_env_is_set_then_restored(monkeypatch):
    monkeypatch.setenv("CFX_KEY", "old")
    monkeypatch.delenv("CFX_TAB", raising=False)
    with cs._cfx_env("newkey", "tab1", "http://x:1"):
        assert (os.environ["CFX_KEY"], os.environ["CFX_TAB"], os.environ["CFX_URL"]) == ("newkey", "tab1", "http://x:1")
        assert os.environ["CFX_USER"] == cs.CAMOFOX_USER_ID
    assert os.environ["CFX_KEY"] == "old"
    assert "CFX_TAB" not in os.environ


async def test_fetch_text_never_raises(monkeypatch):
    monkeypatch.setenv("CAMOFOX_ACCESS_KEY", "k")

    async def boom(*_a, **_k):
        raise RuntimeError("camofox down")

    monkeypatch.setattr(cs, "_fetch", boom)
    assert await cs.fetch_text("https://example.com/") is None


async def test_fetch_text_none_when_disabled(monkeypatch):
    monkeypatch.delenv("CAMOFOX_ACCESS_KEY", raising=False)
    assert await cs.fetch_text("https://example.com/") is None


@pytest.mark.parametrize("recovered", [True, False])
async def test_service_tier_d_recovers_failed_extraction(monkeypatch, recovered):
    monkeypatch.setattr(cs, "enabled", lambda: True)

    async def fake_fetch(_url):
        return "article body " * 30 if recovered else None

    monkeypatch.setattr(cs, "fetch_text", fake_fetch)
    failed = ExtractionResult(success=False, tier_used="C", error="bot_wall:just a moment", bot_wall_marker="just a moment")
    res = await WebExtractionService()._tier_d_camofox("https://walled.example/", failed)
    if recovered:
        assert res.success and res.tier_used == "D" and "article body" in (res.text or "")
    else:
        assert res is failed


async def test_extract_only_tries_tier_d_on_bot_wall(monkeypatch):
    calls = []

    async def fake_tier_d(self, url, failed):
        calls.append(url)
        return failed

    async def no_safety(_url):
        return None

    monkeypatch.setattr(WebExtractionService, "_tier_d_camofox", fake_tier_d)
    monkeypatch.setattr("bot.url_safety.validate_url_with_dns", no_safety)
    monkeypatch.setattr("bot.web_extraction_service.ENABLE_WAYBACK_FALLBACK", False)
    svc = WebExtractionService()
    for failed in (
        ExtractionResult(success=False, tier_used="A", error="http_status:404"),
        ExtractionResult(success=False, tier_used="C", error="bot_wall:x", bot_wall_marker="just a moment"),
    ):

        async def fake_via(_url, original_url=None, _failed=failed):
            return _failed

        monkeypatch.setattr(svc, "_extract_via", fake_via)
        await svc.extract("https://example.com/p")
    assert calls == ["https://example.com/p"]  # only the bot-wall failure
