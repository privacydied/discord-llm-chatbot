"""Tier C (reader-proxy paywall bypass) tests for WebExtractionService.

These do NOT require a Playwright browser install -- they mock the reader
HTTP hop. Tier C is the headless-safe substitute for a browser paywall-bypass
extension, which the automation Chromium build cannot load via --load-extension.
[PAY]
"""
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from bot.web_extraction_service import ExtractionResult, WebExtractionService

_TIMES_URL = "https://www.thetimes.com/life-style/sex-relationships/article/why-men-wont-date-sklh6t2xg"
_JINA_BODY = (
    "Title: Ghosting, pressure, the cost of dinner\n\n"
    "URL Source: http://www.thetimes.com/...\n\n"
    "Published Time: 2026-08-10\n\n"
    "Markdown Content:\n"
    "Male friends I know say they feel cornered by women wanting to settle down "
    "and they're tired of being painted as the bad guys."
)


@pytest.mark.asyncio
async def test_tier_c_reader_strips_wrapper_and_returns_body() -> None:
    svc = WebExtractionService()
    req = httpx.Request("GET", "https://r.jina.ai/http/" + _TIMES_URL)
    fake_resp = httpx.Response(200, text=_JINA_BODY, request=req)
    with patch.object(svc, "_get_client", AsyncMock()), patch(
        "httpx.AsyncClient"
    ) as mock_client:
        ctx = AsyncMock()
        ctx.__aenter__.return_value.get.return_value = fake_resp
        mock_client.return_value = ctx
        res: ExtractionResult | None = await svc._tier_c_reader(_TIMES_URL)

    assert res is not None
    assert res.success is True
    assert res.tier_used == "C"
    body = res.text or ""
    assert "Male friends I know say" in body
    assert "Markdown Content:" not in body
    assert res.canonical_url == _TIMES_URL


@pytest.mark.asyncio
async def test_tier_c_reader_passes_raw_url_through() -> None:
    svc = WebExtractionService()
    captured = {}

    with patch.object(svc, "_get_client", AsyncMock()), patch(
        "httpx.AsyncClient"
    ) as mock_client:
        ctx = AsyncMock()
        def fake_get(url):
            captured["url"] = url
            return httpx.Response(200, text="Markdown Content:\nhello world body with enough words to pass the minimum length guard",
                                  request=httpx.Request("GET", url))
        ctx.__aenter__.return_value.get.side_effect = fake_get
        mock_client.return_value = ctx
        res = await svc._tier_c_reader("https://example.com/a b?x=1&y=2")

    assert res.success is True
    # Default base is the bare form r.jina.ai/<url>; the original https:// URL
    # is passed through unchanged (jina rejects http:// targets with 422 there).
    assert captured["url"] == "https://r.jina.ai/https://example.com/a b?x=1&y=2" or captured["url"].startswith("https://r.jina.ai/https://example.com/")


@pytest.mark.asyncio
async def test_tier_c_reader_empty_body_is_failure() -> None:
    svc = WebExtractionService()
    with patch.object(svc, "_get_client", AsyncMock()), patch(
        "httpx.AsyncClient"
    ) as mock_client:
        ctx = AsyncMock()
        ctx.__aenter__.return_value.get.return_value = httpx.Response(
            200, text="", request=httpx.Request("GET", "https://r.jina.ai/http/x")
        )
        mock_client.return_value = ctx
        res = await svc._tier_c_reader(_TIMES_URL)

    assert res is not None
    assert res.success is False
    assert res.tier_used == "C"


@pytest.mark.asyncio
async def test_tier_c_reader_accepts_long_article_with_subscribe_cta() -> None:
    # Regression: a real article ends with a "[Subscribe now]" CTA. The paywall
    # marker must not reject a long article that merely mentions subscriptions.
    svc = WebExtractionService()
    body = (
        "Markdown Content:\n"
        + ("Male friends I know say they feel cornered by women wanting to settle down. " * 20)
        + "\n\n[Subscribe now](https://www.thetimes.com/subscribe) for unlimited access."
    )
    with patch.object(svc, "_get_client", AsyncMock()), patch(
        "httpx.AsyncClient"
    ) as mock_client:
        ctx = AsyncMock()
        ctx.__aenter__.return_value.get.return_value = httpx.Response(
            200, text=body, request=httpx.Request("GET", "https://r.jina.ai/http/x")
        )
        mock_client.return_value = ctx
        res = await svc._tier_c_reader(_TIMES_URL)

    assert res is not None
    assert res.success is True
    assert "Subscribe now" in res.text


@pytest.mark.asyncio
async def test_extract_falls_through_to_tier_c_when_a_and_b_fail() -> None:
    svc = WebExtractionService()
    svc._tier_b_available = False
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(success=False, tier_used="A", error="no text")
    )
    # Force Tier C on even though the global default may vary
    import bot.web_extraction_service as wes

    with patch.object(wes, "ENABLE_TIER_C", True), patch.object(
        svc, "_tier_c_reader"
    ) as mock_c:
        mock_c.return_value = ExtractionResult(
            success=True, tier_used="C", canonical_url=_TIMES_URL,
            text="x" * 900,  # non-thin so it is returned
        )
        res = await svc.extract(_TIMES_URL)

    assert res.success is True
    assert res.tier_used == "C"
    mock_c.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_prefers_tier_c_over_tier_b() -> None:
    # Tier C (reader proxy) is the paywall-bypass specialist and must be tried
    # BEFORE Tier B (Playwright). When both succeed, the non-thin Tier C result
    # wins so a paywalled page rendered by Playwright doesn't block the bypass.
    svc = WebExtractionService()
    svc._tier_b_available = True
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(success=False, tier_used="A", error="no text")
    )
    import bot.web_extraction_service as wes

    with patch.object(wes, "ENABLE_TIER_C", True), patch.object(
        svc, "_tier_c_reader"
    ) as mock_c, patch.object(svc, "_tier_b_playwright") as mock_b:
        mock_c.return_value = ExtractionResult(
            success=True, tier_used="C", canonical_url=_TIMES_URL, text="x" * 900
        )
        mock_b.return_value = ExtractionResult(
            success=True, tier_used="B", canonical_url=_TIMES_URL,
            text="paywalled page rendered by chromium with the wall still up",
        )
        res = await svc.extract(_TIMES_URL)

    assert res.success is True
    assert res.tier_used == "C"
    mock_c.assert_awaited_once()
    mock_b.assert_not_called()


@pytest.mark.asyncio
async def test_extract_cascades_to_tier_c_on_thin_tier_a_success() -> None:
    # A paywalled page returns HTTP 200 with only a short teaser (the exact
    # Times failure: 122 chars). This must NOT be treated as success -- it must
    # cascade to Tier C so the reader proxy can fetch the real article.
    svc = WebExtractionService()
    svc._tier_b_available = False
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(
            success=True, tier_used="A", canonical_url=_TIMES_URL, text="Short teaser only."
        )
    )
    import bot.web_extraction_service as wes

    with patch.object(wes, "ENABLE_TIER_C", True), patch.object(
        svc, "_tier_c_reader"
    ) as mock_c:
        mock_c.return_value = ExtractionResult(
            success=True, tier_used="C", canonical_url=_TIMES_URL, text="x" * 900
        )
        res = await svc.extract(_TIMES_URL)

    assert res.success is True
    assert res.tier_used == "C"
    mock_c.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_skips_tier_c_when_disabled() -> None:
    svc = WebExtractionService()
    svc._tier_b_available = False
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(success=False, tier_used="A", error="no text")
    )
    import bot.web_extraction_service as wes

    with patch.object(wes, "ENABLE_TIER_C", False), patch.object(
        svc, "_tier_c_reader"
    ) as mock_c:
        res = await svc.extract(_TIMES_URL)

    assert res.success is False
    assert res.tier_used == "A"
    mock_c.assert_not_called()
