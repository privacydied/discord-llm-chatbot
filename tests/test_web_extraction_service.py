import httpx
import pytest
from unittest.mock import AsyncMock

from bot.web_extraction_service import ExtractionResult, WebExtractionService


@pytest.mark.asyncio
async def test_extract_tier_a_success_short_circuits() -> None:
    svc = WebExtractionService()
    svc._tier_b_available = True

    tier_a = ExtractionResult(
        success=True,
        tier_used="A",
        canonical_url="https://example.com",
        text="hello",
    )

    svc._tier_a_httpx = AsyncMock(return_value=tier_a)
    svc._tier_b_playwright = AsyncMock(
        return_value=ExtractionResult(success=True, tier_used="B", text="nope")
    )

    res = await svc.extract("https://example.com")

    assert res.success is True
    assert res.tier_used == "A"
    assert res.text == "hello"
    svc._tier_b_playwright.assert_not_called()


@pytest.mark.asyncio
async def test_extract_falls_back_to_tier_b_on_tier_a_failure() -> None:
    svc = WebExtractionService()
    svc._tier_b_available = True

    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(success=False, tier_used="A", error="no text")
    )
    svc._tier_b_playwright = AsyncMock(
        return_value=ExtractionResult(
            success=True,
            tier_used="B",
            canonical_url="https://example.com",
            text="from browser",
        )
    )

    res = await svc.extract("https://example.com")

    assert res.success is True
    assert res.tier_used == "B"
    assert res.text == "from browser"


@pytest.mark.asyncio
async def test_extract_tier_a_http_status_sets_deterministic_error() -> None:
    svc = WebExtractionService()
    svc._tier_b_available = False

    req = httpx.Request("GET", "https://example.com")
    resp = httpx.Response(404, request=req)

    svc._tier_a_httpx = AsyncMock(
        side_effect=httpx.HTTPStatusError("boom", request=req, response=resp)
    )

    res = await svc.extract("https://example.com")

    assert res.success is False
    assert res.tier_used == "A"
    assert res.error == "http_status:404"


@pytest.mark.asyncio
async def test_extract_disables_tier_b_on_fatal_launch_failure() -> None:
    svc = WebExtractionService()
    svc._tier_b_available = True

    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(success=False, tier_used="A", error="no text")
    )

    svc._tier_b_playwright = AsyncMock(side_effect=Exception("Failed to launch browser"))

    res = await svc.extract("https://example.com")

    assert res.success is False
    assert res.tier_used == "B"
    assert svc._tier_b_available is False
