import httpx
import pytest
from unittest.mock import AsyncMock

# Skip — requires Playwright browser installation
pytestmark = pytest.mark.skip(reason="Requires Playwright browser installation")

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


@pytest.mark.asyncio
async def test_extract_does_not_disable_tier_b_on_transient_page_closed_error() -> None:
    svc = WebExtractionService()
    svc._tier_b_available = True

    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(success=False, tier_used="A", error="no text")
    )

    svc._tier_b_playwright = AsyncMock(
        side_effect=Exception("Target page, context or browser has been closed")
    )

    res = await svc.extract("https://example.com")

    assert res.success is False
    assert res.tier_used == "B"
    assert svc._tier_b_available is True


@pytest.mark.asyncio
async def test_extract_does_not_disable_tier_b_on_version_mismatch_428() -> None:
    """A 428 / version-mismatch error from the remote Playwright server must
    NOT disable Tier B globally -- it is a configuration issue, not fatal."""
    svc = WebExtractionService()
    svc._tier_b_available = True

    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(success=False, tier_used="A", error="http_status:403")
    )

    msg = (
        "BrowserType.connect: WebSocket error: ws://localhost:3006/ "
        "428 Precondition Required Playwright server version 1.59.1 "
        "does not match client version 1.55.0"
    )
    svc._tier_b_playwright = AsyncMock(side_effect=Exception(msg))

    res = await svc.extract("https://example.com")

    assert res.success is False
    assert res.tier_used == "B"
    # Tier B must remain available for the next request
    assert svc._tier_b_available is True


def test_is_playwright_fatal_error_version_mismatch() -> None:
    """Verify that connection/websocket/version-mismatch errors are NOT fatal."""
    non_fatal = [
        "428 Precondition Required Playwright server version 1.59.1 "
        "does not match client version 1.55.0",
        "WebSocket error: ws://localhost:3006/ connection refused",
        "BrowserType.connect: WebSocket connection failed",
        "connect ECONNREFUSED 127.0.0.1:3006",
        "ws:// handshake failed",
    ]
    for msg in non_fatal:
        assert WebExtractionService._is_playwright_fatal_error(msg) is False, f"{msg!r}"

    fatal = [
        "error while loading shared libraries: libatk-1.0.so.0",
        "Executable doesn't exist at /home/user/.cache/ms-playwright/...",
        "Failed to launch browser",
        "browserType.launch: Chromium doesn't exist",
    ]
    for msg in fatal:
        assert WebExtractionService._is_playwright_fatal_error(msg) is True, f"{msg!r}"
