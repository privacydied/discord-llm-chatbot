"""Bot-wall / challenge-page fast-fail and blocklist-message tests for
WebExtractionService. These mock the fetcher tiers so they run without a
Playwright browser or network. [PAY]
"""

from unittest.mock import AsyncMock, patch

import pytest

from bot.web_extraction_service import (
    BOT_WALL_BLOCKED_HOST_MSG,
    BOT_WALL_GENERIC_MSG,
    ExtractionResult,
    WebExtractionService,
    is_blocked_host,
    is_bot_wall,
)


def _bot_wall_html(marker: str = "One more step") -> str:
    return f"<html><body><h1>{marker}</h1><p>Please complete the security check to access example.com</p></body></html>"


# --- pure helpers -----------------------------------------------------------


def test_is_blocked_host_known_mirrors() -> None:
    assert is_blocked_host("https://archive.is/ppfcf")
    assert is_blocked_host("https://archive.ph/abcd")
    assert is_blocked_host("https://sub.archive.today/x")
    assert not is_blocked_host("https://example.com")


def test_is_bot_wall_markers() -> None:
    assert is_bot_wall(_bot_wall_html()) == "one more step"
    assert is_bot_wall("Please verify you are a human") == "verify you are a human"
    assert is_bot_wall("Checking your browser before access") == "checking your browser"
    assert is_bot_wall("real article text with no challenge") is None
    assert is_bot_wall(None) is None
    assert is_bot_wall("") is None


# --- extractor behavior -----------------------------------------------------


@pytest.mark.asyncio
async def test_bot_wall_tier_a_fast_fails_skips_b() -> None:
    """A bot-wall body from Tier A must short-circuit the cascade: no Tier B
    Playwright launch (the ~26s waste), and a bot_wall_marker is set."""
    svc = WebExtractionService()
    svc._tier_b_available = True
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(
            success=True, tier_used="A", canonical_url="https://archive.is/ppfcf", text=_bot_wall_html()
        )
    )
    svc._tier_c_reader = AsyncMock(return_value=ExtractionResult(success=True, tier_used="C", text="x"))
    svc._tier_b_playwright = AsyncMock(return_value=ExtractionResult(success=True, tier_used="B", text="x"))

    res = await svc.extract("https://archive.is/ppfcf")

    assert res.success is False
    assert res.bot_wall_marker is not None
    svc._tier_c_reader.assert_not_called()
    svc._tier_b_playwright.assert_not_called()


@pytest.mark.asyncio
async def test_blocked_host_message_via_to_message() -> None:
    svc = WebExtractionService()
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(
            success=True, tier_used="A", canonical_url="https://archive.is/ppfcf", text=_bot_wall_html()
        )
    )
    res = await svc.extract("https://archive.is/ppfcf")
    msg = res.to_message()
    assert BOT_WALL_BLOCKED_HOST_MSG in msg
    assert "archive.is" not in msg or "this capture host" in msg


@pytest.mark.asyncio
async def test_generic_bot_wall_message_for_unknown_host() -> None:
    svc = WebExtractionService()
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(
            success=True, tier_used="A", canonical_url="https://example.com/x", text=_bot_wall_html()
        )
    )
    res = await svc.extract("https://example.com/x")
    msg = res.to_message()
    assert BOT_WALL_GENERIC_MSG in msg
    assert BOT_WALL_BLOCKED_HOST_MSG not in msg


@pytest.mark.asyncio
async def test_non_bot_wall_still_cascades() -> None:
    """A non-bot-wall Tier A failure must still try the other tiers (no false
    positive)."""
    svc = WebExtractionService()
    svc._tier_b_available = True
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(success=False, tier_used="A", error="no text", text="")
    )
    svc._tier_c_reader = AsyncMock(
        return_value=ExtractionResult(
            success=True, tier_used="C", canonical_url="https://example.com", text="real article body " * 100
        )
    )
    res = await svc.extract("https://example.com")
    assert res.success is True
    assert res.tier_used == "C"
    assert res.bot_wall_marker is None


@pytest.mark.asyncio
async def test_wayback_fallback_on_failure() -> None:
    """When all tiers fail and a Wayback snapshot exists, fetch the snapshot
    instead of giving up."""
    svc = WebExtractionService()
    svc._tier_b_available = True
    svc._tier_a_httpx = AsyncMock(return_value=ExtractionResult(success=False, tier_used="A", error="no text", text=""))
    svc._tier_c_reader = AsyncMock(return_value=ExtractionResult(success=False, tier_used="C", error="no text", text=""))
    svc._tier_b_playwright = AsyncMock(return_value=ExtractionResult(success=False, tier_used="B", error="no text", text=""))
    snap = "https://web.archive.org/web/20240101000000/https://example.com"
    with patch("bot.web_extraction_service._wayback_snapshot", AsyncMock(return_value=snap)):
        # The snapshot fetch re-enters _extract_via; make Tier A succeed on the
        # snapshot URL (different host than the original).
        async def fake_a(url):
            if "web.archive.org" in url:
                return ExtractionResult(success=True, tier_used="A", canonical_url=url, text="archived body " * 100)
            return ExtractionResult(success=False, tier_used="A", error="no text", text="")

        svc._tier_a_httpx = fake_a
        res = await svc.extract("https://example.com")
    assert res.success is True
    assert "web.archive.org" in (res.canonical_url or "")


@pytest.mark.asyncio
async def test_wayback_fallback_skipped_on_bot_wall() -> None:
    """Bot-wall failures must NOT trigger a Wayback lookup (the requested page
    is unreachable, not missing) -- keeps the specific bot-wall message."""
    svc = WebExtractionService()
    svc._tier_a_httpx = AsyncMock(
        return_value=ExtractionResult(
            success=True, tier_used="A", canonical_url="https://archive.is/ppfcf", text=_bot_wall_html()
        )
    )
    with patch("bot.web_extraction_service._wayback_snapshot", AsyncMock(return_value="https://web.archive.org/web/2024/https://archive.is/ppfcf")) as wb:
        res = await svc.extract("https://archive.is/ppfcf")
    wb.assert_not_called()
    assert res.bot_wall_marker is not None
