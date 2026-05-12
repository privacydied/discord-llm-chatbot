"""Tests for bot.routing.screenshot_handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.modality import InputItem
from bot.routing.base import RouteContext
from bot.routing.screenshot_handler import ScreenshotHandler, handle_screenshot_url


@pytest.fixture
def handler() -> ScreenshotHandler:
    return ScreenshotHandler()


@pytest.fixture
def url_ctx() -> RouteContext:
    return RouteContext(source_type="url", payload="https://example.com/page")


# ------------------------------------------------------------------ #
# can_handle tests
# ------------------------------------------------------------------ #


def test_can_handle_url(handler, url_ctx):
    assert handler.can_handle(url_ctx) is True


def test_can_handle_non_url(handler):
    non_url = RouteContext(source_type="text", payload="hello")
    assert handler.can_handle(non_url) is False


# ------------------------------------------------------------------ #
# handle tests — success path
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@patch("bot.utils.external_api.external_screenshot")
@patch("bot.routing.screenshot_handler.see_infer")
async def test_handle_success(mock_see, mock_ss, handler, url_ctx):
    mock_ss = AsyncMock(return_value="/tmp/screenshot.png")
    mock_see = AsyncMock(return_value="A webpage with text")

    patch_ss = patch("bot.utils.external_api.external_screenshot", mock_ss)
    patch_see = patch("bot.routing.screenshot_handler.see_infer", mock_see)

    with patch_ss, patch_see:
        result = await handler.handle(url_ctx)

    assert "A webpage with text" in result
    assert "example.com" in result


# ------------------------------------------------------------------ #
# handle tests — external API failure
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@patch("bot.utils.external_api.external_screenshot", return_value=None)
async def test_handle_screenshot_api_empty(mock_ss, handler, url_ctx):
    result = await handler.handle(url_ctx)

    assert "Could not capture" in result
    assert "example.com" in result


# ------------------------------------------------------------------ #
# handle tests — vision analysis failure
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@patch("bot.utils.external_api.external_screenshot", return_value="/tmp/test.png")
@patch("bot.routing.screenshot_handler.see_infer")
async def test_handle_vision_error(mock_see, mock_ss, handler, url_ctx):
    mock_see.side_effect = RuntimeError("VL service unavailable")

    result = await handler.handle(url_ctx)

    assert "could not analyze" in result
    assert "example.com" in result


@pytest.mark.asyncio
@patch("bot.utils.external_api.external_screenshot", return_value="/tmp/test.png")
@patch("bot.routing.screenshot_handler.see_infer", return_value=None)
async def test_handle_vision_empty(mock_see, mock_ss, handler, url_ctx):
    result = await handler.handle(url_ctx)

    assert "no content" in result
    assert "example.com" in result


# ------------------------------------------------------------------ #
# handle tests — exception recovery
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@patch(
    "bot.utils.external_api.external_screenshot",
    side_effect=ConnectionError("Network down"),
)
async def test_handle_exception_recovery(mock_ss, handler, url_ctx):
    result = await handler.handle(url_ctx)

    assert "Failed to screenshot" in result
    assert "example.com" in result


# ------------------------------------------------------------------ #
# Compatibility function tests
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@patch("bot.utils.external_api.external_screenshot", return_value="/tmp/screenshot.png")
@patch("bot.routing.screenshot_handler.see_infer", return_value="Content here")
async def test_handle_screenshot_url_compat(mock_see, mock_ss):
    item = MagicMock()
    item.source_type = "url"
    item.payload = "https://example.com/page"

    result = await handle_screenshot_url(item)

    assert "Content here" in result
