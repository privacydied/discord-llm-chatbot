"""Shared Playwright connection helper.

All places that do browser automation MUST use this to connect to the
remote Playwright server.  When PW_SERVER_URL is not set, browser-driven
extraction is unavailable -- there is no local-browser fallback.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from playwright.async_api import Browser, BrowserType
from playwright.async_api import Error as PlaywrightError

from .logging import get_logger

logger = get_logger(__name__)


# ---- Cached health state --------------------------------------------------

# Health-probe result cache; updated by check_playwright_health().
_pw_health_available: bool | None = None
_pw_health_last_check: float = 0.0
_PW_HEALTH_TTL: float = 30.0  # seconds
_pw_consecutive_failures: int = 0


def get_playwright_health() -> dict:
    """Return the current Playwright health snapshot (non-blocking).

    Returns a dict with keys: available (bool|None), last_check (float),
    consecutive_failures (int), degraded (bool).
    """
    return {
        "available": _pw_health_available,
        "last_check": _pw_health_last_check,
        "consecutive_failures": _pw_consecutive_failures,
        "degraded": _pw_consecutive_failures >= 2,
    }


async def check_playwright_health() -> bool:
    """Cheap health probe: try to connect to remote Playwright with a 2s timeout.

    Results are cached for ~30 seconds to avoid spamming the server.
    Returns True if Playwright appears available, False otherwise.
    """
    global _pw_health_available, _pw_health_last_check, _pw_consecutive_failures

    now = time.monotonic()
    elapsed = now - _pw_health_last_check
    if elapsed < _PW_HEALTH_TTL and _pw_health_available is not None:
        return _pw_health_available

    ws_url = _pw_server_url()
    if ws_url is None:
        _pw_health_available = False
        _pw_health_last_check = now
        return False

    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            # Use fast connect attempt; 2s timeout is enough to detect dead server
            browser = await asyncio.wait_for(
                pw.chromium.connect_over_cdp(ws_url),
                timeout=2.0,
            )
            try:
                ctx = await browser.new_context()
                await ctx.close()
            finally:
                await browser.close()
        _pw_health_available = True
        _pw_consecutive_failures = 0
        _pw_health_last_check = now
        return True
    except (PlaywrightError, asyncio.TimeoutError, Exception) as exc:
        _pw_consecutive_failures += 1
        _pw_health_available = False
        _pw_health_last_check = now
        logger.warning(
            "playwright:health_check consecutive_failures=%d error=%s",
            _pw_consecutive_failures,
            type(exc).__name__,
        )
        return False


def _pw_server_url() -> Optional[str]:
    """Return the WebSocket endpoint for the remote Playwright server,
    or None if no server is configured.

    Accepts http:// or ws:// URLs and normalises to ws://.
    """
    raw = os.getenv("PW_SERVER_URL", "").strip()
    if not raw:
        return None
    # Normalise http(s):// to ws:// -- Playwright servers speak WS, not CDP.
    if raw.startswith("http://"):
        raw = "ws://" + raw[len("http://") :]
    elif raw.startswith("https://"):
        raw = "wss://" + raw[len("https://") :]
    elif not raw.startswith("ws"):
        # Assume bare host:port, prefix with ws://
        raw = "ws://" + raw
    return raw


async def connect_browser(browser_type: BrowserType) -> Optional[Browser]:
    """Connect to the remote Playwright server if configured.

    Returns a Browser on success, None if PW_SERVER_URL is not set or
    the server is unreachable.
    """
    ws_url = _pw_server_url()
    if ws_url is None:
        return None

    logger.info(f"Connecting to remote Playwright server at {ws_url}")
    try:
        browser = await browser_type.connect(ws_url, timeout=30_000)
        logger.info("Connected to remote Playwright server")
        return browser
    except Exception as exc:
        logger.warning(f"Playwright remote server unreachable at {ws_url}: {exc}")
        return None
