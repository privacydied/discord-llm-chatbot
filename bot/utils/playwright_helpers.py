"""Shared Playwright connection helper.

All places that do browser automation MUST use this to connect.
When PW_SERVER_URL is set (e.g. http://localhost:3006) we connect to the remote
Playwright server.  When it is missing we fall back to a local Chromium launch
so that tests and dev workflows still work without the Docker container.
"""

from __future__ import annotations

import os
from typing import Optional

from .logging import get_logger

logger = get_logger(__name__)

# Remote Playwright server endpoint (Docker playwright:v1.59.1-noble on port 3006).
PW_SERVER_URL: Optional[str] = os.getenv("PW_SERVER_URL", "").strip() or None

_CONNECT_LOGGED = False


def is_remote_playwright_configured() -> bool:
    """Return True if a remote Playwright server URL has been configured."""
    return PW_SERVER_URL is not None


def _maybe_log_remote() -> None:
    global _CONNECT_LOGGED
    if not _CONNECT_LOGGED:
        _CONNECT_LOGGED = True
        if PW_SERVER_URL:
            logger.info(f"Playwright remote server configured: {PW_SERVER_URL}")
        else:
            logger.info("PW_SERVER_URL not set; will launch local Playwright Chromium")


async def connect_browser(playwright) -> Optional:
    """Connect to a Chromium browser — remote server preferred, local fallback.

    Args:
        playwright: the p object from ``async with async_playwright() as p:``

    Returns:
        A connected Browser instance or None if both paths fail.
    """
    _maybe_log_remote()

    # ── 1. Try remote CDP ──────────────────────────────────────────────
    if PW_SERVER_URL:
        try:
            logger.info(f"Connecting to Playwright server {PW_SERVER_URL}")
            browser = await playwright.chromium.connect_over_cdp(
                PW_SERVER_URL, timeout=30_000
            )
            browser._is_remote = True  # type: ignore[attr-defined]
            logger.info("Connected to remote Playwright browser")
            return browser
        except Exception as exc:
            logger.warning(
                f"Remote Playwright connection failed ({PW_SERVER_URL}): {exc}. "
                "Falling back to local launch."
            )

    # ── 2. Local fallback ──────────────────────────────────────────────
    try:
        browser = await playwright.chromium.launch(headless=True)
        browser._is_remote = False  # type: ignore[attr-defined]
        logger.info("Launched local Playwright Chromium")
        return browser
    except Exception as exc:
        logger.error(f"Local browser launch failed: {exc}")
        return None
