"""Shared Playwright connection helper.

All places that do browser automation MUST use this to connect to the
remote Playwright server.  When PW_SERVER_URL is not set, browser-driven
extraction is unavailable -- there is no local-browser fallback.
"""

from __future__ import annotations

from typing import Optional

from playwright.async_api import Browser, BrowserType

from .logging import get_logger

logger = get_logger(__name__)


def _pw_server_url() -> Optional[str]:
    """Return the WebSocket endpoint for the remote Playwright server,
    or None if no server is configured.

    Accepts http:// or ws:// URLs and normalises to ws://.
    """
    import os

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
