"""
Playwright-based web capture utilities for screenshotting and JS-rendered content extraction.
"""
import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from playwright.async_api import async_playwright, Error

from bot.util.logging import get_logger

# Load configuration from environment variables
USE_PLAYWRIGHT_FALLBACK = os.getenv("USE_PLAYWRIGHT_FALLBACK", "true").lower() == "true"
SCREENSHOT_CACHE_DIR = Path(os.getenv("SCREENSHOT_CACHE_DIR", "/tmp/discord_bot_screenshots"))

logger = get_logger(__name__)

async def _get_playwright_context() -> Tuple:
    """Helper to launch Playwright and return a new page object."""
    p = await async_playwright().start()
    browser = await p.chromium.launch()
    page = await browser.new_page()
    return p, browser, page

async def capture_with_playwright(url: str, timeout: int = 20000) -> Dict[str, Any]:
    """
    Fetches and renders a page with Playwright, returning its text and a screenshot.
    Caches the screenshot and returns a dictionary with content or an error.
    """
    if not USE_PLAYWRIGHT_FALLBACK:
        logger.info("Playwright fallback is disabled by configuration.")
        return {'error': 'Playwright fallback is disabled.'}

    SCREENSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    cache_path = SCREENSHOT_CACHE_DIR / f"{url_hash}.png"

    p, browser, page = None, None, None
    try:
        logger.info(f"📸 Capturing content for {url} with Playwright...")
        p, browser, page = await _get_playwright_context()
        
        await page.goto(url, timeout=timeout, wait_until='networkidle')
        
        text_content = await page.inner_text("body")

        if not cache_path.exists():
            logger.info(f"📸 Screenshot for {url} not in cache, capturing now...")
            screenshot_bytes = await page.screenshot(full_page=True)
            with open(cache_path, "wb") as f:
                f.write(screenshot_bytes)
            logger.info(f"📸 Successfully took and cached screenshot of {url}.")
        else:
            logger.info(f"📸 Found cached screenshot for {url}")

        return {'text': text_content, 'screenshot_path': cache_path, 'error': None}

    except Error as e:
        if "Executable doesn't exist" in str(e):
            logger.error("📛 Playwright browser not installed. Run: `uv run playwright install chromium` to fix this.")
            return {'error': 'BROWSER_NOT_INSTALLED'}
        
        logger.error(f"📛 Playwright capture failed for {url}: {e}", exc_info=True)
        return {'error': str(e)}
    finally:
        if page:
            await page.close()
        if browser:
            await browser.close()
        if p:
            await p.stop()
