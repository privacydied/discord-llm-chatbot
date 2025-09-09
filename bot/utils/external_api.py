import httpx
import logging
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse, quote

logger = logging.getLogger(__name__)

SCREENSHOT_CACHE_DIR = Path("cache/screenshots")
SCREENSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_url_for_screenshot(url: str) -> str:
    """Normalizes URLs for the screenshot API (e.g., x.com -> twitter.com)."""
    try:
        # Coerce to string to avoid passing non-str objects (e.g., yarl.URL) into urlparse/quote
        url_str = str(url)
        parsed_url = urlparse(url_str)
        netloc = (parsed_url.netloc or "").lower()
        # Normalize common mirrors and subdomains to canonical twitter.com [IV]
        to_twitter = {
            "x.com",
            "www.x.com",
            "mobile.twitter.com",
            "m.twitter.com",
            "vxtwitter.com",
            "www.vxtwitter.com",
            "fxtwitter.com",
            "www.fxtwitter.com",
        }
        if netloc in to_twitter:
            normalized_parts = parsed_url._replace(netloc="twitter.com")
            normalized_url = urlunparse(normalized_parts)
            logger.debug(f"Normalized screenshot URL from {url} to {normalized_url}")
            return normalized_url
    except Exception as e:
        logger.error(f"URL normalization failed for {url}: {e}", exc_info=True)
    # Always return a string
    return str(url)


async def external_screenshot(url: str) -> str | None:
    """
    Captures a screenshot of a URL using the configurable screenshot API
    and saves it to the local cache.

    Args:
        url (str): The URL to screenshot.

    Returns:
        Optional[str]: The file path to the saved screenshot, or None if failed.
    """
    # Basic input validation [IV]
    if not url or not isinstance(url, str) or not url.strip():
        logger.warning("⚠️ Skipping screenshot: missing URL input")
        return None
    if not url.lower().startswith(("http://", "https://")):
        logger.warning(f"⚠️ Skipping screenshot: unsupported URL scheme: {url}")
        return None

    # Coerce to string early to prevent quote_from_bytes TypeError and handle bytes safely
    original_url_str = str(url)
    normalized_url = _normalize_url_for_screenshot(original_url_str)

    # Defensive: ensure we pass a proper str to urllib.parse.quote
    try:
        if isinstance(normalized_url, (bytes, bytearray)):
            normalized_url_str = bytes(normalized_url).decode("utf-8", errors="replace")
        else:
            normalized_url_str = str(normalized_url)
    except Exception:
        # Fallback to string coercion on any unexpected type
        normalized_url_str = str(normalized_url)

    logger.debug(
        f"🧭 URL types | original={type(url).__name__} normalized={type(normalized_url).__name__} as_str={type(normalized_url_str).__name__}"
    )

    # Load configurable screenshot API parameters
    api_key = os.getenv("SCREENSHOT_API_KEY")
    fallback_enabled = (
        os.getenv("SCREENSHOT_FALLBACK_PLAYWRIGHT", "true").lower() == "true"
    )
    if not api_key:
        logger.warning(
            "⚠️ SCREENSHOT_API_KEY not set. Attempting Playwright fallback..."
        )
        if fallback_enabled:
            return await _playwright_screenshot(normalized_url_str)
        return None

    api_url_base = os.getenv("SCREENSHOT_API_URL", "https://api.screenshotmachine.com")
    device = os.getenv("SCREENSHOT_API_DEVICE", "desktop")
    dimension = os.getenv("SCREENSHOT_API_DIMENSION", "1024x768")
    # Match default format with saved filename extension [CMV]
    format_type = os.getenv("SCREENSHOT_API_FORMAT", "png")
    delay = os.getenv("SCREENSHOT_API_DELAY", "2000")
    cookies = os.getenv("SCREENSHOT_API_COOKIES", "")

    # Construct API URL in the exact format expected by screenshotmachine.com
    # Format: ?key=X&url=Y&device=Z&dimension=W&format=V&delay=U&cookies=T
    # Note: Only encode colons in URL, not forward slashes (per API spec)
    api_url = (
        f"{api_url_base}?key={api_key}"
        f"&url={quote(normalized_url_str, safe='/', encoding='utf-8', errors='strict')}"
        f"&device={device}&dimension={dimension}&format={format_type}&delay={delay}"
    )

    # Add cookies parameter if provided (properly URL-encoded)
    if cookies and cookies.strip():
        # Ensure cookies is a str before quoting
        cookies_str = cookies if isinstance(cookies, str) else str(cookies)
        encoded_cookies = quote(cookies_str, safe="", encoding="utf-8", errors="strict")
        api_url += f"&cookies={encoded_cookies}"
        logger.debug(f"🍪 Added URL-encoded cookies: {encoded_cookies[:50]}...")

    logger.debug(f"⏱️ Using screenshot delay: {delay}ms")
    logger.debug(f"🔗 Final API URL: {api_url}")

    logger.info(
        f"📷 Requesting screenshot from {api_url_base} for {normalized_url} [{dimension}, {format_type}]"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, follow_redirects=True, timeout=60.0)
            response.raise_for_status()

        # Generate a safe filename from the original URL (use coerced string)
        parsed_url = urlparse(original_url_str)
        domain = parsed_url.netloc.replace(".", "_")
        path = parsed_url.path.replace("/", "_").strip("_") or "index"
        # Derive extension from configured format; default to png on unknowns [IV]
        fmt = (format_type or "png").lower()
        if fmt in ("jpg", "jpeg"):
            ext = "jpg"
        elif fmt in ("png",):
            ext = "png"
        elif fmt in ("webp",):
            ext = "webp"
        else:
            ext = "png"
        filename = f"{domain}_{path}.{ext}"
        filepath = SCREENSHOT_CACHE_DIR / filename

        with open(filepath, "wb") as f:
            f.write(response.content)

        logger.info(f"✅ Screenshot saved successfully to {filepath}")
        return str(filepath)

    except httpx.HTTPStatusError as e:
        logger.error(
            f"❌ Screenshot Machine API error: {e.response.status_code} for URL: {original_url_str}"
        )
        if fallback_enabled:
            logger.info("Attempting Playwright fallback after API error...")
            return await _playwright_screenshot(normalized_url_str)
        return None
    except httpx.RequestError as e:
        logger.error(
            f"❌ Failed to connect to Screenshot Machine API for URL: {original_url_str}. Error: {e}"
        )
        if fallback_enabled:
            logger.info("Attempting Playwright fallback after request error...")
            return await _playwright_screenshot(normalized_url_str)
        return None
    except Exception as e:
        logger.error(
            f"❌ Unexpected error during screenshot capture for {original_url_str}: {e}",
            exc_info=True,
        )
        if fallback_enabled:
            logger.info("Attempting Playwright fallback after unexpected error...")
            return await _playwright_screenshot(normalized_url_str)
        return None


async def _playwright_screenshot(url: str) -> str | None:
    """Local fallback: use Playwright to capture a screenshot. [REH][IV]

    Respects env config:
      - SCREENSHOT_PW_VIEWPORT (e.g., 1280x1024)
      - SCREENSHOT_PW_USER_AGENT (optional)
      - SCREENSHOT_PW_TIMEOUT_MS (default 15000)
    """
    try:
        # Lazy import to avoid heavy overhead unless needed
        from playwright.async_api import async_playwright
    except Exception as e:
        logger.error(f"Playwright not available for fallback: {e}")
        return None

    vp = os.getenv("SCREENSHOT_PW_VIEWPORT", "1280x1024")
    ua = os.getenv("SCREENSHOT_PW_USER_AGENT", "")
    timeout_ms = int(os.getenv("SCREENSHOT_PW_TIMEOUT_MS", "15000"))

    # Parse viewport
    try:
        width, height = [int(x) for x in vp.lower().split("x", 1)]
    except Exception:
        width, height = 1280, 1024

    logger.info(f"🧭 Playwright fallback starting for {url} [{width}x{height}]")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                user_agent=ua or None,
            )
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)

            # Produce filename similar to API path building
            parsed = urlparse(url)
            domain = parsed.netloc.replace(".", "_")
            path = parsed.path.replace("/", "_").strip("_") or "index"
            filename = f"{domain}_{path}.png"
            filepath = SCREENSHOT_CACHE_DIR / filename

            await page.screenshot(path=str(filepath), full_page=True)
            await context.close()
            await browser.close()

            logger.info(f"✅ Playwright screenshot saved to {filepath}")
            return str(filepath)

    except Exception as e:
        logger.error(f"❌ Playwright fallback failed for {url}: {e}", exc_info=True)
        return None
