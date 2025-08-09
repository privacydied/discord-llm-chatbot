from __future__ import annotations

from bot.utils.external_api import _normalize_url_for_screenshot
from bot.util.logging import get_logger

logger = get_logger(__name__)

test_urls = [
    "https://x.com/user/status/123",
    "https://www.x.com/user/status/123",
    "https://mobile.twitter.com/user/status/123",
    "https://m.twitter.com/user/status/123",
    "https://vxtwitter.com/user/status/123/photo/1",
    "https://www.fxtwitter.com/user/status/123",
    "https://twitter.com/user/status/123",
    "https://example.com/",
]


def main() -> None:
    logger.info("Checking screenshot normalization")
    for u in test_urls:
        n = _normalize_url_for_screenshot(u)
        logger.info("norm", extra={"detail": {"input": u, "normalized": n}})


if __name__ == "__main__":
    main()
