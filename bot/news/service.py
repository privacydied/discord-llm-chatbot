"""Licensed-source fallback chain for article URLs.
[CA][REH][CMV][PA].

Entry point for the router: when generic extraction yields a stub, ask each
publisher-sanctioned provider in turn for the real article. Every stage is
timeout-bounded and failure-tolerant -- the chain returns None rather than
raising, so a news miss degrades to "we could not read it", never a crash.
"""

from __future__ import annotations

import asyncio

from bot.config import load_config
from bot.utils.logging import get_logger

from . import thin_content
from .base import NewsProvider
from .providers.guardian import GuardianNewsProvider
from .providers.rss import RssNewsProvider
from .types import NewsArticle

logger = get_logger(__name__)

# Per-provider wall clock. The whole chain is additionally bounded by the
# caller's own budget. [CMV][PA]
DEFAULT_FETCH_TIMEOUT_S = 8.0


def build_providers(cfg: dict | None = None) -> list[NewsProvider]:
    """Instantiate the enabled providers, in priority order.

    Guardian first: it returns full licensed body text for an exact article
    id. RSS second: broader host coverage, but often summary-only.
    """
    config = cfg if cfg is not None else load_config()
    providers: list[NewsProvider] = []

    guardian = GuardianNewsProvider(config.get("GUARDIAN_API_KEY"))
    if guardian.enabled:
        providers.append(guardian)

    if config.get("NEWS_RSS_ENABLED", True):
        providers.append(RssNewsProvider(enabled=True))

    return providers


async def _try_provider(provider: NewsProvider, url: str, timeout_s: float) -> NewsArticle | None:
    """Run one provider under a timeout, swallowing its failures. [REH]"""
    if not provider.supports(url):
        return None
    try:
        return await asyncio.wait_for(provider.fetch(url), timeout=timeout_s)
    except TimeoutError:
        logger.info("news.provider.timeout provider=%s url=%s", provider.name, url[:120])
    except Exception as exc:  # [REH]
        logger.info("news.provider.failed provider=%s url=%s error=%s", provider.name, url[:120], exc)
    return None


async def resolve_article(
    url: str,
    *,
    cfg: dict | None = None,
    min_chars: int | None = None,
    timeout_s: float | None = None,
) -> NewsArticle | None:
    """Return an article body from a licensed source, or None.

    Args:
        url: The article URL that generic extraction could not read properly.
        cfg: Optional pre-loaded config (avoids a reload on the hot path).
        min_chars: Reject provider bodies shorter than this; defaults to the
            same threshold used to judge the original extraction.
        timeout_s: Per-provider timeout.

    Returns:
        The first article whose body clears the thinness bar, else None.

    """
    config = cfg if cfg is not None else load_config()
    if not config.get("NEWS_FALLBACK_ENABLED", True):
        return None
    if not url:
        return None

    threshold = min_chars if min_chars is not None else config.get("NEWS_MIN_ARTICLE_CHARS", thin_content.DEFAULT_MIN_ARTICLE_CHARS)
    budget = timeout_s if timeout_s is not None else config.get("NEWS_FETCH_TIMEOUT_S", DEFAULT_FETCH_TIMEOUT_S)

    for provider in build_providers(config):
        article = await _try_provider(provider, url, budget)
        if article is None:
            continue
        verdict = thin_content.assess(article.body, min_chars=threshold)
        if verdict.is_thin:
            logger.info(
                "news.provider.thin provider=%s url=%s reason=%s chars=%d",
                provider.name,
                url[:120],
                verdict.reason,
                verdict.char_count,
            )
            continue
        logger.info(
            "news.resolved provider=%s url=%s chars=%d",
            provider.name,
            url[:120],
            verdict.char_count,
        )
        return article
    return None
