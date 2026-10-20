"""Tools: web search, news search, and reading a web page.
[CA][SFT][IV][REH][PA].

The model's knowledge stops at its training cutoff. These tools let it check
anything newer -- releases, ports, announcements, current events -- instead of
arguing from stale memory. All three are read-only network lookups; fetched
text is wrapped as untrusted so instructions inside a page are not followed.
Page fetches go through bot.web, which applies the SSRF checks in
bot.url_safety (no private IPs, metadata hosts, or unsafe redirects).
"""

from __future__ import annotations

from typing import Any

from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

# Bounds. [CMV][PA]
DEFAULT_RESULTS = 5
MAX_RESULTS = 8
MAX_QUERY_CHARS = 200
SEARCH_TIMEOUT_MS = 8000
MAX_NEWS_DAYS = 31
DEFAULT_NEWS_DAYS = 7
NEWS_LIMIT = 6
MAX_PAGE_CHARS = 3500
MAX_URL_CHARS = 2000


def _wrap(content: str, source: str) -> str:
    from bot.url_safety import wrap_untrusted_content

    return wrap_untrusted_content(content, source=source)


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _clean_query(arguments: dict[str, Any], key: str) -> str:
    return " ".join(str(arguments.get(key) or "").split())[:MAX_QUERY_CHARS]


# ── web_search ───────────────────────────────────────────────────────────────

WEB_SEARCH_DESCRIPTION = (
    "Search the web. Your training data is out of date, so use this for anything that may have changed "
    "or happened recently: game/movie/music releases and ports, announcements, current events, prices, "
    "scores, who currently holds a job or title. Also use it whenever a user states a fact you don't "
    "recognise or think is wrong -- check before disagreeing. Returns titles, links and snippets."
)

WEB_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to search for, written like a search-engine query."},
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_RESULTS,
            "default": DEFAULT_RESULTS,
        },
    },
    "required": ["query"],
}


def format_search_results(results: list[Any]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title} -- {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
    return "\n".join(lines)


async def _ddg(ctx: ToolContext, query: str, limit: int) -> list[Any]:
    from bot.search.factory import get_search_provider
    from bot.search.types import SafeSearch, SearchQueryParams

    try:
        safe = SafeSearch(str(ctx.config.get("SEARCH_SAFE", "moderate")).lower())
    except ValueError:
        safe = SafeSearch.MODERATE
    params = SearchQueryParams(query=query, max_results=limit, safesearch=safe, timeout_ms=SEARCH_TIMEOUT_MS)
    try:
        return await get_search_provider().search(params)
    except Exception as exc:  # noqa: BLE001 — fall through to the next source [REH]
        logger.info("tool.web_search.ddg_failed error=%s", exc)
        return []


async def _search_chain(ctx: ToolContext, query: str, limit: int) -> tuple[str, list[Any]]:
    """First source with results: DDG, Google via camofox, Google News, Wikipedia. [REH]

    DDG answers bots with HTTP 202 and nothing when it rate-limits, so an empty
    result moves on rather than reaching the model.
    """
    from . import web_sources as src

    attempts = (
        ("duckduckgo", lambda: _ddg(ctx, query, limit)),
        ("google", lambda: src.camofox_google(query, limit, ctx.config)),
        ("google news", lambda: _news_results(query, limit)),
        ("wikipedia", lambda: src.wikipedia_search(query, limit)),
    )
    for name, run in attempts:
        results = await run()
        logger.info("tool.web_search.source source=%s results=%d", name, len(results))
        if results:
            return name, results
    return "", []


async def _news_results(query: str, limit: int) -> list[Any]:
    from . import web_sources as src

    return src.news_as_results(await src.google_news(query, limit))


async def web_search(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Search with automatic fallback across keyless sources. Never raises."""
    query = _clean_query(arguments, "query")
    if not query:
        return ToolResult.failure("query is required")
    limit = _clamp_int(arguments.get("max_results"), DEFAULT_RESULTS, 1, MAX_RESULTS)
    source, results = await _search_chain(ctx, query, limit)
    logger.info("tool.web_search query=%s source=%s results=%d", query[:80], source or "none", len(results))
    if not results:
        return ToolResult.success(f'No results for "{query}" from any search source. Try shorter wording, lookup_wikipedia for facts, or answer from your own knowledge.')
    return ToolResult.success(_wrap(format_search_results(results), f"web search ({source}): {query}"))


# ── search_news ──────────────────────────────────────────────────────────────

NEWS_DESCRIPTION = (
    "Recent news articles with publish dates (Google News, Guardian as backup). Use for breaking news, "
    "current events, and checking whether something has already released or happened: games, consoles, "
    "films, politics, sport, business. Leave topic empty for today's top stories."
)

NEWS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "Subject to search for. Empty for top headlines."},
        "days": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_NEWS_DAYS,
            "default": DEFAULT_NEWS_DAYS,
            "description": "How many days back to look.",
        },
    },
    "required": [],
}


async def search_news(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Dated news: Google News RSS first, Guardian if that is empty. Never raises."""
    from bot.news.headlines import fetch_headlines, render_digest

    from . import web_sources as src

    topic = _clean_query(arguments, "topic") or None
    days = _clamp_int(arguments.get("days"), DEFAULT_NEWS_DAYS, 1, MAX_NEWS_DAYS)
    items = await src.google_news(topic, NEWS_LIMIT, days=days if topic else None)
    if not items and topic:
        # Nothing that recent; dated older coverage still answers "has it come out". [REH]
        items = await src.google_news(topic, NEWS_LIMIT)
    logger.info("tool.search_news source=google_news topic=%s days=%d results=%d", (topic or "")[:80], days, len(items))
    if items:
        return ToolResult.success(_wrap(src.render_news(items), f"Google News: {topic or 'top stories'}"))
    headlines = await fetch_headlines(topic, cfg=ctx.config, days=days, limit=NEWS_LIMIT)
    logger.info("tool.search_news topic=%s days=%d results=%d", (topic or "")[:80], days, len(headlines))
    if not headlines:
        return ToolResult.success("No news articles found. Try web_search instead.")
    return ToolResult.success(_wrap(render_digest(headlines, topic, days), "Guardian news search"))


# ── lookup_wikipedia ─────────────────────────────────────────────────────────

WIKI_DESCRIPTION = (
    "Wikipedia article summary for a topic: people, places, countries, games, films, events, science, history. Use for factual questions and stats instead of a web search. Set full=true for more of the article."
)

WIKI_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "Article or subject, e.g. 'Super Mario Galaxy 2' or 'crime in Poland'."},
        "full": {"type": "boolean", "default": False, "description": "Return more of the article, not just the intro."},
    },
    "required": ["topic"],
}


async def lookup_wikipedia(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Best-matching English Wikipedia article. Never raises."""
    from . import web_sources as src

    topic = _clean_query(arguments, "topic")
    if not topic:
        return ToolResult.failure("topic is required")
    article = await src.wikipedia_article(topic, full=bool(arguments.get("full")))
    logger.info("tool.lookup_wikipedia topic=%s found=%s", topic[:80], article.title if article else None)
    if article is None:
        return ToolResult.success(f'No Wikipedia article found for "{topic}". Try web_search.')
    return ToolResult.success(_wrap(f"{article.title}\n{article.text}", article.url))


# ── read_webpage ─────────────────────────────────────────────────────────────

READ_DESCRIPTION = "Read the main text of a web page, usually one of your search results, to confirm a detail (a release date, what an article actually says). http/https links only."

READ_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"url": {"type": "string", "description": "Full http(s) URL of the page."}},
    "required": ["url"],
}


async def read_webpage(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Fetch and extract a page's main text through bot.web (SSRF-checked). Never raises."""
    url = str(arguments.get("url") or "").strip()[:MAX_URL_CHARS]
    if not url.lower().startswith(("http://", "https://")):
        return ToolResult.failure("url must start with http:// or https://")
    from bot.web import process_url

    page = await process_url(url)
    text = (page or {}).get("text") or ""
    logger.info("tool.read_webpage url=%s chars=%d", url[:100], len(text))
    if not text:
        return ToolResult.failure(f"could not read that page ({(page or {}).get('error') or 'no text'})")
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + " [truncated]"
    return ToolResult.success(_wrap(text, url))


WEB_SEARCH_SPEC = ToolSpec(
    name="web_search",
    description=WEB_SEARCH_DESCRIPTION,
    parameters=WEB_SEARCH_PARAMETERS,
    handler=web_search,
    # Worst case walks every source: DDG, camofox Google, Google News, Wikipedia. [CMV]
    timeout_s=60.0,
)
NEWS_SPEC = ToolSpec(
    name="search_news",
    description=NEWS_DESCRIPTION,
    parameters=NEWS_PARAMETERS,
    handler=search_news,
    timeout_s=15.0,
)
WIKI_SPEC = ToolSpec(
    name="lookup_wikipedia",
    description=WIKI_DESCRIPTION,
    parameters=WIKI_PARAMETERS,
    handler=lookup_wikipedia,
    timeout_s=15.0,
)
READ_SPEC = ToolSpec(
    name="read_webpage",
    description=READ_DESCRIPTION,
    parameters=READ_PARAMETERS,
    handler=read_webpage,
    timeout_s=25.0,
)
