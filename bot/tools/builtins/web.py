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


async def web_search(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """DuckDuckGo search via the configured provider. Never raises."""
    from bot.search.factory import get_search_provider
    from bot.search.types import SafeSearch, SearchQueryParams

    query = _clean_query(arguments, "query")
    if not query:
        return ToolResult.failure("query is required")
    limit = _clamp_int(arguments.get("max_results"), DEFAULT_RESULTS, 1, MAX_RESULTS)
    try:
        safe = SafeSearch(str(ctx.config.get("SEARCH_SAFE", "moderate")).lower())
    except ValueError:
        safe = SafeSearch.MODERATE
    params = SearchQueryParams(query=query, max_results=limit, safesearch=safe, timeout_ms=SEARCH_TIMEOUT_MS)
    results = await get_search_provider().search(params)
    logger.info("tool.web_search query=%s results=%d", query[:80], len(results))
    if not results:
        return ToolResult.success(f'No results for "{query}". Try different wording, or say you couldn\'t verify it.')
    return ToolResult.success(_wrap(format_search_results(results), f"web search: {query}"))


# ── search_news ──────────────────────────────────────────────────────────────

NEWS_DESCRIPTION = (
    "Recent news articles from the Guardian, with publish dates. Use for breaking news and current events "
    "(politics, world, business, science, sport). Leave topic empty for today's top stories. "
    "For niche topics like games or gadgets, web_search usually finds more."
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
    """Guardian headlines for a topic. Never raises."""
    from bot.news.headlines import fetch_headlines, render_digest

    topic = _clean_query(arguments, "topic") or None
    days = _clamp_int(arguments.get("days"), DEFAULT_NEWS_DAYS, 1, MAX_NEWS_DAYS)
    headlines = await fetch_headlines(topic, cfg=ctx.config, days=days, limit=NEWS_LIMIT)
    logger.info("tool.search_news topic=%s days=%d results=%d", (topic or "")[:80], days, len(headlines))
    if not headlines:
        return ToolResult.success("No news articles found. Try web_search instead.")
    return ToolResult.success(_wrap(render_digest(headlines, topic, days), "Guardian news search"))


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
    timeout_s=15.0,
)
NEWS_SPEC = ToolSpec(
    name="search_news",
    description=NEWS_DESCRIPTION,
    parameters=NEWS_PARAMETERS,
    handler=search_news,
    timeout_s=15.0,
)
READ_SPEC = ToolSpec(
    name="read_webpage",
    description=READ_DESCRIPTION,
    parameters=READ_PARAMETERS,
    handler=read_webpage,
    timeout_s=25.0,
)
