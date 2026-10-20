"""Keyless lookup sources behind web_search, search_news and lookup_wikipedia.
[CA][REH][IV][PA][SFT].

DuckDuckGo's HTML endpoint answers bots with HTTP 202 and no results when it
rate-limits, which left the model with nothing to go on. These are the
fallbacks, none needing an account or a paid key:

- Camoufox (self-hosted stealth Firefox, CAMOFOX_URL): Google results through
  a real browser. Used only when CAMOFOX_ACCESS_KEY is set.
- Google News RSS: dated news items; every item carries a publish date, which
  stops "is upcoming" answers about things that already happened.
- Wikipedia API: search and article summaries for factual questions.

Fixed hosts only; the model supplies query text, never a URL. Every function
returns an empty result on any failure and never raises.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote, urlencode

from bot.search.types import SearchResult
from bot.utils.logging import get_logger

logger = get_logger(__name__)

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_URL = "https://en.wikipedia.org/wiki/"
DEFAULT_CAMOFOX_URL = "http://127.0.0.1:9377"
CAMOFOX_USER_ID = "ronin-bot"
GOOGLE_SEARCH_URL = "https://www.google.com/search"
# Wikimedia asks API clients to identify themselves. [IV]
USER_AGENT = "ronin-discord-bot/1.0 (self-hosted Discord bot; lookups only)"
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
MAX_SNIPPET_CHARS = 300
MAX_ARTICLE_CHARS = 3500


def _request_config(total_s: float = 8.0) -> Any:
    from bot.http_client import RequestConfig

    return RequestConfig(connect_timeout=3.0, read_timeout=total_s, total_timeout=total_s, max_retries=0)


async def _http(method: str, url: str, total_s: float = 8.0, **kwargs: Any) -> Any | None:
    """One request through the shared client. None on any failure."""
    from bot.http_client import get_http_client

    try:
        client = await get_http_client()
        response = await client.request(method, url, _request_config(total_s), **kwargs)
    except Exception as exc:  # noqa: BLE001 — a failed source just falls through
        logger.info("web_sources.http_failed url=%s error=%s", url[:80], exc)
        return None
    if response.status_code != 200:
        logger.info("web_sources.http_status url=%s status=%s", url[:80], response.status_code)
        return None
    return response


def _clean(text: str | None, limit: int = MAX_SNIPPET_CHARS) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    text = " ".join(html.unescape(text).split())
    return text[:limit]


# ── Camoufox (Google through a real browser) ─────────────────────────────────

_SERP_LINK_RE = re.compile(r'^- link "(?P<title>(?:[^"\\]|\\.)*)" \[e\d+\]:\s*$')
_SERP_FIELD_RE = re.compile(r"^\s+- (?P<key>/url|cite|text): (?P<value>.*)$")


def parse_google_snapshot(snapshot: str, limit: int) -> list[SearchResult]:
    """Organic results from camofox's Google SERP snapshot.

    Format (camofox extractGoogleSerp)::

        - link "Title" [e5]:
          - /url: https://example.com/page
          - cite: example.com › page
          - text: snippet...
    """
    results: list[SearchResult] = []
    current: dict[str, str] | None = None

    def flush() -> None:
        if current and current.get("/url", "").startswith("http") and "google." not in current["/url"].split("/")[2]:
            results.append(SearchResult(title=current["title"], url=current["/url"], snippet=current.get("text") or None))

    for line in (snapshot or "").splitlines():
        link = _SERP_LINK_RE.match(line)
        if link:
            flush()
            current = {"title": _clean(re.sub(r"\\(.)", r"\1", link.group("title")))}
            continue
        field = _SERP_FIELD_RE.match(line)
        if field and current is not None:
            current[field.group("key")] = _clean(field.group("value"))
        elif not line.startswith("  "):
            flush()
            current = None
    flush()
    return results[:limit]


async def camofox_google(query: str, limit: int, cfg: dict[str, Any]) -> list[SearchResult]:
    """Google results via the self-hosted camofox browser, or [] if unavailable."""
    key = str(cfg.get("CAMOFOX_ACCESS_KEY") or "").strip()
    if not key:
        return []
    base = str(cfg.get("CAMOFOX_URL") or DEFAULT_CAMOFOX_URL).rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}
    url = f"{GOOGLE_SEARCH_URL}?{urlencode({'q': query, 'hl': 'en', 'num': max(10, limit)})}"
    created = await _http("POST", f"{base}/tabs", 25.0, headers=headers, json={"userId": CAMOFOX_USER_ID, "sessionKey": "search", "url": url})
    tab_id = _json(created).get("tabId") or _json(created).get("id")
    if not tab_id:
        return []
    try:
        snap = await _http("GET", f"{base}/tabs/{quote(str(tab_id))}/snapshot", 20.0, headers=headers, params={"userId": CAMOFOX_USER_ID})
        data = _json(snap)
        if "/sorry/" in str(data.get("url", "")):
            logger.info("web_sources.camofox_captcha query=%s", query[:60])
            return []
        return parse_google_snapshot(str(data.get("snapshot") or ""), limit)
    finally:
        await _http("DELETE", f"{base}/tabs/{quote(str(tab_id))}", 10.0, headers=headers, params={"userId": CAMOFOX_USER_ID})


def _json(response: Any) -> dict[str, Any]:
    if response is None:
        return {}
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


# ── Google News RSS ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NewsItem:
    title: str
    url: str
    source: str
    published: datetime | None


_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.DOTALL)


def _tag(block: str, name: str) -> str:
    match = re.search(rf"<{name}\b[^>]*>(.*?)</{name}>", block, re.DOTALL)
    if not match:
        return ""
    value = match.group(1)
    cdata = re.fullmatch(r"\s*<!\[CDATA\[(.*?)\]\]>\s*", value, re.DOTALL)
    return cdata.group(1) if cdata else value


def parse_news_rss(xml: str, limit: int) -> list[NewsItem]:
    items: list[NewsItem] = []
    for block in _ITEM_RE.findall(xml or ""):
        title, link, source = _clean(_tag(block, "title")), _clean(_tag(block, "link"), 500), _clean(_tag(block, "source"))
        if not title or not link.startswith("http"):
            continue
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        try:
            published = parsedate_to_datetime(_clean(_tag(block, "pubDate")))
        except (TypeError, ValueError):
            published = None
        items.append(NewsItem(title=title, url=link, source=source, published=published))
        if len(items) >= limit:
            break
    return items


async def google_news(query: str | None, limit: int, days: int | None = None) -> list[NewsItem]:
    """Dated news items for ``query`` (top stories if empty)."""
    params = {"hl": "en-GB", "gl": "GB", "ceid": "GB:en"}
    url = "https://news.google.com/rss"
    if query:
        params["q"] = f"{query} when:{days}d" if days else query
        url = GOOGLE_NEWS_RSS_URL
    response = await _http("GET", url, 8.0, params=params, headers={"User-Agent": BROWSER_UA})
    return parse_news_rss(response.text if response is not None else "", limit)


def news_as_results(items: list[NewsItem]) -> list[SearchResult]:
    return [SearchResult(title=i.title, url=i.url, snippet=_news_meta(i)) for i in items]


def _news_meta(item: NewsItem) -> str:
    date = item.published.strftime("%d %b %Y") if item.published else "date unknown"
    return f"{item.source or 'news'}, published {date}"


def render_news(items: list[NewsItem]) -> str:
    return "\n".join(f"{n}. [{_news_meta(i)}] {i.title} -- {i.url}" for n, i in enumerate(items, 1))


# ── Wikipedia ────────────────────────────────────────────────────────────────


async def wikipedia_search(query: str, limit: int) -> list[SearchResult]:
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json", "utf8": 1}
    data = _json(await _http("GET", WIKIPEDIA_API_URL, 8.0, params=params, headers={"User-Agent": USER_AGENT}))
    hits = (data.get("query") or {}).get("search") or []
    return [
        SearchResult(
            title=f"{h['title']} (Wikipedia)",
            url=WIKIPEDIA_PAGE_URL + quote("_".join(str(h["title"]).split(" "))),
            snippet=_clean(h.get("snippet")) or None,
        )
        for h in hits
        if h.get("title")
    ]


@dataclass(frozen=True, slots=True)
class WikiArticle:
    title: str
    url: str
    text: str


async def wikipedia_article(topic: str, full: bool = False) -> WikiArticle | None:
    """Best-matching article's intro (or first ~3.5k chars when ``full``)."""
    params: dict[str, Any] = {
        "action": "query",
        "generator": "search",
        "gsrsearch": topic,
        "gsrlimit": 1,
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "utf8": 1,
    }
    if not full:
        params["exintro"] = 1
    data = _json(await _http("GET", WIKIPEDIA_API_URL, 10.0, params=params, headers={"User-Agent": USER_AGENT}))
    pages = list(((data.get("query") or {}).get("pages") or {}).values())
    if not pages or not pages[0].get("extract"):
        return None
    page = pages[0]
    text = re.sub(r"\n{3,}", "\n\n", str(page["extract"])).strip()
    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS] + " [truncated]"
    return WikiArticle(title=str(page.get("title")), url=str(page.get("fullurl") or WIKIPEDIA_PAGE_URL), text=text)
