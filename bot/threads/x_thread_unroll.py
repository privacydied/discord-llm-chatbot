from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup

from bot.http_client import RequestConfig, get_http_client
from bot.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TweetItem:
    index: int
    tweet_id: str
    created_at_iso: Optional[str]
    text_plain: str
    urls_resolved: List[str]
    media_summary: Dict[str, Any]


@dataclass
class ThreadContext:
    source: str
    author_handle: str
    author_display: Optional[str]
    canonical_url: str
    tweet_count: int
    items: List[TweetItem]
    joined_text: str
    truncated: bool = False


_X_HOSTS = {
    "x.com",
    "twitter.com",
    "mobile.twitter.com",
    "fxtwitter.com",
    "vxtwitter.com",
    "fixupx.com",
}


def _strip(s: Optional[str]) -> str:
    return (s or "").strip()


def _is_twitter_like(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host in _X_HOSTS or host.endswith(".twitter.com")
    except Exception:
        return False


def _extract_status_id_from_path(path: str) -> Optional[str]:
    try:
        m = re.search(r"/status/(\d{5,20})(?:\D|$)", path)
        return m.group(1) if m else None
    except Exception:
        return None


def _extract_handle_from_path(path: str) -> Optional[str]:
    try:
        parts = [p for p in (path or "").split("/") if p]
        return parts[0] if parts else None
    except Exception:
        return None


async def _expand_tco_if_needed(url: str, timeout_s: float) -> str:
    try:
        p = urlparse(url)
        if p.netloc.lower() != "t.co":
            return url
        http = await get_http_client()
        cfg = RequestConfig(
            connect_timeout=min(timeout_s / 3, 4.0),
            read_timeout=min(timeout_s / 2, 6.0),
            total_timeout=min(timeout_s, 10.0),
            max_retries=1,
        )
        # GET with follow_redirects honors shared client setting
        r = await http.get(url, config=cfg)
        return str(r.request.url) if r is not None else url
    except Exception:
        return url


async def _fetch_html_http(url: str, timeout_s: float) -> Optional[str]:
    """HTTP-only fetch for server-rendered mirrors. [PA][REH]

    Returns page text or None; logs DEBUG diagnostics when enabled.
    """
    try:
        http = await get_http_client()
    except Exception as e:
        try:
            logger.debug(
                "threads.x: http_client_unavailable",
                extra={
                    "subsys": "threads.x",
                    "event": "http_client_unavailable",
                    "detail": {"url": url, "error": str(e)},
                },
            )
        except Exception:
            pass
        return None

    cfg = RequestConfig(
        connect_timeout=min(timeout_s / 3, 2.0),
        read_timeout=min(timeout_s / 2, 4.0),
        total_timeout=min(timeout_s, 6.0),
        max_retries=1,
    )
    try:
        r = await http.get(url, config=cfg)
        if r is None or r.status_code >= 400:
            try:
                logger.debug(
                    "threads.x: http_fetch_non200",
                    extra={
                        "subsys": "threads.x",
                        "event": "http_fetch_non200",
                        "detail": {"url": url, "code": getattr(r, "status_code", None)},
                    },
                )
            except Exception:
                pass
            return None
        text = r.text
        try:
            logger.debug(
                "threads.x: http_fetch_ok",
                extra={
                    "subsys": "threads.x",
                    "event": "http_fetch_ok",
                    "detail": {"url": url, "bytes": len(text or "")},
                },
            )
        except Exception:
            pass
        return text
    except Exception as e:
        try:
            logger.debug(
                "threads.x: http_fetch_error",
                extra={
                    "subsys": "threads.x",
                    "event": "http_fetch_error",
                    "detail": {"url": url, "error": str(e)},
                },
            )
        except Exception:
            pass
        return None


def _canonicalize_status_url(url: str) -> Tuple[str, Optional[str], Optional[str]]:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host not in _X_HOSTS and not host.endswith(".twitter.com"):
            return url, None, None
        # Always normalize to x.com
        canonical = urlunparse(("https", "x.com", p.path, "", "", ""))
        tweet_id = _extract_status_id_from_path(p.path or "")
        handle = _extract_handle_from_path(p.path or "")
        return canonical, tweet_id, handle
    except Exception:
        return url, None, None


def _parse_tweet_blocks(
    html: str, canonical_url: str, op_handle: Optional[str]
) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    blocks: List[Dict[str, Any]] = []

    # Prefer ARTICLE containers (x.com and mirrors commonly use them)
    articles = soup.find_all("article")
    if not articles:
        # Fallback: common mirror structures
        articles = soup.find_all("div", attrs={"data-testid": re.compile("tweet", re.I)})

    # Special-case: FixTweet/vx mirrors often contain only meta tags + redirect; synthesize a single block
    # from meta description when no tweet ARTICLEs are present. [REH][PA]
    if not articles:
        try:
            # Resolve tweet_id from canonical link if present
            tw_id = None
            canonical_link = soup.find("link", attrs={"rel": re.compile("canonical", re.I)})
            can_href = canonical_link.get("href") if canonical_link else None
            if can_href:
                m = re.search(r"/status/(\d{5,20})(?:\D|$)", can_href)
                if m:
                    tw_id = m.group(1)

            # Author from twitter:creator meta or canonical path
            author = None
            m_creator = soup.find("meta", attrs={"property": re.compile("twitter:creator", re.I)})
            if m_creator and m_creator.get("content"):
                c = m_creator.get("content")
                if c.startswith("@"):
                    author = c[1:]
            if not author and can_href:
                pth = urlparse(can_href).path or ""
                author = _extract_handle_from_path(pth)

            # Description text
            desc = None
            for prop in ("og:description", "twitter:description"):
                mdesc = soup.find("meta", attrs={"property": prop})
                if mdesc and mdesc.get("content"):
                    desc = _strip(mdesc.get("content"))
                    break
            if not desc:
                mdesc = soup.find("meta", attrs={"name": "description"})
                if mdesc and mdesc.get("content"):
                    desc = _strip(mdesc.get("content"))

            if tw_id and author and desc:
                blocks.append(
                    {
                        "tweet_id": tw_id,
                        "author": author,
                        "created_at_iso": None,
                        "text": desc,
                        "urls": [can_href] if can_href else [],
                        "media": {},
                        "_dom_index": 0,
                    }
                )
        except Exception:
            # swallow and continue with empty blocks
            pass

    for idx, art in enumerate(articles):
        try:
            # Tweet ID via status link
            tw_id = None
            for a in art.find_all("a", href=True):
                href = a.get("href", "")
                m = re.search(r"/status/(\d{5,20})(?:\D|$)", href)
                if m:
                    tw_id = m.group(1)
                    break
            if not tw_id:
                continue

            # Author handle: prefer header link, else derive from canonical
            author = None
            for a in art.find_all("a", href=True):
                href = a.get("href", "")
                if href.startswith("/") and "/status/" not in href:
                    # likely /<handle>
                    cand = href.strip("/").split("/")[0]
                    if cand and not cand.startswith("i"):
                        author = cand
                        break
            if not author:
                author = op_handle

            # Filter non-author tweets strictly
            if op_handle and author and author.lower() != op_handle.lower():
                continue

            # Timestamp
            created_iso = None
            tt = art.find("time")
            if tt and tt.has_attr("datetime"):
                created_iso = _strip(tt.get("datetime"))

            # Text: prefer data-testid=tweetText
            text = ""
            tt_node = art.find(attrs={"data-testid": "tweetText"})
            if tt_node:
                text = _strip(tt_node.get_text(" "))
            if not text:
                # Mirrors may use p tags
                ps = art.find_all("p")
                if ps:
                    text = _strip(" ".join(p.get_text(" ") for p in ps))
            if not text:
                text = _strip(art.get_text(" "))

            # URLs resolved (best-effort)
            urls: List[str] = []
            for a in art.find_all("a", href=True):
                href = _strip(a.get("href"))
                if not href:
                    continue
                if href.startswith("/"):
                    # Make absolute for x.com
                    urls.append("https://x.com" + href)
                elif href.startswith("http"):
                    urls.append(href)

            # Media summary
            img_tags = art.find_all("img")
            images = 0
            alt_texts: List[str] = []
            for im in img_tags:
                # Skip profile/emoji heuristics
                src = (im.get("src") or "").lower()
                alt = _strip(im.get("alt"))
                if "profile_images" in src or "emoji" in src:
                    continue
                images += 1
                if alt:
                    alt_texts.append(alt)

            media = {"images": images}
            if alt_texts:
                media["alt_text"] = "; ".join(alt_texts[:2])

            blocks.append(
                {
                    "tweet_id": tw_id,
                    "author": author,
                    "created_at_iso": created_iso,
                    "text": text,
                    "urls": urls,
                    "media": media,
                    "_dom_index": idx,
                }
            )
        except Exception:
            continue

    # Deduplicate by tweet_id preserving DOM order
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for b in blocks:
        tid = str(b.get("tweet_id"))
        if tid and tid not in seen:
            seen.add(tid)
            uniq.append(b)
    return uniq


def _format_joined_text(author: str, items: List[TweetItem]) -> str:
    n = len(items)
    parts: List[str] = []
    for i, it in enumerate(items, start=1):
        ts = it.created_at_iso
        try:
            # Normalize to UTC without tz suffix if parseable
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass
        header = f"[{i}/{n}] @{author} – {ts}" if ts else f"[{i}/{n}] @{author}"
        parts.append(header)
        parts.append(it.text_plain)
        parts.append("")
    return "\n".join(parts).strip()


async def _fetch_html_with_playwright(url: str, timeout_s: float) -> Optional[str]:
    """Fetch page HTML using Playwright with guarded errors.

    Returns None on any import/launch/navigation failure so callers can gracefully
    fallback. Emits DEBUG logs for visibility when LOG_LEVEL=DEBUG. [REH][PA]
    """
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        # Playwright not installed/available in this environment
        try:
            logger.debug(
                "threads.x: playwright_unavailable",
                extra={
                    "subsys": "threads.x",
                    "event": "playwright_unavailable",
                    "detail": {"url": url, "error": str(e)},
                },
            )
        except Exception:
            pass
        return None

    timeout_ms = int(max(1.0, timeout_s) * 1000)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(java_script_enabled=True)
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                html = await page.content()
                try:
                    logger.debug(
                        "threads.x: playwright_fetch_ok",
                        extra={
                            "subsys": "threads.x",
                            "event": "playwright_fetch_ok",
                            "detail": {"url": url, "bytes": len(html or "")},
                        },
                    )
                except Exception:
                    pass
                return html
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass
    except Exception as e:
        # Launch/navigation failure: log at DEBUG and fallback cleanly
        try:
            logger.debug(
                "threads.x: playwright_fetch_error",
                extra={
                    "subsys": "threads.x",
                    "event": "playwright_fetch_error",
                    "detail": {"url": url, "error": str(e.__class__.__name__) + ": " + str(e)},
                },
            )
        except Exception:
            pass
        return None


async def unroll_author_thread(
    url: str,
    *,
    timeout_s: float,
    max_tweets: int,
    max_chars: int,
) -> Tuple[Optional[ThreadContext], Optional[str]]:
    # Eligibility gate: minimal, cheap checks
    try:
        if not _is_twitter_like(url):
            return None, "not_twitter"
        url = await _expand_tco_if_needed(url, timeout_s)
        canonical, tid, handle = _canonicalize_status_url(url)
        try:
            logger.debug(
                "threads.x: unroll_normalized",
                extra={
                    "subsys": "threads.x",
                    "event": "unroll_normalized",
                    "detail": {"canonical": canonical, "tweet_id": tid, "handle": handle},
                },
            )
        except Exception:
            pass
        if not tid:
            return None, "not_status"
    except Exception:
        return None, "normalize_failed"

    # Phase 1A: JSON mirror probe (fx/vx) to stitch minimal thread without browsers [PA]
    try:
        ctx_json = await _unroll_via_mirror_json(tid, handle, timeout_s, max_tweets, max_chars)
        if ctx_json is not None and getattr(ctx_json, "joined_text", None):
            try:
                logger.debug(
                    "threads.x: json_probe_ok",
                    extra={
                        "subsys": "threads.x",
                        "event": "json_probe_ok",
                        "detail": {"tweets": ctx_json.tweet_count},
                    },
                )
            except Exception:
                pass
            return ctx_json, None
    except Exception:
        try:
            logger.debug(
                "threads.x: json_probe_error",
                extra={"subsys": "threads.x", "event": "json_probe_error"},
            )
        except Exception:
            pass

    # Phase 1B: HTTP-only server-rendered mirrors (fast path, no browser)
    html: Optional[str] = None
    try:
        p = urlparse(canonical)
        for host in ("fxtwitter.com", "vxtwitter.com", "fixupx.com"):
            mirror = urlunparse(("https", host, p.path, "", "", ""))
            try:
                logger.debug(
                    "threads.x: http_probe_start",
                    extra={
                        "subsys": "threads.x",
                        "event": "http_probe_start",
                        "detail": {"mirror": mirror},
                    },
                )
            except Exception:
                pass
            html_try = await _fetch_html_http(mirror, min(timeout_s, 6.0))
            if not html_try:
                continue
            blocks_try = _parse_tweet_blocks(html_try, canonical, handle)
            try:
                logger.debug(
                    "threads.x: http_probe_parse",
                    extra={
                        "subsys": "threads.x",
                        "event": "http_probe_parse",
                        "detail": {"mirror": mirror, "blocks": len(blocks_try)},
                    },
                )
            except Exception:
                pass
            if blocks_try:
                html = html_try
                break
    except Exception:
        # Non-fatal; proceed to Playwright path
        pass

    # Phase 2: Playwright; fallback to mirrors if login wall
    if html is None:
        html = await _fetch_html_with_playwright(canonical, timeout_s)
    # Login wall/empty → try mirrors
    if not html or ("Sign in" in html and "X" in html and "password" in html):
        # Try mirrors conservatively
        for host in ("fxtwitter.com", "vxtwitter.com"):
            try:
                p = urlparse(canonical)
                mirror = urlunparse(("https", host, p.path, "", "", ""))
                try:
                    logger.debug(
                        "threads.x: mirror_probe",
                        extra={
                            "subsys": "threads.x",
                            "event": "mirror_probe",
                            "detail": {"mirror": mirror},
                        },
                    )
                except Exception:
                    pass
                html = await _fetch_html_with_playwright(mirror, min(timeout_s, 10.0))
                if html:
                    break
            except Exception:
                continue

    if not html:
        try:
            logger.debug(
                "threads.x: fetch_failed",
                extra={
                    "subsys": "threads.x",
                    "event": "fetch_failed",
                    "detail": {"url": canonical},
                },
            )
        except Exception:
            pass
        return None, "fetch_failed"

    # Parse DOM into candidate blocks
    blocks = _parse_tweet_blocks(html, canonical, handle)
    try:
        logger.debug(
            "threads.x: parse_result",
            extra={
                "subsys": "threads.x",
                "event": "parse_result",
                "detail": {"blocks": len(blocks)},
            },
        )
    except Exception:
        pass
    if not blocks:
        # Optional X API probe as a final chance before giving up [REH]
        ctx_api = await _maybe_xapi_unroll(canonical, tid, handle, timeout_s, max_tweets, max_chars)
        if ctx_api is not None:
            return ctx_api, None
        return None, "dom_mismatch"

    # Find OP index
    op_idx = next((i for i, b in enumerate(blocks) if b.get("tweet_id") == tid), None)
    if op_idx is None:
        # fall back to closest in DOM
        op_idx = 0

    # Build contiguous author-only chain around OP
    def is_author(b: Dict[str, Any]) -> bool:
        a = _strip(b.get("author"))
        return bool(handle) and a.lower() == handle.lower()

    # Expand backward
    start = op_idx
    while start - 1 >= 0 and is_author(blocks[start - 1]):
        start -= 1
    # Expand forward
    end = op_idx
    while end + 1 < len(blocks) and is_author(blocks[end + 1]):
        end += 1
    chain = blocks[start : end + 1]

    # Sort by timestamp if available; otherwise stable by DOM
    def ts_key(b: Dict[str, Any]) -> Tuple[int, int]:
        t = b.get("created_at_iso") or b.get("created_at")
        try:
            dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            return (0, int(dt.timestamp()))
        except Exception:
            return (1, int(b.get("_dom_index", 0)))

    chain_sorted = sorted(chain, key=ts_key)

    # Enforce limits
    items: List[TweetItem] = []
    agg_chars = 0
    truncated = False
    for i, b in enumerate(chain_sorted, start=1):
        text = _strip(b.get("text"))
        extra = len(text)
        if (len(items) + 1) > max_tweets or (agg_chars + extra) > max_chars:
            truncated = True
            break
        urls = [u for u in b.get("urls", []) if u]
        media = b.get("media") or {}
        items.append(
            TweetItem(
                index=i,
                tweet_id=str(b.get("tweet_id")),
                created_at_iso=_strip(b.get("created_at_iso")),
                text_plain=text,
                urls_resolved=urls,
                media_summary=media,
            )
        )
        agg_chars += extra

    if not items:
        # Optional X API probe as a final chance before giving up [REH]
        ctx_api = await _maybe_xapi_unroll(canonical, tid, handle, timeout_s, max_tweets, max_chars)
        if ctx_api is not None:
            return ctx_api, None
        return None, "no_items"

    joined_text = _format_joined_text(handle or "author", items)
    try:
        logger.debug(
            "threads.x: unroll_ok_internal",
            extra={
                "subsys": "threads.x",
                "event": "unroll_ok_internal",
                "detail": {"tweets": len(items), "truncated": truncated},
            },
        )
    except Exception:
        pass

    ctx = ThreadContext(
        source="twitter-thread",
        author_handle=handle or "",
        author_display=None,
        canonical_url=canonical,
        tweet_count=len(items),
        items=items,
        joined_text=joined_text,
        truncated=truncated,
    )
    return ctx, None


async def _unroll_via_mirror_json(
    tweet_id: Optional[str],
    author_handle: Optional[str],
    timeout_s: float,
    max_tweets: int,
    max_chars: int,
) -> Optional[ThreadContext]:
    """Build a minimal author-only chain using fx/vx JSON endpoints. [PA][REH]

    - Fetch fx JSON for root; if unavailable, try vx JSON.
    - If JSON exposes a quoted tweet (same author) or raw_text link, chase within caps.
    """
    if not tweet_id or not author_handle:
        return None

    try:
        logger.debug(
            "threads.x: json_probe_start",
            extra={"subsys": "threads.x", "event": "json_probe_start", "detail": {"id": tweet_id}},
        )
    except Exception:
        pass

    try:
        http = await get_http_client()
    except Exception:
        return None
    cfg = RequestConfig(
        connect_timeout=min(timeout_s / 3, 2.0),
        read_timeout=min(timeout_s / 2, 4.0),
        total_timeout=min(timeout_s, 6.0),
        max_retries=1,
    )

    async def _fx(id_: str) -> Optional[Dict[str, Any]]:
        try:
            r = await http.get(f"https://api.fxtwitter.com/Tweet/status/{id_}", config=cfg)
            if r is None or r.status_code >= 400:
                return None
            return r.json()
        except Exception:
            return None

    async def _vx(id_: str) -> Optional[Dict[str, Any]]:
        try:
            r = await http.get(f"https://api.vxtwitter.com/Tweet/status/{id_}", config=cfg)
            if r is None or r.status_code >= 400:
                return None
            return r.json()
        except Exception:
            return None

    items: List[TweetItem] = []
    seen_ids: set[str] = set()
    current_id: Optional[str] = str(tweet_id)
    depth = 0

    while current_id and depth < max_tweets:
        depth += 1
        data = await _fx(current_id) or await _vx(current_id)
        if not data:
            break

        # Normalize fx/vx root tweet fields
        t = None
        if isinstance(data.get("tweet"), dict):
            t = data["tweet"]
        else:
            # vx puts fields at top-level; map to t-like
            if data.get("tweetID") and data.get("text"):
                t = {
                    "id": data.get("tweetID"),
                    "text": data.get("text"),
                    "author": {"screen_name": data.get("user_screen_name")},
                    "created_timestamp": data.get("date_epoch"),
                    "quote": {
                        "id": (data.get("qrt") or {}).get("tweetID"),
                        "author": {"screen_name": (data.get("qrt") or {}).get("user_screen_name")},
                    },
                    "raw_text": None,
                }
        if not t:
            break

        tid = str(t.get("id")) if t.get("id") else current_id
        if tid in seen_ids:
            break
        seen_ids.add(tid)

        # Author handle check
        a = None
        try:
            a = (t.get("author") or {}).get("screen_name")
        except Exception:
            a = None
        if not a:
            a = author_handle
        if not a or a.lower() != (author_handle or "").lower():
            break

        # Text and timestamp
        text = _strip((t.get("text") or ""))
        ts_iso = None
        try:
            ts = t.get("created_timestamp")
            if isinstance(ts, (int, float)):
                ts_iso = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            ts_iso = None

        if (len(items) + 1) > max_tweets or (sum(len(it.text_plain) for it in items) + len(text)) > max_chars:
            break
        items.append(
            TweetItem(
                index=len(items) + 1,
                tweet_id=tid,
                created_at_iso=ts_iso,
                text_plain=text,
                urls_resolved=[],
                media_summary={},
            )
        )

        # Follow quoted tweet if same author; else try raw_text facets
        nxt = None
        try:
            q = t.get("quote") or {}
            qid = q.get("id")
            qaq = (q.get("author") or {}).get("screen_name")
            if qid and qaq and qaq.lower() == (author_handle or "").lower():
                nxt = str(qid)
        except Exception:
            nxt = None
        if not nxt:
            try:
                raw = t.get("raw_text") or {}
                for f in raw.get("facets") or []:
                    if f.get("type") == "url":
                        rep = _strip(f.get("replacement"))
                        if "/status/" in rep:
                            m = re.search(r"/status/(\d{5,20})(?:\D|$)", rep)
                            if m:
                                nxt = m.group(1)
                                break
            except Exception:
                pass
        current_id = nxt

    if not items:
        return None

    # We walked backwards via quotes → reverse to chronological
    items = list(reversed(items))
    return ThreadContext(
        source="twitter-thread-json",
        author_handle=author_handle or "",
        author_display=None,
        canonical_url=f"https://x.com/{author_handle}/status/{tweet_id}",
        tweet_count=len(items),
        items=items,
        joined_text=_format_joined_text(author_handle or "author", items),
        truncated=False,
    )


async def _maybe_xapi_unroll(
    canonical_url: str,
    tweet_id: Optional[str],
    author_handle: Optional[str],
    timeout_s: float,
    max_tweets: int,
    max_chars: int,
) -> Optional[ThreadContext]:
    """Optional X API fallback to collect self-replies within a conversation. [AS][REH]

    Uses v2 endpoints with Bearer token from env. Returns ThreadContext or None.
    """
    allow = str(os.getenv("TWITTER_UNROLL_ALLOW_X_API", "false")).lower() == "true"
    token = os.getenv("X_API_BEARER_TOKEN")
    if not allow or not token or not tweet_id or not author_handle:
        return None

    http = await get_http_client()
    cfg = RequestConfig(
        connect_timeout=min(timeout_s / 3, 2.0),
        read_timeout=min(timeout_s / 2, 4.0),
        total_timeout=min(timeout_s, 6.0),
        max_retries=1,
    )
    headers = {"Authorization": f"Bearer {token}"}

    r = await http.get(
        f"https://api.twitter.com/2/tweets/{tweet_id}",
        config=cfg,
        params={"tweet.fields": "author_id,conversation_id,created_at"},
        headers=headers,
    )
    if r is None or r.status_code >= 400:
        return None
    try:
        root = (r.json() or {}).get("data") or {}
    except Exception:
        return None
    conv_id = root.get("conversation_id")
    if not conv_id:
        return None

    sr = await http.get(
        "https://api.twitter.com/2/tweets/search/recent",
        config=cfg,
        params={
            "query": f"conversation_id:{conv_id} from:{author_handle}",
            "max_results": "100",
            "tweet.fields": "created_at,author_id,conversation_id,text",
        },
        headers=headers,
    )
    if sr is None or sr.status_code >= 400:
        return None
    try:
        data = sr.json() or {}
        rows = list(data.get("data") or [])
    except Exception:
        rows = []
    if root and not any(str(t.get("id")) == str(root.get("id")) for t in rows):
        rows.append(root)
    if not rows:
        return None

    def _key(t):
        ts = t.get("created_at")
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return (0, int(dt.timestamp()))
        except Exception:
            return (1, 0)

    rows_sorted = sorted(rows, key=_key)
    items: List[TweetItem] = []
    agg = 0
    for i, t in enumerate(rows_sorted, start=1):
        tx = _strip(t.get("text"))
        if (len(items) + 1) > max_tweets or (agg + len(tx)) > max_chars:
            break
        items.append(
            TweetItem(
                index=i,
                tweet_id=str(t.get("id")),
                created_at_iso=_strip(t.get("created_at")),
                text_plain=tx,
                urls_resolved=[],
                media_summary={},
            )
        )
        agg += len(tx)
    if not items:
        return None
    return ThreadContext(
        source="twitter-thread-xapi",
        author_handle=author_handle or "",
        author_display=None,
        canonical_url=canonical_url,
        tweet_count=len(items),
        items=items,
        joined_text=_format_joined_text(author_handle or "author", items),
        truncated=False,
    )
