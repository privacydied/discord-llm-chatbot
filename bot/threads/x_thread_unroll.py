from __future__ import annotations

import asyncio
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
    try:
        from playwright.async_api import async_playwright
    except Exception:
        return None
    timeout_ms = int(max(1.0, timeout_s) * 1000)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context(java_script_enabled=True)
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            return await page.content()
        finally:
            await browser.close()


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
        if not tid:
            return None, "not_status"
    except Exception:
        return None, "normalize_failed"

    # Fetch HTML via Playwright; fallback to fxtwitter/vxtwitter if login wall
    html = await _fetch_html_with_playwright(canonical, timeout_s)
    if not html or ("Sign in" in html and "X" in html and "password" in html):
        # Try mirrors conservatively
        for host in ("fxtwitter.com", "vxtwitter.com"):
            try:
                p = urlparse(canonical)
                mirror = urlunparse(("https", host, p.path, "", "", ""))
                html = await _fetch_html_with_playwright(mirror, min(timeout_s, 10.0))
                if html:
                    break
            except Exception:
                continue

    if not html:
        return None, "fetch_failed"

    # Parse DOM into candidate blocks
    blocks = _parse_tweet_blocks(html, canonical, handle)
    if not blocks:
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
        return None, "no_items"

    joined_text = _format_joined_text(handle or "author", items)

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

