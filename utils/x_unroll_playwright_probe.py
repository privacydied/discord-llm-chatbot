from __future__ import annotations

"""
Quick standalone Playwright probe to extract an author's contiguous self-reply thread
from an X/Twitter status URL, and print specific tweet indices (e.g. 7, 13, 22, 26).

Usage:
  uv run playwright install chromium   # one-time
  uv run python utils/x_unroll_playwright_probe.py \
      https://x.com/<handle>/status/<id> --only 7 13 22 26

Notes:
  - This script is API-less and uses Playwright to render the SPA.
  - It filters to tweets authored by the same handle and orders them by timestamp.
  - It prints a compact text-only output for easy verification.
"""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from playwright.async_api import async_playwright


@dataclass
class Tweet:
    idx: int
    id: str
    author: str
    created_at: Optional[str]
    text: str


def _parse_handle_from_url(url: str) -> Optional[str]:
    try:
        p = urlparse(url)
        parts = [x for x in (p.path or "").split("/") if x]
        if len(parts) >= 2 and parts[1] == "status":
            return parts[0]
    except Exception:
        pass
    return None


async def _extract_tweets_from_page(page, author: str) -> List[Tweet]:
    # Each tweet is an <article>. We extract:
    # - tweet id from the first /status/<id> link
    # - author handle from links like /<handle>
    # - text from [data-testid="tweetText"]
    # - timestamp from <time datetime="...">
    items: List[Tuple[str, str, Optional[str], str]] = []
    articles = page.locator("article")
    count = await articles.count()
    for i in range(count):
        a = articles.nth(i)
        try:
            # Tweet ID
            tid = None
            links = a.locator("a[href]")
            lcount = await links.count()
            for j in range(lcount):
                href = await links.nth(j).get_attribute("href")
                if href and "/status/" in href:
                    # /<handle>/status/<id>
                    try:
                        tid = href.split("/status/")[1].split("/")[0]
                    except Exception:
                        tid = None
                    break
            if not tid:
                continue

            # Author
            found_author = None
            for j in range(lcount):
                href = await links.nth(j).get_attribute("href")
                if href and href.startswith("/") and "/status/" not in href:
                    cand = href.strip("/").split("/")[0]
                    if cand and not cand.startswith("i"):
                        found_author = cand
                        break
            if not found_author:
                found_author = author

            # Only keep tweets by the same author (self-replies)
            if not found_author or found_author.lower() != author.lower():
                continue

            # Text
            text = ""
            tt = a.locator('[data-testid="tweetText"]')
            if await tt.count() > 0:
                text = await tt.inner_text()
            if not text:
                # Fallback: grab visible text
                text = (await a.inner_text())[:2000]

            # Timestamp
            ts_iso = None
            tnode = a.locator("time")
            if await tnode.count() > 0:
                ts_iso = await tnode.first.get_attribute("datetime")

            items.append((tid, found_author, ts_iso, text.strip()))
        except Exception:
            continue

    # Sort by timestamp if possible, else keep DOM order
    def _k(v: Tuple[str, str, Optional[str], str]) -> Tuple[int, int]:
        _, _, ts, _ = v
        try:
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return (0, int(dt.timestamp()))
        except Exception:
            pass
        return (1, 0)

    items_sorted = sorted(items, key=_k)
    tweets: List[Tweet] = []
    for i, (tid, auth, ts, text) in enumerate(items_sorted, start=1):
        tweets.append(Tweet(i, tid, auth, ts, text))
    return tweets


async def fetch_thread(url: str, *, timeout_ms: int = 25000, engine: str = "chromium") -> List[Tweet]:
    author = _parse_handle_from_url(url) or ""
    engines = [engine] if engine in ("chromium", "firefox", "webkit") else ["chromium", "firefox", "webkit"]
    last_err: Optional[Exception] = None
    async with async_playwright() as p:
        for eng in engines:
            try:
                browser_type = getattr(p, eng)
                browser = await browser_type.launch(headless=True)
                try:
                    context = await browser.new_context(java_script_enabled=True)
                    page = await context.new_page()
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    try:
                        await page.wait_for_selector("article", timeout=8000)
                    except Exception:
                        pass
                    return await _extract_tweets_from_page(page, author)
                finally:
                    await browser.close()
            except Exception as e:
                last_err = e
                continue
    if last_err:
        raise last_err
    return []


async def amain() -> None:
    ap = argparse.ArgumentParser(description="Playwright X thread probe (self-replies)")
    ap.add_argument("url", help="X status URL e.g. https://x.com/<handle>/status/<id>")
    ap.add_argument("--engine", default="auto", choices=["auto", "chromium", "firefox", "webkit"], help="Browser engine to use (auto tries chromium→firefox→webkit)")
    ap.add_argument("--only", nargs="*", type=int, default=[], help="Specific indices to print (e.g. 7 13 22 26)")
    args = ap.parse_args()

    engine = args.engine if args.engine != "auto" else "auto"
    tweets = await fetch_thread(args.url, engine=engine)
    if not tweets:
        print("No tweets extracted (login wall or no self-replies detected).")
        return

    only = set(args.only)
    for t in tweets:
        if only and t.idx not in only:
            continue
        header = f"[{t.idx}] @{t.author} – {t.created_at or ''}".strip()
        print(header)
        print(t.text)
        print()


if __name__ == "__main__":
    asyncio.run(amain())
