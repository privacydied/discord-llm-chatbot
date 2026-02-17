"""X/Twitter routing helper utilities extracted from Router."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

from bot.x_api_client import XApiClient


def parse_twitter_status_id(url: str) -> Optional[str]:
    """Extract tweet/status ID from a Twitter/X URL."""
    return XApiClient.extract_tweet_id(url)


def extract_primary_tweet_id(url: str) -> Optional[str]:
    """Extract stable primary tweet ID, preferring explicit URL hint fragments."""
    raw_url = str(url or "").strip()
    if not raw_url:
        return None

    try:
        parsed = urlparse(raw_url)
        for params in (parse_qs(parsed.fragment or ""), parse_qs(parsed.query or "")):
            for key in ("ptid", "primary", "tweet_id", "status_id", "id"):
                values = params.get(key) or []
                if not values:
                    continue
                candidate = str(values[0] or "").strip()
                if candidate and candidate.isdigit():
                    return candidate
    except Exception:
        pass

    return parse_twitter_status_id(raw_url)


def canonicalize_twitter_status_url(url: str) -> str:
    """Convert any Twitter status URL to canonical form https://x.com/i/status/{id}."""
    status_id = parse_twitter_status_id(url)
    if status_id:
        return f"https://x.com/i/status/{status_id}"
    return url


def is_twitter_url(url: str) -> bool:
    try:
        u = str(url)
    except Exception:
        return False
    try:
        if parse_twitter_status_id(u):
            return True
    except Exception:
        pass
    try:
        low = u.lower()
    except Exception:
        return False
    return any(
        d in low
        for d in (
            "twitter.com/",
            "x.com/",
            "vxtwitter.com/",
            "fxtwitter.com/",
            "fixupx.com/",
        )
    )


def collect_x_candidate_urls(item: Any) -> List[str]:
    urls: List[str] = []
    try:
        if item.source_type == "url":
            urls.append(str(item.payload))
        elif item.source_type == "embed":
            embed = item.payload
            primary_url = getattr(embed, "url", None)
            if primary_url:
                urls.append(primary_url)
            video = getattr(embed, "video", None)
            if video and getattr(video, "url", None):
                urls.append(video.url)
            image = getattr(embed, "image", None)
            if image and getattr(image, "url", None):
                urls.append(image.url)
            thumb = getattr(embed, "thumbnail", None)
            if thumb and getattr(thumb, "url", None):
                urls.append(thumb.url)
        elif item.source_type == "attachment":
            attachment = item.payload
            url = getattr(attachment, "url", None)
            if url:
                urls.append(url)
            proxy = getattr(attachment, "proxy_url", None)
            if proxy:
                urls.append(proxy)
    except Exception:
        pass
    return [u for u in urls if u]


def is_twitter_thumbnail_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in {
        "pbs.twimg.com",
        "pbs-0.twimg.com",
        "pbs-1.twimg.com",
        "pbs-2.twimg.com",
        "pbs-3.twimg.com",
    }


def is_twitter_media_cdn(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in {
        "pbs.twimg.com",
        "pbs-0.twimg.com",
        "pbs-1.twimg.com",
        "pbs-2.twimg.com",
        "pbs-3.twimg.com",
        "video.twimg.com",
        "ton.twimg.com",
    }


def is_tweet_media_url(url: str) -> bool:
    """Check if URL is valid tweet media, excluding profile/banner metadata images."""
    try:
        u = str(url).lower()
    except Exception:
        return False
    try:
        path = urlparse(u).path or ""
    except Exception:
        return False

    blocked_prefixes = (
        "/profile_images/",
        "/profile_banners/",
        "/card_img/",
        "/ad_img/",
        "/emoji/",
    )
    if any(path.startswith(prefix) for prefix in blocked_prefixes):
        return False

    poster_prefixes = (
        "/amplify_video_thumb/",
        "/ext_tw_video_thumb/",
        "/tweet_video_thumb/",
    )
    if any(prefix in path for prefix in poster_prefixes):
        return False

    return "/media/" in path


def normalize_x_url(url: str) -> str:
    """Normalize X/Twitter URLs to canonical host/path, dropping query/fragment."""
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        aliases = {
            "mobile.twitter.com",
            "www.twitter.com",
            "twitter.com",
            "www.x.com",
            "x.com",
            "fxtwitter.com",
            "www.fxtwitter.com",
            "vxtwitter.com",
            "www.vxtwitter.com",
            "fixupx.com",
            "www.fixupx.com",
        }
        if host in aliases:
            host = "x.com"
        path = p.path or ""
        if path.endswith("/"):
            path = path[:-1]
        return urlunparse(("https", host, path, "", "", ""))
    except Exception:
        return url


def unwrap_x_media_url(url: str) -> str:
    """Unwrap fx/vx API proxy URLs back to the media CDN when possible."""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host in {"api.fxtwitter.com", "api.vxtwitter.com"}:
            params = parse_qs(parsed.query or "")
            for key in ("url", "media_url", "target", "u"):
                values = params.get(key)
                if values:
                    candidate = unquote(values[0])
                    if candidate.startswith("http"):
                        return candidate
        return url
    except Exception:
        return url


def extract_x_status_urls_from_text(
    text: str,
    *,
    is_status_url: Callable[[str], bool],
    canonicalize_status_url: Callable[[str], str],
) -> List[str]:
    """Extract canonical X/Twitter status URLs from text preserving order."""
    urls: List[str] = []
    try:
        for m in re.finditer(r"https?://[^\s<>\"'\[\]{}|\\^`]+", text or "", re.IGNORECASE):
            raw = m.group(0)
            if is_status_url(raw):
                cu = canonicalize_status_url(raw)
                if cu not in urls:
                    urls.append(cu)
    except Exception:
        pass
    return urls


def extract_raw_urls_from_texts(texts: Iterable[str]) -> List[str]:
    """Extract raw URLs from multiple text blobs in-order with de-duplication."""
    raw_urls: List[str] = []
    try:
        url_re = re.compile(r"https?://[^\s<>\"'\[\]{}|\\^`]+", re.IGNORECASE)
        for t in texts:
            for m in url_re.finditer(t or ""):
                u = m.group(0)
                if u and u not in raw_urls:
                    raw_urls.append(u)
    except Exception:
        pass
    return raw_urls


def filter_canonical_x_urls(
    raw_urls: Iterable[str],
    *,
    is_x_url: Callable[[str], bool],
    canonicalize_x_url: Callable[[str], str],
) -> List[str]:
    """Filter URL list to X/Twitter URLs and canonicalize with de-duplication."""
    out: List[str] = []
    for u in raw_urls:
        if is_x_url(u):
            cu = canonicalize_x_url(u)
            if cu not in out:
                out.append(cu)
    return out
