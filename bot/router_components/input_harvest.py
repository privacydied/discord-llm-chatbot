"""Input harvest helper utilities extracted from Router."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

_URL_STRICT_PATTERN = r"https?://[^\s<>\"'\[\]{}|\\^`]+"
_URL_LOOSE_PATTERN = r"https?://\S+"
_DISCORD_MENTION_TOKEN_PATTERN = r"<[@#][^>]+>"


def is_text_attachment(attachment: Any) -> bool:
    """Return True for plain-text style attachments."""
    try:
        name = (getattr(attachment, "filename", "") or "").lower()
        ctype = (getattr(attachment, "content_type", "") or "").lower()
    except Exception:
        return False
    return name.endswith(".txt") or ctype.startswith("text/")


def all_attachments_are_text(attachments: Any) -> bool:
    """Return True when a non-empty attachment iterable contains only text files."""
    try:
        atts = list(attachments or [])
    except Exception:
        return False
    return bool(atts) and all(is_text_attachment(a) for a in atts)


def has_meaningful_text(text: str) -> bool:
    """Relaxed chat signal check for routing defaults."""
    try:
        s = (text or "").strip()
        if not s:
            return False
        if re.search(r"[A-Za-z0-9]", s):
            return True
        if s in {"?", "!", "…", "??", "!!"}:
            return True
        try:
            if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", s):
                return True
        except re.error:
            if re.search(r"[^\s\w]", s):
                return True
        if len(s) <= 3 and re.match(r"^[^\s]+$", s):
            return True
        return bool(s)
    except Exception:
        return bool(text and str(text).strip())


def has_explicit_media_intent(text: str) -> bool:
    """Detect explicit media-analysis intent in user text."""
    try:
        s = (text or "").lower()
        if not s:
            return False
        keywords = (
            "summarize this video",
            "summarise this video",
            "summarize the video",
            "summarise the video",
            "summarize video",
            "analyze this video",
            "analyse this video",
            "analyze video",
            "analyse video",
            "what's in this pic",
            "whats in this pic",
            "what is in this pic",
            "what's in this image",
            "analyze this image",
            "analyse this image",
            "analyze this picture",
            "analyse this picture",
            "read this thread",
            "analyze this thread",
            "analyse this thread",
            "summarize this link",
            "summarise this link",
            "summarize the link",
            "summarise the link",
            "summarize this post",
            "summarise this post",
        )
        return any(k in s for k in keywords)
    except Exception:
        return False


def is_direct_image_url(url: str) -> bool:
    """Return True when URL appears to target a direct image asset."""
    try:
        u = str(url).lower()
    except Exception:
        return False
    return bool(re.search(r"\.(jpe?g|png|webp)(?:\?|#|$)", u) or re.search(r"[?&]format=(jpe?g|png|webp)(?:&|$)", u))


def extract_urls_loose(text: str) -> list[str]:
    """Extract URLs using permissive whitespace-based matching."""
    try:
        return re.findall(_URL_LOOSE_PATTERN, text or "")
    except Exception:
        return []


def extract_urls_strict(text: str) -> list[str]:
    """Extract URLs using stricter boundary matching."""
    try:
        return re.findall(_URL_STRICT_PATTERN, text or "")
    except Exception:
        return []


def strip_urls(text: str) -> str:
    """Remove strict URL matches from text."""
    try:
        return re.sub(_URL_STRICT_PATTERN, "", text or "").strip()
    except Exception:
        return (text or "").strip()


def strip_discord_mentions_and_urls(text: str) -> str:
    """Strip Discord mention/channel tokens and loose URLs from text."""
    try:
        out = re.sub(_DISCORD_MENTION_TOKEN_PATTERN, "", text or "")
        out = re.sub(_URL_LOOSE_PATTERN, "", out)
        return out.strip()
    except Exception:
        return (text or "").strip()


def existing_url_payloads(items: Sequence[Any], *, strip_payload: bool = False) -> set[str]:
    """Collect URL payloads from InputItem-like entries into a dedupe set."""
    urls: set[str] = set()
    for it in items or []:
        try:
            if getattr(it, "source_type", None) != "url":
                continue
            raw = str(getattr(it, "payload", ""))
            urls.add(raw.strip() if strip_payload else raw)
        except Exception as exc:
            logger.debug(f"URL collection failed: {exc}")
            continue
    return urls


def append_unique_url_items(
    items: list[Any],
    urls: Iterable[str],
    *,
    item_ctor: Callable[..., Any],
    strip_key: bool = False,
    existing_urls: set[str] | None = None,
) -> int:
    """Append URL InputItem-like entries in-order, skipping duplicates."""
    seen = existing_urls if existing_urls is not None else existing_url_payloads(items)
    added = 0
    for u in urls or []:
        try:
            key = str(u).strip() if strip_key else str(u)
        except Exception as exc:
            logger.debug(f"key generation failed: {exc}")
            key = ""
        if not key or key in seen:
            continue
        items.append(item_ctor(source_type="url", payload=u, order_index=len(items) + 1))
        seen.add(key)
        added += 1
    return added


def append_embed_related_urls(found_urls: list[str], embeds: Iterable[Any]) -> None:
    """Append URL/video/author URLs from embeds into list with in-list dedupe."""
    for em in embeds or []:
        try:
            em_url = getattr(em, "url", None)
            if em_url and em_url not in found_urls:
                found_urls.append(em_url)
            v = getattr(em, "video", None)
            if v and hasattr(v, "url"):
                vu = getattr(v, "url", None)
                if vu and vu not in found_urls:
                    found_urls.append(vu)
            a = getattr(em, "author", None)
            if a and hasattr(a, "url"):
                au = getattr(a, "url", None)
                if au and au not in found_urls:
                    found_urls.append(au)
        except Exception as exc:
            logger.debug(f"embed URL extraction failed: {exc}")
            continue
