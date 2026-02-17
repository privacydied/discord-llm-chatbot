"""Input harvest helper utilities extracted from Router."""

from __future__ import annotations

import re
from typing import Any, List

_URL_STRICT_PATTERN = r"https?://[^\s<>\"'\[\]{}|\\^`]+"
_URL_LOOSE_PATTERN = r"https?://\S+"


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


def extract_urls_loose(text: str) -> List[str]:
    """Extract URLs using permissive whitespace-based matching."""
    try:
        return re.findall(_URL_LOOSE_PATTERN, text or "")
    except Exception:
        return []


def extract_urls_strict(text: str) -> List[str]:
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
