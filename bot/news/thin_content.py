"""Detect extraction results that are stubs rather than articles.
[CA][CMV][IV].

Generic web extraction reports success whenever it recovers *any* text. For
a metered or subscriber-only page that text is typically a headline plus a
subscription prompt. Feeding that to the model produces confident commentary
on an article nobody read.

This module answers one question -- "is this actually an article?" -- so the
caller can go ask a licensed source instead. It performs no bypass of any
kind; a thin verdict simply routes the request to a publisher-sanctioned
channel, and when none is available the caller reports the shortfall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Below this many characters, a "successful" extraction is treated as a stub
# rather than an article body. Tuned above a typical headline + standfirst +
# subscription prompt (~400 chars) and below a short wire story (~1200). [CMV]
DEFAULT_MIN_ARTICLE_CHARS = 800

# Phrases that indicate the extracted text is a subscription interstitial
# rather than editorial content. Matched case-insensitively. [CMV]
SUBSCRIPTION_MARKERS: tuple[str, ...] = (
    "subscribe to continue",
    "to continue reading",
    "continue reading this article",
    "already a subscriber",
    "this article is for subscribers",
    "subscribers only",
    "create a free account to read",
    "sign in to read",
    "register to continue",
    "you have reached your limit",
    "you've reached your limit",
    "free articles remaining",
    "start your free trial",
    "become a member to read",
)

# A marker only counts when the surrounding text is short enough that the
# marker plausibly *is* the content, rather than a footer on a real article. [CMV]
MARKER_MAX_CHARS = 2500

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ThinVerdict:
    """Why a body was judged a stub (or not)."""

    is_thin: bool
    reason: str
    char_count: int

    def __bool__(self) -> bool:
        return self.is_thin


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()


def find_subscription_marker(text: str) -> str | None:
    """Return the first subscription marker present in ``text``, if any."""
    lowered = _normalize(text).lower()
    for marker in SUBSCRIPTION_MARKERS:
        if marker in lowered:
            return marker
    return None


def assess(text: str | None, min_chars: int = DEFAULT_MIN_ARTICLE_CHARS) -> ThinVerdict:
    """Judge whether ``text`` is a usable article body.

    Args:
        text: Extracted page text, or None.
        min_chars: Length below which the text is treated as a stub.

    Returns:
        A ThinVerdict; truthy when the text should not be treated as an article.

    """
    normalized = _normalize(text or "")
    count = len(normalized)

    if count == 0:
        return ThinVerdict(is_thin=True, reason="empty", char_count=0)

    if count < min_chars:
        return ThinVerdict(is_thin=True, reason="below_min_chars", char_count=count)

    if count <= MARKER_MAX_CHARS:
        marker = find_subscription_marker(normalized)
        if marker:
            return ThinVerdict(is_thin=True, reason=f"subscription_marker:{marker}", char_count=count)

    return ThinVerdict(is_thin=False, reason="ok", char_count=count)
