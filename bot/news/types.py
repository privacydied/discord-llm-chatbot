"""Types for licensed news-source resolution.
[CA][CMV][IV].

These types describe an article retrieved from a *publisher-sanctioned*
source -- an official content API or a publisher's own syndication feed --
used when generic web extraction returns a stub instead of an article.
"""

from __future__ import annotations

from dataclasses import dataclass

# Hard cap on body text injected into an LLM prompt, so a long feature
# article cannot blow the context budget. [CMV][PA]
DEFAULT_MAX_BODY_CHARS = 6000

# Ellipsis appended when a body is capped. [CMV]
TRUNCATION_SUFFIX = "…"


@dataclass(frozen=True)
class NewsArticle:
    """A normalized article body from a licensed provider."""

    url: str
    body: str
    title: str | None = None
    author: str | None = None
    published: str | None = None
    provider: str = "unknown"

    def capped_body(self, max_chars: int = DEFAULT_MAX_BODY_CHARS) -> str:
        """Return the body truncated to ``max_chars`` on a word boundary."""
        body = (self.body or "").strip()
        if len(body) <= max_chars:
            return body
        cut = body[:max_chars]
        # Prefer breaking at the last whitespace so we do not split a word.
        space = cut.rfind(" ")
        if space > max_chars // 2:
            cut = cut[:space]
        return cut + TRUNCATION_SUFFIX

    def to_message(self, max_chars: int = DEFAULT_MAX_BODY_CHARS) -> str:
        """Render the article for inclusion in an LLM prompt."""
        parts: list[str] = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.author:
            parts.append(f"Author: {self.author}")
        if self.published:
            parts.append(f"Published: {self.published}")
        parts.append(f"Source: {self.provider} ({self.url})")
        body = self.capped_body(max_chars)
        if body:
            parts.append(f"Text: {body}")
        return "\n".join(parts)
