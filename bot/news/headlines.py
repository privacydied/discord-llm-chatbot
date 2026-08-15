"""Current-headlines retrieval for news digests.
[CA][REH][CMV][PA][SFT].

Backs "what's happening in the news today" style questions. Uses the Guardian
Open Platform search endpoint, which returns licensed headlines, trail text and
body copy -- so the model summarises material the publisher syndicates for this
purpose rather than anything scraped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from bot.http_client import RequestConfig, get_http_client
from bot.utils.logging import get_logger

from .providers.guardian import API_ROOT

logger = get_logger(__name__)

# Guardian caps page-size at 50; stay well under it to keep prompts small. [CMV]
DEFAULT_LIMIT = 8
MAX_LIMIT = 20

# Per-headline summary budget inside the digest prompt. [CMV][PA]
SUMMARY_CHARS = 400

# Sections that read as "serious news" for an unfiltered world digest, so a
# general question does not come back full of recipes and football. [CMV]
WORLD_SECTIONS = "world|us-news|uk-news|politics|business|environment|technology|science"

_REQUEST = RequestConfig(connect_timeout=2.0, read_timeout=8.0, total_timeout=10.0, max_retries=2)


@dataclass(frozen=True)
class Headline:
    """One story in a digest."""

    title: str
    url: str
    section: str | None = None
    published: str | None = None
    summary: str = ""

    def to_line(self) -> str:
        """Render as a single prompt-friendly bullet."""
        bits = [f"- {self.title}"]
        if self.section:
            bits.append(f"[{self.section}]")
        if self.published:
            bits.append(f"({self.published[:10]})")
        head = " ".join(bits)
        body = self.summary.strip()
        return f"{head}\n  {body}\n  {self.url}" if body else f"{head}\n  {self.url}"


def _from_date(days: int) -> str:
    """ISO date ``days`` before now, for the Guardian from-date filter."""
    span = max(1, min(days, 31))
    return (datetime.now(UTC) - timedelta(days=span)).strftime("%Y-%m-%d")


def _quote_topic(topic: str) -> str:
    """Quote a multi-word topic so the API matches the phrase, not any word.

    Without this, ``artificial intelligence`` matches articles containing
    merely "intelligence", which -- ordered by date -- returns whatever was
    published most recently rather than anything on the subject.
    """
    cleaned = " ".join(topic.split())
    if '"' in cleaned or len(cleaned.split()) < 2:
        return cleaned
    return f'"{cleaned}"'


def _build_params(api_key: str, topic: str | None, days: int, limit: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "api-key": api_key,
        "from-date": _from_date(days),
        "page-size": max(1, min(limit, MAX_LIMIT)),
        "show-fields": "headline,trailText,byline,bodyText",
        "format": "json",
    }
    if topic:
        # Relevance, not recency: from-date already bounds freshness, and
        # ordering a topic search by date surfaces unrelated same-day stories.
        params["q"] = _quote_topic(topic)
        params["order-by"] = "relevance"
    else:
        # No topic: newest first, restricted to hard-news sections rather
        # than the whole wire.
        params["order-by"] = "newest"
        params["section"] = WORLD_SECTIONS
    return params


def _to_headline(result: dict[str, Any]) -> Headline | None:
    fields = result.get("fields") or {}
    title = (fields.get("headline") or result.get("webTitle") or "").strip()
    url = (result.get("webUrl") or "").strip()
    if not title or not url:
        return None
    summary = (fields.get("trailText") or "").strip()
    if not summary:
        summary = (fields.get("bodyText") or "").strip()[:SUMMARY_CHARS]
    return Headline(
        title=title,
        url=url,
        section=result.get("sectionName"),
        published=result.get("webPublicationDate"),
        summary=summary[:SUMMARY_CHARS],
    )


async def fetch_headlines(
    topic: str | None = None,
    *,
    cfg: dict | None = None,
    days: int = 1,
    limit: int = DEFAULT_LIMIT,
) -> list[Headline]:
    """Return recent headlines, newest first. Returns [] on any failure. [REH]

    Args:
        topic: Free-text subject filter, or None for a general world digest.
        cfg: Config mapping supplying GUARDIAN_API_KEY.
        days: How far back to look.
        limit: Maximum stories.

    """
    from bot.config import load_config

    config = cfg if cfg is not None else load_config()
    api_key = (config.get("GUARDIAN_API_KEY") or "").strip()
    if not api_key:
        logger.info("news.headlines.no_key topic=%s", topic)
        return []

    try:
        client = await get_http_client()
        response = await client.get(
            f"{API_ROOT}/search",
            config=_REQUEST,
            params=_build_params(api_key, topic, days, limit),
        )
    except Exception as exc:  # [REH]
        logger.warning("news.headlines.failed topic=%s error=%s", topic, exc)
        return []

    if response.status_code != 200:
        logger.warning("news.headlines.http_status status=%s topic=%s", response.status_code, topic)
        return []

    return _parse_results(response.json(), topic)


def _parse_results(payload: dict[str, Any], topic: str | None) -> list[Headline]:
    body = (payload or {}).get("response") or {}
    if body.get("status") != "ok":
        logger.warning("news.headlines.api_error topic=%s status=%s", topic, body.get("status"))
        return []
    headlines = [h for h in (_to_headline(r) for r in body.get("results") or []) if h]
    logger.info("news.headlines.ok topic=%s count=%d", topic, len(headlines))
    return headlines


def render_digest(headlines: list[Headline], topic: str | None, days: int) -> str:
    """Render headlines as grounding context for the model."""
    if not headlines:
        return ""
    window = "the last 24 hours" if days <= 1 else f"the last {days} days"
    subject = f' on "{topic}"' if topic else ""
    order = "most relevant first" if topic else "newest first"
    header = f"Current news headlines{subject} from the Guardian, covering {window}, {order}:"
    lines = "\n".join(h.to_line() for h in headlines)
    return f"{header}\n\n{lines}"
