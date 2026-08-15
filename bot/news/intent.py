"""Detect "what's happening in the news" questions in free text.
[CA][IV][CMV].

Deliberately conservative. A false negative costs one ordinary LLM answer; a
false positive hijacks an unrelated question ("what's going on with my
deploy?") and answers it with world headlines. Every trigger therefore needs
an explicit news signal, not merely a "what's happening" phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Look-back windows in days. [CMV]
DAYS_TODAY = 1
DAYS_WEEK = 7
DAYS_TOPIC_DEFAULT = 7
DAYS_GENERAL_DEFAULT = 2

# Longest topic we will lift out of a sentence, in words. [CMV]
MAX_TOPIC_WORDS = 6

# Unambiguous news nouns -- these alone are enough to trigger.
_NEWS_NOUNS = r"(?:the\s+)?(?:news|headlines?|current\s+events|world\s+events)"

# "What's happening" style openers, which trigger only alongside a news noun,
# a time-of-interest word, or a world/global scope word.
_HAPPENING = r"(?:what(?:'s|s| is| are)?\s+)?(?:happening|going\s+on|new|the\s+latest|up)"

_SCOPE = r"(?:in\s+the\s+world|around\s+the\s+world|globally|in\s+the\s+news|worldwide)"

_TIME_TODAY = r"(?:today|right\s+now|currently|at\s+the\s+moment|this\s+morning|tonight)"
_TIME_WEEK = r"(?:this\s+week|past\s+week|last\s+week|recently|lately|past\s+few\s+days)"

# Direct news requests: "what's in the news", "any news", "news today",
# "catch me up on the news", "give me the headlines".
_DIRECT_NEWS_RE = re.compile(
    rf"\b(?:any|some|the)?\s*{_NEWS_NOUNS}\b|"
    rf"\bcatch\s+me\s+up\b|"
    rf"\bwhat(?:'s|s| is)?\s+in\s+{_NEWS_NOUNS}\b",
    re.IGNORECASE,
)

# "What's happening in the world today" -- opener plus scope or time.
_HAPPENING_RE = re.compile(
    rf"\b{_HAPPENING}\b.{{0,40}}?\b(?:{_SCOPE}|{_TIME_TODAY}|{_TIME_WEEK})\b|"
    rf"\b(?:{_SCOPE})\b.{{0,40}}?\b{_HAPPENING}\b",
    re.IGNORECASE,
)

_WEEK_RE = re.compile(rf"\b{_TIME_WEEK}\b", re.IGNORECASE)
_TODAY_RE = re.compile(rf"\b{_TIME_TODAY}\b", re.IGNORECASE)

# Topic capture: "news about X", "what's happening with X",
# "any news on X", "headlines about X".
# The lookbehind stops the "on" of "going on" from opening a topic, which
# would otherwise turn "what's going on in the news" into topic="in the news".
_TOPIC_RE = re.compile(
    r"\b(?<!going )(?:about|on|regarding|with|re)\s+(?P<topic>.+?)\s*[?.!]*$",
    re.IGNORECASE,
)

# Leading prepositions/articles stripped before deciding whether what remains
# is a real subject ("in the world" -> "world" -> not a topic).
_LEADING_FILLER_RE = re.compile(r"^(?:in|at|for|on|about|the|a|an)\s+", re.IGNORECASE)

# Words that are scope/time rather than a subject, so they never become topics.
_NON_TOPICS = {
    "the world",
    "world",
    "the news",
    "news",
    "today",
    "right now",
    "now",
    "currently",
    "this week",
    "the moment",
    "at the moment",
    "lately",
    "recently",
    "everything",
    "anything",
    "it",
    "that",
    "things",
    "stuff",
}

# Contexts that look like news phrasing but are about the user's own work --
# never hijack these. [IV]
_PERSONAL_CONTEXT_RE = re.compile(
    r"\b(?:my|our|your|this)\s+(?:code|build|deploy|deployment|server|bot|test|tests|"
    r"pr|branch|repo|project|ticket|job|script|container|pipeline|database|db)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NewsQuery:
    """A resolved request for a news digest."""

    topic: str | None
    days: int
    raw: str

    @property
    def is_general(self) -> bool:
        return self.topic is None


def _strip_leading_filler(topic: str) -> str:
    """Remove stacked leading prepositions/articles ("in the world" -> "world")."""
    previous = None
    while previous != topic:
        previous = topic
        topic = _LEADING_FILLER_RE.sub("", topic).strip()
    return topic


def _extract_topic(text: str) -> str | None:
    """Pull a subject out of the sentence, or None for a general digest."""
    match = _TOPIC_RE.search(text)
    if not match:
        return None
    topic = re.sub(r"\s+", " ", match.group("topic").strip().strip("\"'"))

    # Drop a trailing time phrase: "AI today" -> "AI".
    topic = re.sub(rf"\s*\b(?:{_TIME_TODAY}|{_TIME_WEEK})\b\s*$", "", topic, flags=re.IGNORECASE).strip()
    topic = _strip_leading_filler(topic)

    if not topic or topic.lower() in _NON_TOPICS:
        return None
    if len(topic.split()) > MAX_TOPIC_WORDS:
        return None
    return topic


def _resolve_days(text: str, topic: str | None) -> int:
    if _WEEK_RE.search(text):
        return DAYS_WEEK
    if _TODAY_RE.search(text):
        return DAYS_TODAY
    return DAYS_TOPIC_DEFAULT if topic else DAYS_GENERAL_DEFAULT


def detect_news_intent(text: str | None) -> NewsQuery | None:
    """Return a NewsQuery when ``text`` asks about the news, else None.

    Args:
        text: The user's message, with any bot mention already stripped.

    """
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned or len(cleaned) > 300:
        return None

    # Never hijack a question about the user's own systems.
    if _PERSONAL_CONTEXT_RE.search(cleaned):
        return None

    if not (_DIRECT_NEWS_RE.search(cleaned) or _HAPPENING_RE.search(cleaned)):
        return None

    topic = _extract_topic(cleaned)
    return NewsQuery(topic=topic, days=_resolve_days(cleaned, topic), raw=cleaned)
