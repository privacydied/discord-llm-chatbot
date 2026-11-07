"""Detect reminder and timezone requests written in plain words.
[IV][CMV].

"@bot remind me to post a fit pic on may 2027", "set a reminder for friday
5pm: pay rent", "yo ping me in 2 hours to check the oven". Only messages that
*open* with the request count, so "can you remind me what we said" or a
story about reminders stays ordinary chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LEAD = (
    r"(?:(?:yo|ayo|ay|aye|hey|oi|bro|bruh|ronin|pls|plz|please|ok|okay|and|also|just|"
    r"can\s+(?:you|u)|could\s+(?:you|u)|will\s+(?:you|u)|would\s+(?:you|u)|"
    r"i\s+(?:need|want)\s+(?:you|u)\s+to|go\s+ahead\s+and)[\s,]+)*"
)
_EXPLICIT = r"set\s+(?:me\s+)?(?:up\s+)?(?:a\s+)?reminder(?:\s+for\s+me)?|(?:make|add|create)\s+(?:me\s+)?(?:a\s+)?reminder|remindme|reminder\s*[:\-]"
_SOFT = r"remind\s+me|don'?t\s+let\s+me\s+forget|ping\s+me"
REMINDER_RE = re.compile(rf"^\s*{_LEAD}(?:(?P<explicit>{_EXPLICIT})|(?P<soft>{_SOFT})\b)[\s,:\-]*(?P<rest>.*)$", re.IGNORECASE | re.DOTALL)

# "remind me what you said" is a question, not a reminder.
_QUESTION_RE = re.compile(r"^(?:who|what|whats|what's|how|why|where|which|whether|if|when\s+(?:we|you|u|i|he|she|they|it)|of\s+(?:what|who|how|the\s+time))\b", re.IGNORECASE)

TIMEZONE_STATEMENT_RE = re.compile(
    rf"^\s*{_LEAD}(?:(?:set|change|update|make)\s+my\s+(?:time\s*zone|tz)\s+(?:to|as)|my\s+(?:time\s*zone|tz)(?:\s+is|'s|\s*[:=])|"
    r"(?:time\s*zone|tz)\s*[:=]|i'?m\s+(?:on|in)\s+(?=.*\btime\b))\s*(?P<zone>.+?)\s*(?:time(?:\s*zone)?)?\s*[.!]*$",
    re.IGNORECASE | re.DOTALL,
)
# Only while a reminder is waiting on a timezone: "i'm in london", "i live in texas".
WHERE_I_LIVE_RE = re.compile(r"^\s*(?:i'?m\s+(?:in|from)|i\s+live\s+in|i\s+am\s+in|in|it'?s)\s+(?P<zone>.+?)\s*[.!]*$", re.IGNORECASE | re.DOTALL)
MAX_BARE_ZONE_WORDS = 4


@dataclass(frozen=True)
class ReminderRequest:
    """``rest`` is the text after the trigger phrase; ``explicit`` = surely a reminder."""

    rest: str
    explicit: bool


def parse_reminder_request(text: str | None) -> ReminderRequest | None:
    match = REMINDER_RE.match(text or "")
    if not match:
        return None
    rest = match.group("rest").strip()
    if match.group("soft") and _QUESTION_RE.match(rest):
        return None
    explicit = bool(match.group("explicit")) or rest.lower().startswith(("to ", "that "))
    return ReminderRequest(rest=rest, explicit=explicit)


@dataclass(frozen=True)
class ZoneStatement:
    """``explicit`` = "my timezone is X" (answer even if X is unknown); else a
    guess from a bare reply, to be ignored silently when it isn't a zone."""

    zone_text: str
    explicit: bool


def parse_timezone_statement(text: str | None, *, awaiting_zone: bool) -> ZoneStatement | None:
    """The zone in "my timezone is X". While a reminder awaits a zone, a bare
    "london" or "i'm in texas" counts too."""
    body = (text or "").strip()
    match = TIMEZONE_STATEMENT_RE.match(body)
    if match:
        return ZoneStatement(match.group("zone").strip(), explicit=True)
    if not awaiting_zone:
        return None
    match = WHERE_I_LIVE_RE.match(body)
    if match:
        return ZoneStatement(match.group("zone").strip(), explicit=False)
    return ZoneStatement(body, explicit=False) if 0 < len(body.split()) <= MAX_BARE_ZONE_WORDS else None
