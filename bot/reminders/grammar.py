"""Regex grammar for the "when" part of a reminder.
[IV][CMV].

Hand-written on purpose. ``dateparser.search_dates`` was tried first and gave
silently wrong answers on everyday input ("tomorrow 9am" kept the current
time; "saturday at 10am" became a date eleven months out). A reminder that
fires at the wrong time is worse than one that asks again, so this grammar
only accepts forms it can read exactly, and everything else is "no time found".

Each finder returns ``(span, value)`` where ``span`` is the ``(start, end)``
of the matched text, so the caller can cut it out to leave the task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

Span = tuple[int, int]

NUMBER_WORDS: dict[str, float] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "a couple of": 2,
    "a couple": 2,
    "couple of": 2,
    "couple": 2,
    "a few": 3,
    "few": 3,
    "half an": 0.5,
    "half a": 0.5,
}

# unit spelling -> canonical unit
UNITS: dict[str, str] = {
    **dict.fromkeys(["s", "sec", "secs", "second", "seconds"], "seconds"),
    **dict.fromkeys(["m", "min", "mins", "minute", "minutes"], "minutes"),
    **dict.fromkeys(["h", "hr", "hrs", "hour", "hours"], "hours"),
    **dict.fromkeys(["d", "day", "days"], "days"),
    **dict.fromkeys(["w", "wk", "wks", "week", "weeks"], "weeks"),
    **dict.fromkeys(["mo", "mos", "month", "months"], "months"),
    **dict.fromkeys(["y", "yr", "yrs", "year", "years"], "years"),
}

MONTHS: dict[str, int] = {
    **dict.fromkeys(["january", "jan"], 1),
    **dict.fromkeys(["february", "feb"], 2),
    **dict.fromkeys(["march", "mar"], 3),
    **dict.fromkeys(["april", "apr"], 4),
    "may": 5,
    **dict.fromkeys(["june", "jun"], 6),
    **dict.fromkeys(["july", "jul"], 7),
    **dict.fromkeys(["august", "aug"], 8),
    **dict.fromkeys(["september", "sept", "sep"], 9),
    **dict.fromkeys(["october", "oct"], 10),
    **dict.fromkeys(["november", "nov"], 11),
    **dict.fromkeys(["december", "dec"], 12),
}

WEEKDAYS: dict[str, int] = {
    **dict.fromkeys(["monday", "mon"], 0),
    **dict.fromkeys(["tuesday", "tue", "tues"], 1),
    **dict.fromkeys(["wednesday", "wed"], 2),
    **dict.fromkeys(["thursday", "thu", "thur", "thurs"], 3),
    **dict.fromkeys(["friday", "fri"], 4),
    **dict.fromkeys(["saturday", "sat"], 5),
    **dict.fromkeys(["sunday", "sun"], 6),
}

# Named parts of the day -> (hour, minute). [CMV]
DAYPARTS: dict[str, tuple[int, int]] = {
    "morning": (9, 0),
    "afternoon": (14, 0),
    "evening": (18, 0),
    "night": (21, 0),
    "noon": (12, 0),
    "midday": (12, 0),
    "midnight": (0, 0),
}
TONIGHT_DEFAULT = (20, 0)


def _alt(words) -> str:
    return "|".join(sorted((re.escape(w).replace(r"\ ", r"\s+") for w in words), key=len, reverse=True))


_NUM = rf"(?:\d+(?:\.\d+)?|(?<![a-z])(?:{_alt(NUMBER_WORDS)})\b)"
_UNIT = _alt(UNITS)
_PIECE = rf"{_NUM}\s*(?:{_UNIT})(?![a-z])"
_DURATION = rf"{_PIECE}(?:\s*(?:,|and)?\s*{_PIECE})*"
_PIECE_RE = re.compile(rf"(?P<n>{_NUM})\s*(?P<u>{_UNIT})(?![a-z])", re.IGNORECASE)
RELATIVE_RE = re.compile(rf"\b(?:(?:in|after|within)\s+(?P<a>{_DURATION})|(?P<b>{_DURATION})\s+(?:from\s+now|later))", re.IGNORECASE)

_MONTH = _alt(MONTHS)
_FULL_MONTH = _alt([m for m in MONTHS if len(m) > 3 or m == "may"])
_ORD = r"(?:st|nd|rd|th)?"
_YEAR = r"(?P<y>\d{4})"
# "5th may 2027", "the 6th of may"
DAY_MONTH_RE = re.compile(rf"\b(?:on\s+)?(?:the\s+)?(?P<d>\d{{1,2}}){_ORD}(?:\s+of)?\s+(?P<mon>{_MONTH})\b(?:,?\s+{_YEAR}\b)?", re.IGNORECASE)
# "may 5th, 2027", "may 5"
MONTH_DAY_RE = re.compile(rf"\b(?:on\s+)?(?P<mon>{_MONTH})\s+(?P<d>\d{{1,2}}){_ORD}\b(?!\s*(?::|am|pm|a\.m|p\.m))(?:,?\s+{_YEAR}\b)?", re.IGNORECASE)
# "may 2027", "in may", "on may 2027", "early june" (a bare "may" is a verb: needs in/on/year)
MONTH_ONLY_RE = re.compile(rf"\b(?:(?:in|on|by|during|early|start\s+of|beginning\s+of)\s+(?P<mon>{_FULL_MONTH})\b(?:\s+{_YEAR}\b)?|(?P<mon2>{_FULL_MONTH})\s+(?P<y2>\d{{4}})\b)", re.IGNORECASE)
ISO_RE = re.compile(r"\b(?P<y>\d{4})-(?P<mo>\d{1,2})-(?P<d>\d{1,2})(?:[T\s](?P<h>\d{1,2}):(?P<mi>\d{2})(?::\d{2}(?:\.\d+)?)?)?\b", re.IGNORECASE)
NUMERIC_DATE_RE = re.compile(r"\b(?:on\s+)?(?P<a>\d{1,2})[/.](?P<b>\d{1,2})(?:[/.](?P<c>\d{2}|\d{4}))?\b(?![/.:]\d)")
DAY_OF_MONTH_RE = re.compile(r"\b(?:on\s+)?the\s+(?P<d>\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)
YEAR_ONLY_RE = re.compile(r"\b(?:in|by)\s+(?P<y>20\d{2})\b", re.IGNORECASE)

_WD_FULL = _alt([w for w in WEEKDAYS if len(w) >= 4])
_WD_ANY = _alt(WEEKDAYS)
# Three-letter forms ("sat", "sun") need a lead word: "sat down", "the sun" are not dates.
WEEKDAY_RE = re.compile(rf"\b(?:(?P<mod>next|this|coming|on|this\s+coming)\s+(?P<wd>{_WD_ANY})|(?P<wd2>{_WD_FULL}))\b", re.IGNORECASE)
WEEKEND_RE = re.compile(r"\b(?:on\s+)?(?:the\s+|this\s+|next\s+)?weekend\b", re.IGNORECASE)
DAY_WORD_RE = re.compile(
    r"\b(?P<w>(?:the\s+)?day\s+after\s+(?:tomorrow|tmrw|tmr)|today|tonight|tomorrow|tomorow|tommorow|tommorrow|tmrw|tmr|tmw)\b",
    re.IGNORECASE,
)
NEXT_PERIOD_RE = re.compile(r"\bnext\s+(?P<p>week|month|year)\b", re.IGNORECASE)

_AMPM = r"(?P<ap>a\.?m\.?|p\.?m\.?)(?![a-z])"
TIME_AMPM_RE = re.compile(rf"\b(?:(?:at|by|around|@)\s*)?(?P<h>\d{{1,2}})(?:[:.](?P<m>\d{{2}}))?\s*{_AMPM}", re.IGNORECASE)
TIME_24_RE = re.compile(r"(?<![\d/.-])\b(?:(?:at|by|around|@)\s*)?(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\b(?![/.:]\d)", re.IGNORECASE)
TIME_OCLOCK_RE = re.compile(r"\b(?:(?:at|by|around)\s+)?(?P<h>\d{1,2})\s*o'?\s*clock\b", re.IGNORECASE)
TIME_BARE_AT_RE = re.compile(rf"\b(?:at|@)\s*(?P<h>\d{{1,2}})\b(?!\s*(?:[:/.]\d|%|(?:{_UNIT})(?![a-z])|(?:st|nd|rd|th)\b))", re.IGNORECASE)
DAYPART_RE = re.compile(rf"\b(?:(?:in\s+the|this|at)\s+)?(?P<w>{_alt(DAYPARTS)})\b", re.IGNORECASE)

# Optional zone right after a time: "5pm EST", "17:00 uk time", "9am utc+2".
# Used with .match(text, pos), so it is anchored at the end of the time.
TZ_SUFFIX_RE = re.compile(
    r"\s*(?:(?P<off>(?:utc|gmt)\s*[+-]\s*\d{1,2}(?::?\d{2})?)|(?P<abbr>[a-z]{2,5})\b(?:\s+time\b)?|(?P<city>[a-z]+(?:\s+[a-z]+)?)\s+time\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Hit:
    """One recognised piece of the time expression."""

    span: Span
    kind: str
    value: object


def number_value(word: str) -> float:
    word = " ".join(word.lower().split())
    return float(word) if word[0].isdigit() else NUMBER_WORDS[word]


def duration_pieces(text: str) -> list[tuple[float, str]]:
    """'1h 30 mins' -> [(1.0, 'hours'), (30.0, 'minutes')]."""
    return [(number_value(m.group("n")), UNITS[m.group("u").lower()]) for m in _PIECE_RE.finditer(text)]
