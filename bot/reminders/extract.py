"""Collect the pieces of a reminder's time expression from free text.
[IV][CMV].

Finds at most one relative duration, one date, one time of day and an
optional zone suffix ("5pm EST"), each at a non-overlapping span. Turning the
pieces into an instant is ``bot.reminders.parsing``'s job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import grammar as g
from .timezones import ALIASES, AmbiguousTimezoneError, resolve_timezone


@dataclass
class Pieces:
    spans: list[g.Span] = field(default_factory=list)
    relative: list[tuple[float, str]] | None = None
    date_kind: str | None = None
    date_groups: dict = field(default_factory=dict)
    hour: int | None = None
    minute: int = 0
    bare_hour: bool = False
    daypart: str | None = None
    zone: str | None = None

    def free(self, span: g.Span) -> bool:
        return all(span[1] <= s or span[0] >= e for s, e in self.spans)


_DATE_FINDERS = (
    ("iso", g.ISO_RE),
    ("day_month", g.DAY_MONTH_RE),
    ("month_day", g.MONTH_DAY_RE),
    ("month_only", g.MONTH_ONLY_RE),
    ("numeric", g.NUMERIC_DATE_RE),
    ("day_of_month", g.DAY_OF_MONTH_RE),
    ("year", g.YEAR_ONLY_RE),
    ("weekend", g.WEEKEND_RE),
    ("next_period", g.NEXT_PERIOD_RE),
    ("day_word", g.DAY_WORD_RE),
    ("weekday", g.WEEKDAY_RE),
)
_TIME_FINDERS = (("ampm", g.TIME_AMPM_RE), ("h24", g.TIME_24_RE), ("oclock", g.TIME_OCLOCK_RE), ("bare", g.TIME_BARE_AT_RE))


def _first_free(regex, text: str, pieces: Pieces):
    return next((m for m in regex.finditer(text) if pieces.free(m.span())), None)


def _find_relative(text: str, pieces: Pieces) -> None:
    match = _first_free(g.RELATIVE_RE, text, pieces)
    if match:
        pieces.relative = g.duration_pieces(match.group("a") or match.group("b"))
        pieces.spans.append(match.span())


def _find_date(text: str, pieces: Pieces) -> None:
    for kind, regex in _DATE_FINDERS:
        match = _first_free(regex, text, pieces)
        if match:
            pieces.date_kind, pieces.date_groups = kind, {k: v for k, v in match.groupdict().items() if v}
            pieces.spans.append(match.span())
            if kind == "iso" and match.group("h"):
                pieces.hour, pieces.minute = int(match.group("h")), int(match.group("mi"))
                _find_zone_suffix(text, match.end(), pieces)
            return


def _time_value(kind: str, match) -> tuple[int, int, bool] | None:
    hour, minute = int(match.group("h")), int(match.groupdict().get("m") or 0)
    if kind == "ampm":
        if not 1 <= hour <= 12:
            return None
        pm = match.group("ap").lower().startswith("p")
        return (hour % 12) + (12 if pm else 0), minute, False
    if kind == "h24":
        return hour, minute, False
    return (hour, minute, 1 <= hour <= 12) if hour <= 23 else None


def _find_time(text: str, pieces: Pieces) -> None:
    if pieces.hour is not None:
        return
    for kind, regex in _TIME_FINDERS:
        match = _first_free(regex, text, pieces)
        value = _time_value(kind, match) if match else None
        if value:
            pieces.hour, pieces.minute, pieces.bare_hour = value
            pieces.spans.append(match.span())
            _find_zone_suffix(text, match.end(), pieces)
            return
    match = _first_free(g.DAYPART_RE, text, pieces)
    if match:
        pieces.daypart = match.group("w").lower()
        pieces.spans.append(match.span())


def _suffix_zone(match) -> str | None:
    candidate = match.group("off") or match.group("abbr") or match.group("city")
    if match.group("abbr") and candidate.lower() not in ALIASES:
        return None
    try:
        found = resolve_timezone(candidate)
    except AmbiguousTimezoneError:
        return None
    return found.name if found else None


def _find_zone_suffix(text: str, pos: int, pieces: Pieces) -> None:
    match = g.TZ_SUFFIX_RE.match(text, pos)
    zone = _suffix_zone(match) if match else None
    if zone:
        pieces.zone = zone
        pieces.spans.append((pos, match.end()))


def collect(text: str) -> Pieces:
    pieces = Pieces()
    _find_relative(text, pieces)
    _find_date(text, pieces)
    _find_time(text, pieces)
    return pieces


def add_months(moment: datetime, months: int) -> datetime:
    """Calendar month arithmetic, clamping the day (31 Jan + 1 month = 28/29 Feb)."""
    index = moment.month - 1 + months
    year, month = moment.year + index // 12, index % 12 + 1
    for day in range(moment.day, 27, -1):
        try:
            return moment.replace(year=year, month=month, day=day)
        except ValueError:
            continue
    return moment.replace(year=year, month=month, day=min(moment.day, 28))


def apply_duration(moment: datetime, parts: list[tuple[float, str]]) -> datetime:
    for amount, unit in parts:
        if unit in ("months", "years"):
            months = amount * (12 if unit == "years" else 1)
            moment = add_months(moment, int(months)) + timedelta(days=round((months % 1) * 30))
        else:
            moment += timedelta(**{unit: amount})
    return moment
