"""Find the time expression in a reminder and turn it into a UTC instant.
[IV][REH][CMV].

``parse_reminder("to post a fit pic on may 2027", zone_name="Europe/London", now=...)``
returns the due instant (1 May 2027 09:00 London time, in UTC) and the task
("post a fit pic").

Two kinds of "when":

* **Floating** ("in 3 hours", "tomorrow", "next week") is the same instant for
  everyone, so it needs no timezone. "tomorrow" keeps the current time.
* **Wall clock** ("at 5pm", "friday", "may 2027") means that clock time where
  the person lives. Without a known zone the result is ``needs_timezone`` and
  the caller asks for one instead of guessing.

A date with no time fires at ``DEFAULT_HOUR`` local time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from . import grammar as g
from .extract import Pieces, add_months, apply_duration, collect
from .timezones import country_zones, load_zone

DEFAULT_HOUR = 9
MAX_HORIZON = timedelta(days=366 * 10)
# Small grace so "in 0 seconds" or clock skew is not "in the past". [CMV]
PAST_GRACE = timedelta(seconds=5)
# With a date, a bare "at 5" means 5pm, "at 9" means 9am. [CMV]
BARE_HOUR_PM_BELOW = 8
_NEXT_DAY_WORDS = ("tomorrow", "tomorow", "tommorow", "tommorrow", "tmrw", "tmr", "tmw")
_MDY_COUNTRIES = ("united states",)

_LEAD_WORDS = frozenset("to that about me on at in by for and please pls plz of".split())
_TRAIL_WORDS = frozenset("to on at in by for next this the and around from of until".split())
_EDGE_PUNCT = " \t\n,.;:-–—!"


class WhenError(ValueError):
    """A time was found but cannot be used (in the past, too far, impossible)."""


@dataclass(frozen=True)
class ParsedWhen:
    """Outcome. ``due_utc`` is None only when ``needs_timezone`` is True."""

    due_utc: datetime | None
    task: str
    when_text: str
    zone: str | None
    needs_timezone: bool = False


def date_order_for(zone_name: str | None) -> str:
    """'MDY' for US zones (12/25), else 'DMY' (25/12)."""
    if not zone_name:
        return "DMY"
    if zone_name.startswith("US/"):
        return "MDY"
    us_zones = {z for country in _MDY_COUNTRIES for z in country_zones().get(country, [])}
    return "MDY" if zone_name in us_zones else "DMY"


def clean_task(text: str, spans: list[g.Span]) -> str:
    """Cut the time spans out and trim connector words left at the edges."""
    for start, end in sorted(spans, reverse=True):
        text = f"{text[:start]} {text[end:]}"
    words = " ".join(text.split()).strip(_EDGE_PUNCT).split()
    while words and words[0].lower().strip(_EDGE_PUNCT) in _LEAD_WORDS:
        words.pop(0)
    while words and words[-1].lower().strip(_EDGE_PUNCT) in _TRAIL_WORDS:
        words.pop()
    return " ".join(words).strip(_EDGE_PUNCT)


def _day_word(p: Pieces) -> str:
    return re.sub(r"\s+", " ", p.date_groups.get("w", "")).lower() if p.date_kind == "day_word" else ""


def _is_floating(p: Pieces) -> bool:
    if p.hour is not None or p.daypart:
        return False
    if p.date_kind is None:
        return bool(p.relative)
    word = _day_word(p)
    return p.date_kind == "next_period" or word in _NEXT_DAY_WORDS or "day after" in word


def _period_shift(moment: datetime, p: Pieces) -> datetime:
    if p.date_kind == "next_period":
        period = p.date_groups["p"].lower()
        return moment + timedelta(days=7) if period == "week" else add_months(moment, 12 if period == "year" else 1)
    word = _day_word(p)
    if "day after" in word:
        return moment + timedelta(days=2)
    return moment + timedelta(days=1) if word in _NEXT_DAY_WORDS else moment


def _floating_due(p: Pieces, now: datetime) -> datetime:
    return _period_shift(apply_duration(now, p.relative or []), p)


def _safe_date(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise WhenError(f"{day}/{month}/{year} isn't a real date") from exc


def _numeric_date(groups: dict, base: date, order: str) -> tuple[date, str | None]:
    first, second = int(groups["a"]), int(groups["b"])
    day, month = (second, first) if order == "MDY" else (first, second)
    if month > 12 >= day:
        day, month = month, day
    if "c" in groups:
        year = int(groups["c"])
        return _safe_date(year + 2000 if year < 100 else year, month, day), None
    return _safe_date(base.year, month, day), "year"


def _month_date(p: Pieces, base: date) -> tuple[date, str | None]:
    groups = p.date_groups
    month = g.MONTHS[(groups.get("mon") or groups.get("mon2")).lower()]
    year = groups.get("y") or groups.get("y2")
    day = int(groups.get("d", 0))
    if p.date_kind == "month_only":
        if month == base.month and (not year or int(year) == base.year):
            return base, "day"
        day = 1
    if year:
        return _safe_date(int(year), month, day), None
    return _safe_date(base.year, month, day), "year"


def _day_of_month(day: int, today: date) -> date:
    """This month's ``day``, or the next month that has one ("the 31st" in November)."""
    for ahead in range(12):
        index = today.month - 1 + ahead
        try:
            return date(today.year + index // 12, index % 12 + 1, day)
        except ValueError:
            continue
    raise WhenError(f"no month has a {day}th")


def _weekday_date(p: Pieces, base: date) -> tuple[date, str | None]:
    if p.date_kind == "weekend":
        return base + timedelta(days=(5 - base.weekday()) % 7), "week"
    name = (p.date_groups.get("wd") or p.date_groups.get("wd2")).lower()
    return base + timedelta(days=(g.WEEKDAYS[name] - base.weekday()) % 7), "week"


def _resolve_date(p: Pieces, base: datetime, order: str) -> tuple[date | None, str | None]:
    """(local date, roll unit if that date turns out to be past)."""
    kind, groups, today = p.date_kind, p.date_groups, base.date()
    if kind is None:
        return None, None
    if kind == "iso":
        return _safe_date(int(groups["y"]), int(groups["mo"]), int(groups["d"])), None
    if kind in ("day_month", "month_day", "month_only"):
        return _month_date(p, today)
    if kind == "numeric":
        return _numeric_date(groups, today, order)
    if kind == "day_of_month":
        return _day_of_month(int(groups["d"]), today), "month"
    if kind == "year":
        return date(int(groups["y"]), 1, 1), None
    if kind in ("weekend", "weekday"):
        return _weekday_date(p, today)
    return _period_shift(base, p).date(), None


def _resolve_time(p: Pieces, has_date: bool) -> tuple[int, int] | None:
    word = _day_word(p)
    if p.hour is not None:
        hour = p.hour
        if p.bare_hour and hour < 12 and (word == "tonight" or (has_date and hour < BARE_HOUR_PM_BELOW)):
            hour += 12
        return hour, p.minute
    if p.daypart:
        return g.DAYPARTS[p.daypart]
    if word == "tonight":
        return g.TONIGHT_DEFAULT
    if word == "today":
        raise WhenError("what time today?")
    return None


def _next_clock(base: datetime, hm: tuple[int, int], bare: bool, *, fixed_day: bool) -> datetime:
    """The first moment after ``base`` showing hm (or hm+12h for a bare hour)."""
    hours = [hm[0]] + ([hm[0] + 12] if bare and hm[0] < 12 else [])
    days = [base.date()] if fixed_day else [base.date(), base.date() + timedelta(days=1)]
    options = [datetime.combine(d, time(h, hm[1])) for d in days for h in hours]
    future = [o for o in options if o > base]
    return min(future) if future else min(options)


def _roll(moment: datetime, unit: str | None) -> datetime:
    if unit == "day":
        return moment + timedelta(days=1)
    if unit == "week":
        return moment + timedelta(days=7)
    if unit in ("month", "year"):
        return add_months(moment, 1 if unit == "month" else 12)
    return moment


def _wallclock(p: Pieces, local_now: datetime, order: str) -> datetime:
    """Naive local datetime for a wall-clock expression."""
    base = apply_duration(local_now.replace(tzinfo=None), p.relative or [])
    day, roll = _resolve_date(p, base, order)
    hm = _resolve_time(p, day is not None)
    if day is None:
        return _next_clock(base, hm or (DEFAULT_HOUR, 0), p.bare_hour, fixed_day=bool(p.relative))
    moment = datetime.combine(day, time(*(hm or (DEFAULT_HOUR, 0))))
    return _roll(moment, roll) if moment <= base else moment


def _checked(due: datetime, now: datetime) -> datetime:
    if due < now - PAST_GRACE:
        raise WhenError("that time is already in the past")
    if due - now > MAX_HORIZON:
        raise WhenError("that's more than 10 years away")
    return due


def parse_reminder(text: str, *, zone_name: str | None, now: datetime | None = None) -> ParsedWhen | None:
    """Parse a reminder. None = no time expression. Raises WhenError."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    pieces = collect(text or "")
    if not pieces.spans:
        return None
    task = clean_task(text, pieces.spans)
    when_text = " ".join(text[s:e].strip() for s, e in sorted(pieces.spans))
    if _is_floating(pieces):
        return ParsedWhen(_checked(_floating_due(pieces, now), now), task, when_text, None)
    zone = pieces.zone or zone_name
    if not zone:
        # Surface impossible dates now, not after asking for a zone. The day of
        # slack covers every zone's offset from UTC (at most 14h).
        _checked(_wallclock(pieces, now, "DMY").replace(tzinfo=UTC) + timedelta(days=1), now)
        return ParsedWhen(None, task, when_text, None, needs_timezone=True)
    tz = load_zone(zone)
    local = _wallclock(pieces, now.astimezone(tz), date_order_for(zone))
    return ParsedWhen(_checked(local.replace(tzinfo=tz).astimezone(UTC), now), task, when_text, zone)
