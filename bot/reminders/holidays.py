"""Named holidays for reminder times ("on black friday", "christmas eve").
[IV][CMV].

Without this table, "black friday" was read as the weekday "friday" and the
word "black" was left in the task. Each holiday maps to a function of
``(year, order)`` that returns the local date; ``order`` is the date order
('MDY' for US users, else 'DMY') and picks the US or UK date where the two
countries differ (Mother's Day).

``HOLIDAY_PATTERNS`` is (regex fragment, key). ``bot.reminders.grammar`` joins
them into one regex with a named group per key. Apostrophes are optional so
"new years eve" also works.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

_THU, _MON, _SUN = 3, 0, 6


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def thanksgiving(year: int) -> date:
    """US Thanksgiving: fourth Thursday of November."""
    return _nth_weekday(year, 11, _THU, 4)


def _after_thanksgiving(days: int) -> Callable[[int, str], date]:
    return lambda year, _order: thanksgiving(year) + timedelta(days=days)


def _after_easter(days: int) -> Callable[[int, str], date]:
    return lambda year, _order: easter(year) + timedelta(days=days)


def _fixed(month: int, day: int) -> Callable[[int, str], date]:
    return lambda year, _order: date(year, month, day)


def _mothers_day(year: int, order: str) -> date:
    """US: second Sunday of May. UK: Mothering Sunday, three weeks before Easter."""
    return _nth_weekday(year, 5, _SUN, 2) if order == "MDY" else easter(year) - timedelta(days=21)


HOLIDAYS: dict[str, Callable[[int, str], date]] = {
    "black_friday": _after_thanksgiving(1),
    "small_business_saturday": _after_thanksgiving(2),
    "cyber_monday": _after_thanksgiving(4),
    "giving_tuesday": _after_thanksgiving(5),
    "thanksgiving": _after_thanksgiving(0),
    "christmas_eve": _fixed(12, 24),
    "christmas": _fixed(12, 25),
    "boxing_day": _fixed(12, 26),
    "new_years_eve": _fixed(12, 31),
    "new_years_day": _fixed(1, 1),
    "valentines": _fixed(2, 14),
    "st_patricks": _fixed(3, 17),
    "halloween": _fixed(10, 31),
    "bonfire_night": _fixed(11, 5),
    "independence_day": _fixed(7, 4),
    "good_friday": _after_easter(-2),
    "easter_monday": _after_easter(1),
    "easter": _after_easter(0),
    "mothers_day": _mothers_day,
    "fathers_day": lambda year, _order: _nth_weekday(year, 6, _SUN, 3),
    "memorial_day": lambda year, _order: _last_weekday(year, 5, _MON),
    "labor_day": lambda year, _order: _nth_weekday(year, 9, _MON, 1),
}

_APOS = r"(?:'|’)?"
_DAY = r"(?:\s+day)?"
# Longer names first so "christmas eve" wins over "christmas".
HOLIDAY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"black\s+friday", "black_friday"),
    (r"small\s+business\s+saturday", "small_business_saturday"),
    (r"cyber\s+monday", "cyber_monday"),
    (r"giving\s+tuesday", "giving_tuesday"),
    (rf"thanksgiving{_DAY}", "thanksgiving"),
    (r"(?:christmas|xmas)\s+eve", "christmas_eve"),
    (rf"(?:christmas|xmas){_DAY}", "christmas"),
    (r"boxing\s+day", "boxing_day"),
    (rf"new\s+year{_APOS}s?{_APOS}\s+eve|nye", "new_years_eve"),
    (rf"new\s+year{_APOS}s?{_APOS}\s+day", "new_years_day"),
    (rf"valentine{_APOS}s?{_APOS}{_DAY}", "valentines"),
    (rf"st\.?\s+(?:patrick|paddy){_APOS}s?{_APOS}{_DAY}|saint\s+patrick{_APOS}s?{_APOS}{_DAY}", "st_patricks"),
    (r"halloween", "halloween"),
    (r"bonfire\s+night|guy\s+fawkes(?:\s+night)?", "bonfire_night"),
    (r"independence\s+day|fourth\s+of\s+july|july\s+(?:4th|fourth)", "independence_day"),
    (r"good\s+friday", "good_friday"),
    (r"easter\s+monday", "easter_monday"),
    (r"easter(?:\s+sunday)?", "easter"),
    (rf"mother{_APOS}s?{_APOS}\s+day|mothering\s+sunday", "mothers_day"),
    (rf"father{_APOS}s?{_APOS}\s+day", "fathers_day"),
    (r"memorial\s+day", "memorial_day"),
    (r"labou?r\s+day", "labor_day"),
)

# Names with a weekday inside: always a date, so "friday" is never read alone.
WEEKDAY_SHAPED = frozenset({"black_friday", "small_business_saturday", "cyber_monday", "giving_tuesday", "good_friday", "easter_monday"})


def holiday_date(key: str, year: int, order: str) -> date:
    return HOLIDAYS[key](year, order)


def next_holiday(key: str, today: date, order: str, year: int | None = None) -> date:
    """The date of ``key`` in ``year``, else this year's or (if passed) next year's."""
    if year is not None:
        return holiday_date(key, year, order)
    this_year = holiday_date(key, today.year, order)
    return this_year if this_year >= today else holiday_date(key, today.year + 1, order)
