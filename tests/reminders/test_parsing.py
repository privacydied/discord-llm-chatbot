"""Reminder time parsing: wall-clock vs floating times, zones, and task text."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from bot.reminders.parsing import WhenError, parse_reminder

# Saturday 7 Nov 2026, 18:30 UTC (= 18:30 London, 13:30 New York).
NOW = datetime(2026, 11, 7, 18, 30, tzinfo=UTC)
LONDON = "Europe/London"
NEW_YORK = "America/New_York"


def _local(text: str, zone: str | None = LONDON) -> tuple[str, str]:
    parsed = parse_reminder(text, zone_name=zone, now=NOW)
    assert parsed is not None and parsed.due_utc is not None, text
    shown = parsed.due_utc.astimezone(ZoneInfo(parsed.zone or zone or "UTC"))
    return shown.strftime("%Y-%m-%d %H:%M"), parsed.task


@pytest.mark.parametrize(
    ("text", "expected", "task"),
    [
        ("to post a fit pic on may 2027", "2027-05-01 09:00", "post a fit pic"),
        ("to post a fit pic in may", "2027-05-01 09:00", "post a fit pic"),
        ("to call mum tomorrow at 5pm", "2026-11-08 17:00", "call mum"),
        ("tomorrow 9am gym", "2026-11-08 09:00", "gym"),
        ("tomorrow morning check car", "2026-11-08 09:00", "check car"),
        ("to take meds at 9pm", "2026-11-07 21:00", "take meds"),
        ("at noon to eat", "2026-11-08 12:00", "eat"),
        ("next friday to pay rent", "2026-11-13 09:00", "pay rent"),
        ("saturday at 10am football", "2026-11-14 10:00", "football"),
        ("friday at 5 drinks", "2026-11-13 17:00", "drinks"),
        ("tonight at 11 to sleep", "2026-11-07 23:00", "sleep"),
        ("to post on the 15th", "2026-11-15 09:00", "post"),
        ("on the 31st", "2026-12-31 09:00", ""),
        ("on 3rd june to renew passport", "2027-06-03 09:00", "renew passport"),
        ("may 5 at 3:30pm call bank", "2027-05-05 15:30", "call bank"),
        ("2027-05-01 14:00 launch", "2027-05-01 14:00", "launch"),
        ("to email boss at 9:15", "2026-11-08 09:15", "email boss"),
        ("in 2 days at 5pm meet", "2026-11-09 17:00", "meet"),
        ("this weekend clean", "2026-11-14 09:00", "clean"),
        ("to watch 1917 on friday", "2026-11-13 09:00", "watch 1917"),
    ],
)
def test_wall_clock_in_users_zone(text, expected, task):
    assert _local(text) == (expected, task)


def test_same_words_mean_each_users_own_clock():
    """'9pm' is 21:00 London for a Brit and 21:00 New York for an American."""
    brit = parse_reminder("at 9pm", zone_name=LONDON, now=NOW)
    american = parse_reminder("at 9pm", zone_name=NEW_YORK, now=NOW)
    assert brit.due_utc == datetime(2026, 11, 7, 21, 0, tzinfo=UTC)
    assert american.due_utc == datetime(2026, 11, 8, 2, 0, tzinfo=UTC)


def test_dst_is_applied_for_summer_dates():
    parsed = parse_reminder("on may 2027", zone_name=LONDON, now=NOW)
    assert parsed.due_utc == datetime(2027, 5, 1, 8, 0, tzinfo=UTC)  # 09:00 BST


@pytest.mark.parametrize(
    ("text", "expected_utc"),
    [
        ("in 2 hours", datetime(2026, 11, 7, 20, 30, tzinfo=UTC)),
        ("in 1h30m stretch", datetime(2026, 11, 7, 20, 0, tzinfo=UTC)),
        ("in 3 hours and 20 minutes", datetime(2026, 11, 7, 21, 50, tzinfo=UTC)),
        ("in half an hour tea", datetime(2026, 11, 7, 19, 0, tzinfo=UTC)),
        ("to buy 2 apples tomorrow", datetime(2026, 11, 8, 18, 30, tzinfo=UTC)),
        ("next week dentist", datetime(2026, 11, 14, 18, 30, tzinfo=UTC)),
        ("1 year from now", datetime(2027, 11, 7, 18, 30, tzinfo=UTC)),
        ("in a couple of days", datetime(2026, 11, 9, 18, 30, tzinfo=UTC)),
    ],
)
def test_floating_times_need_no_timezone(text, expected_utc):
    parsed = parse_reminder(text, zone_name=None, now=NOW)
    assert parsed.due_utc == expected_utc
    assert parsed.zone is None and not parsed.needs_timezone


def test_wall_clock_without_zone_asks_for_one():
    parsed = parse_reminder("to post a fit pic on may 2027", zone_name=None, now=NOW)
    assert parsed.needs_timezone and parsed.due_utc is None
    assert parsed.task == "post a fit pic" and parsed.when_text == "on may 2027"


@pytest.mark.parametrize(
    ("text", "zone", "expected_utc"),
    [
        ("to stream at 8pm EST", None, datetime(2026, 11, 8, 1, 0, tzinfo=UTC)),
        ("at 5pm london time call", NEW_YORK, datetime(2026, 11, 8, 17, 0, tzinfo=UTC)),
        ("at 9am utc+2", LONDON, datetime(2026, 11, 8, 7, 0, tzinfo=UTC)),
    ],
)
def test_zone_written_in_the_text_wins(text, zone, expected_utc):
    assert parse_reminder(text, zone_name=zone, now=NOW).due_utc == expected_utc


def test_numeric_dates_follow_the_users_locale():
    assert _local("on 5/6 bday", LONDON)[0] == "2027-06-05 09:00"
    assert _local("on 5/6 bday", NEW_YORK)[0] == "2027-05-06 09:00"
    assert _local("on 12/25 open presents", LONDON)[0] == "2026-12-25 09:00"  # 25 can't be a month


@pytest.mark.parametrize("text", ["to sit down", "remind about the sun", "buy 2 apples", "i may need milk", "post a fit pic"])
def test_no_time_returns_none(text):
    assert parse_reminder(text, zone_name=LONDON, now=NOW) is None


@pytest.mark.parametrize("text", ["on 31/02", "in 11 years", "on 25/12/2020", "today feed cat"])
def test_unusable_times_raise(text):
    with pytest.raises(WhenError):
        parse_reminder(text, zone_name=LONDON, now=NOW)
