"""Timezone resolution and plain-words intent detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bot.reminders.intent import parse_reminder_request, parse_timezone_statement
from bot.reminders.timezones import AmbiguousTimezoneError, load_zone, resolve_timezone


@pytest.mark.parametrize(
    ("text", "zone"),
    [
        ("London", "Europe/London"),
        ("europe/london", "Europe/London"),
        ("UK", "Europe/London"),
        ("GMT", "Europe/London"),
        ("new york", "America/New_York"),
        ("EST", "America/New_York"),
        ("pacific time", "America/Los_Angeles"),
        ("texas", "America/Chicago"),
        ("Tokyo", "Asia/Tokyo"),
        ("germany", "Europe/Berlin"),
        ("India", "Asia/Kolkata"),
        ("utc", "UTC"),
        ("UTC+5:30", "UTC+05:30"),
        ("gmt-4", "UTC-04:00"),
    ],
)
def test_resolve_timezone(text, zone):
    assert resolve_timezone(text).name == zone


@pytest.mark.parametrize("text", ["usa", "America", "canada", "australia"])
def test_multi_zone_countries_are_ambiguous(text):
    with pytest.raises(AmbiguousTimezoneError):
        resolve_timezone(text)


@pytest.mark.parametrize("text", ["", "lol ok", "pizza", "utc+19"])
def test_unknown_timezone_is_none(text):
    assert resolve_timezone(text) is None


def test_fixed_offset_zone_loads():
    zone = load_zone("UTC+05:30")
    assert datetime(2026, 1, 1, tzinfo=UTC).astimezone(zone).utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize(
    ("text", "rest", "explicit"),
    [
        ("remind me to post a fit pic on may 2027", "to post a fit pic on may 2027", True),
        ("yo can you remind me in 2 hours to stretch", "in 2 hours to stretch", False),
        ("set a reminder for friday 5pm: pay rent", "for friday 5pm: pay rent", True),
        ("Set me a reminder tomorrow gym", "tomorrow gym", True),
        ("reminder: dentist on the 15th", "dentist on the 15th", True),
        ("remindme in 5 mins", "in 5 mins", True),
        ("don't let me forget to call mum tomorrow", "to call mum tomorrow", True),
        ("ping me in an hour", "in an hour", False),
    ],
)
def test_reminder_request_detected(text, rest, explicit):
    request = parse_reminder_request(text)
    assert request is not None and (request.rest, request.explicit) == (rest, explicit)


@pytest.mark.parametrize(
    "text",
    ["remind me what we talked about", "can you remind me who that was", "i forgot to set a reminder lol", "what's a reminder", "remind me how this works"],
)
def test_not_a_reminder_request(text):
    assert parse_reminder_request(text) is None


def test_timezone_statements():
    assert parse_timezone_statement("my timezone is London", awaiting_zone=False).zone_text == "London"
    assert parse_timezone_statement("set my timezone to new york", awaiting_zone=False).explicit
    assert parse_timezone_statement("london", awaiting_zone=False) is None
    bare = parse_timezone_statement("london", awaiting_zone=True)
    assert bare.zone_text == "london" and not bare.explicit
    assert parse_timezone_statement("i'm in texas", awaiting_zone=True).zone_text == "texas"
    assert parse_timezone_statement("this is a long sentence about my day today", awaiting_zone=True) is None
