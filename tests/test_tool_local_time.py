"""get_local_time tool: '@bot what's the time in new york'."""

from __future__ import annotations

import zoneinfo
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from bot.reminders import timezones
from bot.tools.builtins.local_time import get_local_time
from bot.tools.types import ToolContext

# Saturday 7 Nov 2026, 18:30 UTC.
NOW = datetime(2026, 11, 7, 18, 30, tzinfo=UTC)


async def _ask(location: str | None, ctx: ToolContext | None = None):
    args = {} if location is None else {"location": location}
    return await get_local_time(ctx or ToolContext(), args, now=NOW)


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("new york", "America/New_York): Saturday 7 November 2026, 13:30 (1:30 pm), UTC-05:00, EST"),
        ("Tokyo", "Asia/Tokyo): Sunday 8 November 2026, 03:30 (3:30 am), UTC+09:00, JST"),
        ("UK", "Europe/London): Saturday 7 November 2026, 18:30 (6:30 pm), UTC+00:00, GMT"),
        ("texas", "America/Chicago): Saturday 7 November 2026, 12:30"),
        ("PST", "America/Los_Angeles): Saturday 7 November 2026, 10:30"),
        ("india", "Asia/Kolkata): Sunday 8 November 2026, 00:00 (12:00 am), UTC+05:30, IST"),
        ("utc+5:30", "UTC+05:30): Sunday 8 November 2026, 00:00 (12:00 am), UTC+05:30"),
    ],
)
async def test_time_in_a_place(location, expected):
    result = await _ask(location)
    assert result.ok and expected in result.content


async def test_multi_zone_country_lists_main_zones():
    result = await _ask("usa")
    assert result.ok and "spans several timezones" in result.content
    assert "New York (America/New_York): Saturday 7 November 2026, 13:30" in result.content
    assert "Los Angeles (America/Los_Angeles): Saturday 7 November 2026, 10:30" in result.content
    assert "Sydney" in (await _ask("australia")).content


async def test_unknown_place_asks_model_to_retry():
    result = await _ask("narnia")
    assert not result.ok and "nearest major city" in (result.error or "")


async def test_no_location_uses_asker_saved_zone():
    service = SimpleNamespace(timezone_of=lambda uid: "Asia/Tokyo" if uid == 7 else None)
    ctx = ToolContext(message=SimpleNamespace(author=SimpleNamespace(id=7)), bot=SimpleNamespace(reminder_service=service))
    result = await _ask(None, ctx)
    assert result.ok and "saved timezone" in result.content and "03:30" in result.content

    stranger = ToolContext(message=SimpleNamespace(author=SimpleNamespace(id=8)), bot=SimpleNamespace(reminder_service=service))
    result = await _ask("", stranger)
    assert result.ok and "UTC" in result.content and "!timezone" in result.content


def test_zones_load_from_pinned_tzdata_not_host(monkeypatch):
    """The NAS host zoneinfo is older than the tzdata package; the package must win."""

    class HostZoneInfo:
        from_file = staticmethod(zoneinfo.ZoneInfo.from_file)

        def __new__(cls, name):
            raise AssertionError(f"host zoneinfo used for {name}")

    timezones._iana_zone.cache_clear()
    monkeypatch.setattr(timezones, "zoneinfo", SimpleNamespace(ZoneInfo=HostZoneInfo, ZoneInfoNotFoundError=zoneinfo.ZoneInfoNotFoundError))
    try:
        zone = timezones.load_zone("America/Winnipeg")
        assert getattr(zone, "key", None) == "America/Winnipeg"
        assert NOW.astimezone(zone).utcoffset() is not None
    finally:
        monkeypatch.undo()
        timezones._iana_zone.cache_clear()


async def test_registered_in_allowlist():
    from bot.tools.builtins import BUILTIN_SPECS
    from bot.tools.registry import ALLOWED_TOOL_NAMES

    assert "get_local_time" in ALLOWED_TOOL_NAMES
    assert any(spec.name == "get_local_time" for spec in BUILTIN_SPECS)
