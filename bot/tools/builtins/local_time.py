"""Tool: current local time in a named place, or in the asker's own timezone.
[CA][IV].

"@bot what's the time in new york" -> 13:30 EST. Uses the same place-name
resolution as ``!timezone`` (cities, IANA names, EST/PST, UTC+5:30, US states,
countries), so the bot and the reminder system agree on what a place means.
A country that spans several zones (US, Canada, Australia...) answers with
each of its main zones instead of guessing one. Read-only: touches nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from bot.reminders.timezones import AmbiguousTimezoneError, country_zones, load_zone, resolve_timezone
from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

MAX_LOCATION_CHARS = 100
MAX_ZONES_LISTED = 8

# Main zones for countries that span several, west to east. [IV]
_US = ["Pacific/Honolulu", "America/Anchorage", "America/Los_Angeles", "America/Denver", "America/Phoenix", "America/Chicago", "America/New_York"]
MULTI_ZONE_MAIN: dict[str, list[str]] = {
    **dict.fromkeys(["us", "usa", "u.s.", "u.s.a.", "america", "united states", "the us", "the states", "states"], _US),
    "canada": ["America/Vancouver", "America/Edmonton", "America/Winnipeg", "America/Toronto", "America/Halifax", "America/St_Johns"],
    "australia": ["Australia/Perth", "Australia/Darwin", "Australia/Adelaide", "Australia/Brisbane", "Australia/Sydney"],
    "russia": ["Europe/Kaliningrad", "Europe/Moscow", "Asia/Yekaterinburg", "Asia/Novosibirsk", "Asia/Irkutsk", "Asia/Vladivostok", "Asia/Kamchatka"],
    "brazil": ["America/Rio_Branco", "America/Manaus", "America/Sao_Paulo", "America/Noronha"],
    "mexico": ["America/Tijuana", "America/Hermosillo", "America/Mexico_City", "America/Cancun"],
    "indonesia": ["Asia/Jakarta", "Asia/Makassar", "Asia/Jayapura"],
}

DESCRIPTION = (
    "Get the current local date and time in a place: a city, country, US state, timezone name or abbreviation "
    "(e.g. 'New York', 'Japan', 'Texas', 'PST', 'Europe/London', 'UTC+5:30'). Use for any question about what time "
    "or day it is somewhere, or the time difference between places (call once per place). Omit `location` for the "
    "asking user's own local time. If a small town isn't recognised, retry with its country or nearest major city."
)

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "Place or timezone, e.g. 'new york'. Omit for the user's own time."},
    },
    "required": [],
}


def _line(label: str, zone: str, now: datetime) -> str:
    local = now.astimezone(load_zone(zone))
    offset = local.strftime("%z")
    offset = f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"
    abbrev = local.tzname() or ""
    tag = f"{offset}, {abbrev}" if abbrev and not abbrev.startswith(("UTC", "+", "-")) else offset
    return f"{label} ({zone}): {local.strftime('%A %-d %B %Y, %H:%M')} ({local.strftime('%-I:%M %p').lower()}), {tag}"


def _multi_zones(key: str, now: datetime) -> list[str]:
    zones = MULTI_ZONE_MAIN.get(key)
    if zones is None:  # another multi-zone country: one zone per distinct current offset
        by_offset: dict[Any, str] = {}
        for zone in country_zones().get(key, []):
            by_offset.setdefault(now.astimezone(load_zone(zone)).utcoffset(), zone)
        zones = [by_offset[k] for k in sorted(by_offset)]
    return zones[:MAX_ZONES_LISTED]


def _own_zone(ctx: ToolContext) -> str | None:
    service = getattr(ctx.bot, "reminder_service", None)
    author = getattr(getattr(ctx.message, "author", None), "id", None)
    if service is None or author is None:
        return None
    try:
        return service.timezone_of(int(author))
    except (AttributeError, TypeError, ValueError):
        return None


def _own_time(ctx: ToolContext, now: datetime) -> ToolResult:
    zone = _own_zone(ctx)
    if zone:
        return ToolResult.success("The user's saved timezone. " + _line("Their local time", zone, now))
    return ToolResult.success(_line("UTC", "UTC", now) + ". The user's own timezone is unknown: ask where they are, or tell them `!timezone <city>` saves it.")


async def get_local_time(ctx: ToolContext, arguments: dict[str, Any], *, now: datetime | None = None) -> ToolResult:
    """Return the local time for ``location`` (or the asker). Never raises."""
    now = now or datetime.now(UTC)
    location = " ".join(str(arguments.get("location") or "").split())[:MAX_LOCATION_CHARS]
    try:
        if not location:
            return _own_time(ctx, now)
        try:
            found = resolve_timezone(location)
        except AmbiguousTimezoneError as exc:
            zones = _multi_zones(str(exc.args[0] if exc.args else location).lower(), now)
            if not zones:
                return ToolResult.failure(f"{location!r} spans several timezones: ask which city")
            lines = [_line(" ".join(z.split("/")[-1].split("_")), z, now) for z in zones]
            return ToolResult.success(f"{location} spans several timezones:\n" + "\n".join(lines))
        if found is None:
            return ToolResult.failure(f"unknown place {location!r}: retry with its country or nearest major city")
        return ToolResult.success(_line(location, found.name, now))
    except (ValueError, OSError) as exc:  # bad zone data must not break the reply [REH]
        logger.error("tool.get_local_time failed: %s", exc, exc_info=True)
        return ToolResult.failure("couldn't work out the time there")


SPEC = ToolSpec(
    name="get_local_time",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=get_local_time,
)
