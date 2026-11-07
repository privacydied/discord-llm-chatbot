"""Turn what a person types ("london", "EST", "new york", "UTC+5:30") into a timezone.
[IV][CMV].

Discord never tells a bot a user's timezone, so reminders ask once and remember
it. Accepted spellings, most specific first:

1. an IANA name, any case (``America/New_York``, ``europe/london``);
2. a fixed UTC offset (``UTC+5:30``, ``GMT-4``, ``+01:00``);
3. a common abbreviation or nickname (``EST``, ``pacific``, ``uk``);
4. a city that names an IANA zone (``london``, ``new york``, ``tokyo``);
5. a US state (``texas``) or a country with one timezone (``germany``).

A country that spans several zones (US, Canada, Australia...) is reported as
ambiguous so the caller can ask for a city instead of guessing.
"""

from __future__ import annotations

import re
import zoneinfo
from importlib import resources
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from functools import lru_cache

# Zone names stored for fixed offsets look like "UTC+05:30". [CMV]
OFFSET_PREFIX = "UTC"
MAX_OFFSET_HOURS = 14

_OFFSET_RE = re.compile(r"^(?:utc|gmt)?\s*(?P<sign>[+-])\s*(?P<h>\d{1,2})(?::?(?P<m>\d{2}))?$", re.IGNORECASE)
_FILLER_RE = re.compile(r"\b(?:time\s*zone|timezone|tz|time|standard|daylight|zone)\b", re.IGNORECASE)

# Abbreviations and nicknames. GMT/BST map to UK time: Brits say "GMT" for
# their local clock all year, so a fixed +0 would be an hour wrong all summer.
ALIASES: dict[str, str] = {
    "utc": "UTC",
    "z": "UTC",
    "zulu": "UTC",
    "gmt": "Europe/London",
    "bst": "Europe/London",
    "uk": "Europe/London",
    "u.k.": "Europe/London",
    "england": "Europe/London",
    "scotland": "Europe/London",
    "wales": "Europe/London",
    "britain": "Europe/London",
    "great britain": "Europe/London",
    "northern ireland": "Europe/London",
    "united kingdom": "Europe/London",
    "ireland": "Europe/Dublin",
    "ist": "Asia/Kolkata",
    "india": "Asia/Kolkata",
    "est": "America/New_York",
    "edt": "America/New_York",
    "eastern": "America/New_York",
    "et": "America/New_York",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "central": "America/Chicago",
    "ct": "America/Chicago",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "mountain": "America/Denver",
    "mt": "America/Denver",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "pacific": "America/Los_Angeles",
    "pt": "America/Los_Angeles",
    "akst": "America/Anchorage",
    "akdt": "America/Anchorage",
    "alaska": "America/Anchorage",
    "hst": "Pacific/Honolulu",
    "hawaii": "Pacific/Honolulu",
    "ast": "America/Halifax",
    "atlantic": "America/Halifax",
    "nst": "America/St_Johns",
    "newfoundland": "America/St_Johns",
    "cet": "Europe/Paris",
    "cest": "Europe/Paris",
    "eet": "Europe/Athens",
    "eest": "Europe/Athens",
    "wet": "Europe/Lisbon",
    "msk": "Europe/Moscow",
    "jst": "Asia/Tokyo",
    "kst": "Asia/Seoul",
    "sgt": "Asia/Singapore",
    "hkt": "Asia/Hong_Kong",
    "pkt": "Asia/Karachi",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
    "acst": "Australia/Adelaide",
    "awst": "Australia/Perth",
    "nzst": "Pacific/Auckland",
    "nzdt": "Pacific/Auckland",
    "nz": "Pacific/Auckland",
    "new zealand": "Pacific/Auckland",
    "sast": "Africa/Johannesburg",
    "wat": "Africa/Lagos",
    "eat": "Africa/Nairobi",
    "brt": "America/Sao_Paulo",
    "art": "America/Argentina/Buenos_Aires",
    # Big cities whose name is not an IANA zone.
    "nyc": "America/New_York",
    "la": "America/Los_Angeles",
    "sf": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "las vegas": "America/Los_Angeles",
    "washington dc": "America/New_York",
    "dc": "America/New_York",
    "boston": "America/New_York",
    "miami": "America/New_York",
    "atlanta": "America/New_York",
    "philadelphia": "America/New_York",
    "houston": "America/Chicago",
    "dallas": "America/Chicago",
    "austin": "America/Chicago",
    "san antonio": "America/Chicago",
    "nashville": "America/Chicago",
    "new orleans": "America/Chicago",
    "minneapolis": "America/Chicago",
    "salt lake city": "America/Denver",
    "san diego": "America/Los_Angeles",
    "portland": "America/Los_Angeles",
    "montreal": "America/Toronto",
    "ottawa": "America/Toronto",
    "manchester": "Europe/London",
    "birmingham": "Europe/London",
    "leeds": "Europe/London",
    "liverpool": "Europe/London",
    "glasgow": "Europe/London",
    "edinburgh": "Europe/London",
    "cardiff": "Europe/London",
    "bristol": "Europe/London",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "new delhi": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "beijing": "Asia/Shanghai",
    "china": "Asia/Shanghai",
    "dubai": "Asia/Dubai",
    "melbourne": "Australia/Melbourne",
}

# State -> majority zone. Split states use the zone most residents live in.
US_STATES: dict[str, str] = {
    **dict.fromkeys(
        "connecticut delaware georgia maine maryland massachusetts michigan ohio pennsylvania vermont virginia".split(),
        "America/New_York",
    ),
    **dict.fromkeys(["new hampshire", "new jersey", "new york state", "north carolina", "rhode island", "south carolina", "west virginia", "florida", "indiana", "kentucky"], "America/New_York"),
    **dict.fromkeys("alabama arkansas illinois iowa kansas louisiana minnesota mississippi missouri nebraska oklahoma tennessee texas wisconsin".split(), "America/Chicago"),
    **dict.fromkeys(["north dakota", "south dakota"], "America/Chicago"),
    **dict.fromkeys("colorado idaho montana utah wyoming".split(), "America/Denver"),
    **dict.fromkeys(["new mexico"], "America/Denver"),
    "arizona": "America/Phoenix",
    **dict.fromkeys("california nevada oregon washington".split(), "America/Los_Angeles"),
}

# Countries spanning several zones: always ask for a city. [IV]
AMBIGUOUS_REGIONS: frozenset[str] = frozenset({"us", "usa", "u.s.", "u.s.a.", "america", "united states", "the us", "the states", "states", "canada", "australia", "russia", "brazil", "mexico", "indonesia"})


class AmbiguousTimezoneError(ValueError):
    """The name covers more than one timezone (a multi-zone country)."""


@dataclass(frozen=True)
class TimezoneMatch:
    """A resolved zone: storable ``name`` and how it was recognised."""

    name: str
    source: str


def _clean(text: str) -> str:
    text = _FILLER_RE.sub(" ", (text or "").strip().lower())
    return " ".join(text.replace("_", " ").strip(" .,!?:;\"'`").split())


def _offset_name(sign: str, hours: int, minutes: int) -> str | None:
    if hours > MAX_OFFSET_HOURS or minutes >= 60:
        return None
    return f"{OFFSET_PREFIX}{sign}{hours:02d}:{minutes:02d}"


def _match_offset(raw: str) -> str | None:
    match = _OFFSET_RE.match(raw.strip())
    if not match:
        return None
    return _offset_name(match.group("sign"), int(match.group("h")), int(match.group("m") or 0))


@lru_cache(maxsize=1)
def _zone_index() -> dict[str, str]:
    """lowercase IANA name and lowercase city part -> canonical IANA name."""
    index: dict[str, str] = {}
    for name in sorted(zoneinfo.available_timezones()):
        if name.startswith(("Etc/", "SystemV/")) or "/" not in name:
            continue
        index[name.lower()] = name
        city = name.rsplit("/", 1)[-1].replace("_", " ").lower()
        index.setdefault(city, name)
    return index


@lru_cache(maxsize=512)
def _iana_zone(name: str) -> zoneinfo.ZoneInfo:
    """Load ``name`` from the pinned ``tzdata`` package, not the host's zoneinfo.

    The NAS ships tzdata 2026b while the package is 2026f; zones whose rules
    changed in between (e.g. America/Winnipeg) came out an hour wrong. The
    host copy is only a fallback for names the package lacks. [IV][CMV]
    """
    try:
        with resources.files("tzdata.zoneinfo").joinpath(*name.split("/")).open("rb") as handle:
            return zoneinfo.ZoneInfo.from_file(handle, key=name)
    except (ModuleNotFoundError, FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError):
        return zoneinfo.ZoneInfo(name)


def _same_clock(zones: list[str]) -> bool:
    """True when every zone shows the same UTC offset all year (one effective zone)."""
    start = datetime(datetime.now(UTC).year, 1, 1, tzinfo=UTC)
    samples = [start + timedelta(days=30 * i) for i in range(13)]
    offsets = {tuple(s.astimezone(_iana_zone(z)).utcoffset() for s in samples) for z in zones}
    return len(offsets) == 1


@lru_cache(maxsize=1)
def country_zones() -> dict[str, list[str]]:
    """lowercase country name -> its IANA zones, from tzdata's own tables."""
    try:
        from importlib.resources import files

        base = files("tzdata").joinpath("zoneinfo")
        names = dict(line.split("\t", 1) for line in base.joinpath("iso3166.tab").read_text().splitlines() if line and not line.startswith("#"))
        rows = [line.split("\t") for line in base.joinpath("zone.tab").read_text().splitlines() if line and not line.startswith("#")]
    except (ImportError, OSError, ValueError):
        return {}
    by_code: dict[str, list[str]] = {}
    for row in rows:
        by_code.setdefault(row[0], []).append(row[2])
    out: dict[str, list[str]] = {}
    for code, zones in by_code.items():
        label = names.get(code, "").strip().lower()
        for alias in {label, re.sub(r"\s*\(.*\)$", "", label)} - {""}:
            out[alias] = zones
    return out


def _match_country(key: str) -> str | None:
    zones = country_zones().get(key)
    if not zones:
        return None
    if len(zones) == 1 or _same_clock(zones):
        return zones[0]
    raise AmbiguousTimezoneError(key)


def resolve_timezone(text: str | None) -> TimezoneMatch | None:
    """Resolve free text to a timezone, or None. Raises AmbiguousTimezoneError."""
    raw = (text or "").strip()
    if not raw:
        return None
    offset = _match_offset(raw)
    if offset:
        return TimezoneMatch(offset, "offset")
    key = _clean(raw)
    if not key:
        return None
    if key in AMBIGUOUS_REGIONS:
        raise AmbiguousTimezoneError(key)
    if key in ALIASES:
        return TimezoneMatch(ALIASES[key], "alias")
    index = _zone_index()
    slashed = key.replace(" ", "_") if "/" in key else key
    for candidate in (slashed, key):
        if candidate in index:
            return TimezoneMatch(index[candidate], "iana")
    if key in US_STATES:
        return TimezoneMatch(US_STATES[key], "us_state")
    country = _match_country(key)
    return TimezoneMatch(country, "country") if country else None


def load_zone(name: str) -> tzinfo:
    """Stored zone name -> tzinfo. Raises ValueError on an unknown name."""
    offset = _OFFSET_RE.match(name) if name.startswith(OFFSET_PREFIX) and len(name) > len(OFFSET_PREFIX) else None
    if offset:
        delta = timedelta(hours=int(offset.group("h")), minutes=int(offset.group("m") or 0))
        return timezone(delta if offset.group("sign") == "+" else -delta, name)
    try:
        return _iana_zone(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone {name!r}") from exc


def describe_now(name: str, now: datetime | None = None) -> str:
    """'Sat 7 Nov, 18:30' in the zone: lets a user confirm the zone is right."""
    local = (now or datetime.now(UTC)).astimezone(load_zone(name))
    return local.strftime("%a %-d %b, %H:%M")
