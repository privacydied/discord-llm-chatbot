"""Reminder records and their JSON file.
[CA][REH][RM].

One small JSON file holds every pending reminder plus each user's timezone.
Sent and cancelled reminders are deleted, so the file only ever holds the live
set. Writes are atomic (temp file + rename) via ``bot.atomic_json``; a corrupt
or missing file loads as empty instead of crashing startup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bot.atomic_json import read_json_safe, write_json_atomic
from bot.utils.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
DEFAULT_PATH = "data/reminders.json"


def _iso(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).isoformat(timespec="seconds") if moment else None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


@dataclass
class Reminder:
    """One reminder. ``due_at`` is None while waiting for the user's timezone."""

    id: int
    user_id: int
    task: str
    created_at: str
    due_at: str | None
    zone: str | None = None
    channel_id: int | None = None
    guild_id: int | None = None
    jump_url: str | None = None
    # The words that gave the time ("on may 2027"), for display.
    when_text: str = ""
    # Original request text, re-parsed once the user tells us their timezone.
    source_text: str = ""
    attempts: int = 0

    @property
    def due(self) -> datetime | None:
        return _parse_iso(self.due_at)

    @property
    def created(self) -> datetime:
        return _parse_iso(self.created_at) or datetime.now(UTC)

    def set_due(self, moment: datetime | None) -> None:
        self.due_at = _iso(moment)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reminder:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class ReminderStore:
    """In-memory reminders + timezones, persisted to one JSON file."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.reminders: dict[int, Reminder] = {}
        self.timezones: dict[str, str] = {}
        self._next_id = 1

    def load(self) -> None:
        """Read the file (sync, startup only). Bad rows are skipped and logged."""
        data = read_json_safe(self.path, default={}) or {}
        for row in data.get("reminders", []) if isinstance(data, dict) else []:
            try:
                reminder = Reminder.from_dict(row)
                self.reminders[reminder.id] = reminder
            except (TypeError, ValueError) as exc:
                logger.warning(f"reminders.load.bad_row | {exc}")
        self.timezones = {str(k): str(v) for k, v in (data.get("timezones") or {}).items()} if isinstance(data, dict) else {}
        self._next_id = max([int(data.get("next_id", 1)) if isinstance(data, dict) else 1, *[r.id + 1 for r in self.reminders.values()]])
        logger.info(f"reminders.loaded | count={len(self.reminders)} zones={len(self.timezones)} path={self.path}")

    async def save(self) -> None:
        payload = {
            "version": SCHEMA_VERSION,
            "next_id": self._next_id,
            "reminders": [asdict(r) for r in sorted(self.reminders.values(), key=lambda r: r.id)],
            "timezones": self.timezones,
        }
        await write_json_atomic(self.path, payload)

    def add(self, reminder: Reminder) -> Reminder:
        reminder.id = self._next_id
        self._next_id += 1
        self.reminders[reminder.id] = reminder
        return reminder

    def owned_by(self, user_id: int) -> list[Reminder]:
        mine = [r for r in self.reminders.values() if r.user_id == user_id]
        return sorted(mine, key=lambda r: (r.due is None, r.due or r.created, r.id))

    def scheduled(self) -> list[Reminder]:
        """Every reminder with a due time, soonest first."""
        return sorted((r for r in self.reminders.values() if r.due_at), key=lambda r: (r.due, r.id))
