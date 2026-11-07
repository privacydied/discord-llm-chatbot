"""Create, re-time and cancel reminders.
[CA][REH][IV].

The one place reminders are made, shared by ``!remindme``, the natural-language
"remind me ..." path and the ``set_reminder`` model tool, so all three apply
the same parsing, limits and timezone rules.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from bot.utils.logging import get_logger

from . import messages
from .parsing import WhenError, parse_reminder
from .store import Reminder, ReminderStore
from .timezones import AmbiguousTimezoneError, resolve_timezone

logger = get_logger(__name__)

DEFAULT_MAX_PER_USER = 25
MAX_TASK_CHARS = 500
# Task words that mean "the message I'm replying to". [CMV]
DEICTIC_TASKS = frozenset({"this", "that", "it", "this message", "that message", "this one", "that one", "this msg", "that msg"})
REPLIED_TASK = "that message"


class Status(StrEnum):
    SCHEDULED = "scheduled"
    NEEDS_TIMEZONE = "needs_timezone"
    NO_TIME = "no_time"
    FAILED = "failed"
    LIMIT = "limit"


@dataclass(frozen=True)
class Origin:
    """Who asked and where: the reminder pings back in the same channel."""

    user_id: int
    channel_id: int | None = None
    guild_id: int | None = None
    jump_url: str | None = None
    # Set when the request replies to another message ("remind me about this").
    referenced_url: str | None = None


@dataclass(frozen=True)
class Outcome:
    status: Status
    reminder: Reminder | None = None
    reason: str = ""

    def render(self) -> str:
        if self.status is Status.SCHEDULED and self.reminder:
            return messages.scheduled(self.reminder)
        if self.status is Status.NEEDS_TIMEZONE and self.reminder:
            return messages.needs_timezone(self.reminder)
        if self.status is Status.LIMIT:
            return messages.limit_reached(int(self.reason or 0))
        return messages.no_time() if self.status is Status.NO_TIME else messages.failed(self.reason or "unknown error")


def _hint_zone(zone_hint: str | None) -> str | None:
    try:
        found = resolve_timezone(zone_hint)
    except AmbiguousTimezoneError:
        return None
    return found.name if found else None


def _task_and_link(task: str, origin: Origin) -> tuple[str, str | None]:
    task = " ".join(task.split())[:MAX_TASK_CHARS]
    if origin.referenced_url and (not task or task.lower() in DEICTIC_TASKS):
        return REPLIED_TASK, origin.referenced_url
    return task, origin.jump_url


class ReminderService:
    """Owns the store; the scheduler waits on ``changed`` for new due times."""

    def __init__(self, store: ReminderStore, *, max_per_user: int = DEFAULT_MAX_PER_USER) -> None:
        self.store = store
        self.max_per_user = max_per_user
        self.lock = asyncio.Lock()
        self.changed = asyncio.Event()

    def timezone_of(self, user_id: int) -> str | None:
        return self.store.timezones.get(str(user_id))

    def list_for(self, user_id: int) -> list[Reminder]:
        return self.store.owned_by(user_id)

    async def create(self, origin: Origin, text: str, *, task: str | None = None, zone_hint: str | None = None, now: datetime | None = None) -> Outcome:
        """Parse ``text`` and save a reminder. With ``task`` given, ``text`` is only the time."""
        now = now or datetime.now(UTC)
        count = len(self.list_for(origin.user_id))
        if count >= self.max_per_user:
            return Outcome(Status.LIMIT, reason=str(count))
        try:
            parsed = parse_reminder(text, zone_name=_hint_zone(zone_hint) or self.timezone_of(origin.user_id), now=now)
        except (WhenError, ValueError) as exc:
            return Outcome(Status.FAILED, reason=str(exc))
        if parsed is None:
            return Outcome(Status.NO_TIME)
        task_text, link = _task_and_link(parsed.task if task is None else task, origin)
        reminder = Reminder(
            id=0,
            user_id=origin.user_id,
            task=task_text,
            created_at=now.astimezone(UTC).isoformat(timespec="seconds"),
            due_at=None,
            zone=parsed.zone,
            channel_id=origin.channel_id,
            guild_id=origin.guild_id,
            jump_url=link,
            when_text=parsed.when_text,
            source_text=text,
        )
        reminder.set_due(parsed.due_utc)
        return await self._save_new(reminder)

    async def _save_new(self, reminder: Reminder) -> Outcome:
        async with self.lock:
            self.store.add(reminder)
            try:
                await self.store.save()
            except OSError as exc:
                self.store.reminders.pop(reminder.id, None)
                logger.error(f"reminders.save_failed | {exc}")
                return Outcome(Status.FAILED, reason="I couldn't save it, try again in a bit")
        self.changed.set()
        status = Status.SCHEDULED if reminder.due_at else Status.NEEDS_TIMEZONE
        logger.info(f"reminders.created | id={reminder.id} user={reminder.user_id} status={status} due={reminder.due_at} zone={reminder.zone}")
        return Outcome(status, reminder)

    def _retime(self, reminder: Reminder, zone: str, now: datetime) -> str | None:
        """Give a waiting reminder its due time in ``zone``. Returns a problem or None."""
        try:
            parsed = parse_reminder(reminder.source_text, zone_name=zone, now=reminder.created)
        except (WhenError, ValueError) as exc:
            return f"reminder #{reminder.id} dropped: {exc}"
        if parsed is None or parsed.due_utc is None:
            return f"reminder #{reminder.id} dropped: couldn't read its time any more"
        reminder.zone = parsed.zone
        reminder.set_due(max(parsed.due_utc, now))
        return None

    async def set_timezone(self, user_id: int, zone: str, *, now: datetime | None = None) -> tuple[list[Reminder], list[str]]:
        """Store the user's zone and schedule reminders that were waiting for it."""
        now = now or datetime.now(UTC)
        activated: list[Reminder] = []
        problems: list[str] = []
        async with self.lock:
            self.store.timezones[str(user_id)] = zone
            for reminder in [r for r in self.list_for(user_id) if not r.due_at]:
                problem = self._retime(reminder, zone, now)
                if problem:
                    self.store.reminders.pop(reminder.id, None)
                    problems.append(problem)
                else:
                    activated.append(reminder)
            await self.store.save()
        self.changed.set()
        logger.info(f"reminders.timezone_set | user={user_id} zone={zone} activated={len(activated)}")
        return activated, problems

    async def cancel(self, user_id: int, reminder_id: int) -> Reminder | None:
        """Delete one of the user's own reminders. None if not found or not theirs."""
        async with self.lock:
            reminder = self.store.reminders.get(reminder_id)
            if reminder is None or reminder.user_id != user_id:
                return None
            del self.store.reminders[reminder_id]
            await self.store.save()
        self.changed.set()
        logger.info(f"reminders.cancelled | id={reminder_id} user={user_id}")
        return reminder
