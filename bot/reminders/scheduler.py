"""Deliver reminders when they fall due.
[REH][RM][PA].

One background task sleeps until the next due reminder (at most
``MAX_SLEEP_S``) and wakes early whenever a reminder is added or changed.
Reminders missed while the bot was offline fire on startup, marked late.

Delivery pings the user in the channel the reminder was set in; if that
channel is gone or the bot lost access, it falls back to a DM. Transient
Discord errors retry with backoff; a DM the user has blocked is dropped.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import discord

from bot.core.output import safe_send
from bot.utils.logging import get_logger

from . import messages
from .service import ReminderService
from .store import Reminder

logger = get_logger(__name__)

MAX_SLEEP_S = 300.0
MIN_SLEEP_S = 0.5
SEND_TIMEOUT_S = 15.0
MAX_ATTEMPTS = 5
RETRY_BASE = timedelta(minutes=1)
LOOP_ERROR_BACKOFF_S = 30.0


class PermanentDeliveryError(RuntimeError):
    """Neither the channel nor a DM can reach the user; retrying won't help."""


def _mentions(user_id: int) -> discord.AllowedMentions:
    # Ping the owner only: a task containing @everyone or a role must not ping. [SFT]
    return discord.AllowedMentions(everyone=False, roles=False, users=[discord.Object(id=user_id)], replied_user=False)


async def _channel(bot, channel_id: int | None):
    if not channel_id:
        return None
    channel = bot.get_channel(channel_id)
    if channel is not None:
        return channel
    try:
        return await bot.fetch_channel(channel_id)
    except (discord.NotFound, discord.Forbidden, discord.InvalidData):
        return None


async def deliver(bot, reminder: Reminder, now: datetime) -> None:
    """Send the reminder. Raises PermanentDeliveryError or a transient error."""
    content = messages.fire(reminder, now)
    channel = await _channel(bot, reminder.channel_id)
    if channel is not None:
        try:
            await safe_send(channel, content, allowed_mentions=_mentions(reminder.user_id))
            return
        except (discord.Forbidden, discord.NotFound) as exc:
            logger.warning(f"reminders.channel_send_failed | id={reminder.id} channel={reminder.channel_id} err={exc}; trying DM")
    try:
        user = bot.get_user(reminder.user_id) or await bot.fetch_user(reminder.user_id)
        await safe_send(user, content, allowed_mentions=_mentions(reminder.user_id))
    except (discord.Forbidden, discord.NotFound) as exc:
        raise PermanentDeliveryError(str(exc)) from exc


class ReminderScheduler:
    """Background loop that fires due reminders."""

    def __init__(self, bot, service: ReminderService, *, sender=deliver) -> None:
        self.bot = bot
        self.service = service
        self._send = sender
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="reminders.scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        await self.bot.wait_until_ready()
        logger.info(f"reminders.scheduler.started | pending={len(self.service.store.reminders)}")
        while True:
            try:
                self.service.changed.clear()
                await self.fire_due()
                delay = self._sleep_for(datetime.now(UTC))
            except Exception as exc:  # noqa: BLE001 - the loop must outlive one bad reminder [REH]
                logger.error(f"reminders.scheduler.error | {exc}", exc_info=True)
                delay = LOOP_ERROR_BACKOFF_S
            with suppress(TimeoutError):
                await asyncio.wait_for(self.service.changed.wait(), timeout=delay)

    def _sleep_for(self, now: datetime) -> float:
        upcoming = self.service.store.scheduled()
        if not upcoming:
            return MAX_SLEEP_S
        return max(MIN_SLEEP_S, min(MAX_SLEEP_S, ((upcoming[0].due or now) - now).total_seconds()))

    async def fire_due(self, now: datetime | None = None) -> int:
        """Fire every reminder due by ``now``. Returns how many were delivered."""
        now = now or datetime.now(UTC)
        due = [r for r in self.service.store.scheduled() if (r.due or now) <= now]
        results = [await self._fire_one(r, now) for r in due]
        return sum(results)

    async def _fire_one(self, reminder: Reminder, now: datetime) -> bool:
        error: str | None = None
        permanent = False
        try:
            await asyncio.wait_for(self._send(self.bot, reminder, now), timeout=SEND_TIMEOUT_S)
        except PermanentDeliveryError as exc:
            error, permanent = str(exc), True
        except (discord.HTTPException, TimeoutError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        await self._settle(reminder, now, error, permanent)
        return error is None

    async def _settle(self, reminder: Reminder, now: datetime, error: str | None, permanent: bool) -> None:
        async with self.service.lock:
            if error is None:
                self.service.store.reminders.pop(reminder.id, None)
                logger.info(f"reminders.fired | id={reminder.id} user={reminder.user_id}")
            else:
                reminder.attempts += 1
                if permanent or reminder.attempts >= MAX_ATTEMPTS:
                    self.service.store.reminders.pop(reminder.id, None)
                    logger.error(f"reminders.dropped | id={reminder.id} user={reminder.user_id} attempts={reminder.attempts} err={error}")
                else:
                    reminder.set_due(now + RETRY_BASE * (2 ** (reminder.attempts - 1)))
                    logger.warning(f"reminders.retry | id={reminder.id} attempt={reminder.attempts} next={reminder.due_at} err={error}")
            try:
                await self.service.store.save()
            except OSError as exc:
                logger.error(f"reminders.save_failed | {exc}")
