"""User-facing reminder text.
[CMV].

Times are written as Discord timestamps (``<t:unix:F>``): every reader's own
client shows them in *their* local time, so a reminder set by someone in New
York reads correctly to a friend in London too.
"""

from __future__ import annotations

from datetime import datetime

from .store import Reminder
from .timezones import describe_now

# A reminder this late was missed while the bot was offline. [CMV]
LATE_AFTER_S = 120
ZONE_EXAMPLES = "`London`, `New York`, `Texas`, `EST`, `UTC+2`"
USAGE = "e.g. `!remindme to stretch in 2 hours`, `!remindme to post a fit pic on may 2027`, `!remindme friday 5pm pay rent`"


def stamp(moment: datetime | None, style: str = "F") -> str:
    return f"<t:{int(moment.timestamp())}:{style}>" if moment else "?"


def _task(reminder: Reminder) -> str:
    return f"**{reminder.task}**" if reminder.task and "*" not in reminder.task else (reminder.task or "this")


def scheduled(reminder: Reminder) -> str:
    due = reminder.due
    zone = f"your time ({reminder.zone})" if reminder.zone else "from now"
    return f"⏰ got it, I'll remind you: {_task(reminder)}\n📅 {stamp(due)} ({stamp(due, 'R')})\n-# {zone} · reminder #{reminder.id} · `!reminders` to list or cancel"


def needs_timezone(reminder: Reminder) -> str:
    return f"⏰ saved {_task(reminder)} for *{reminder.when_text or 'later'}*, but I don't know your timezone yet.\nreply with your city or zone ({ZONE_EXAMPLES}) or use `!timezone <city>`, and I'll lock it in."


def no_time() -> str:
    return f"⏰ when should I remind you? {USAGE}"


def failed(reason: str) -> str:
    return f"⏰ couldn't set that reminder: {reason}."


def limit_reached(count: int) -> str:
    return f"⏰ you already have {count} reminders. cancel one first with `!reminders cancel <id>`."


def ambiguous_zone(text: str) -> str:
    return f"🌍 *{text}* has more than one timezone. tell me your city instead ({ZONE_EXAMPLES})."


def unknown_zone(text: str) -> str:
    return f"🌍 I don't recognise *{text}* as a timezone. try a city or zone ({ZONE_EXAMPLES})."


def zone_set(zone: str, activated: list[Reminder], problems: list[str]) -> str:
    lines = [f"🌍 timezone set to **{zone}** (it's {describe_now(zone)} there now)."]
    lines += [f"⏰ reminder #{r.id} {_task(r)} is set for {stamp(r.due)} ({stamp(r.due, 'R')})." for r in activated]
    lines += [f"⚠️ {p}" for p in problems]
    return "\n".join(lines)


def zone_current(zone: str | None) -> str:
    if not zone:
        return f"🌍 no timezone set. use `!timezone <city>` ({ZONE_EXAMPLES})."
    return f"🌍 your timezone is **{zone}** (it's {describe_now(zone)} there now)."


def listing(reminders: list[Reminder]) -> str:
    if not reminders:
        return f"⏰ you have no reminders. {USAGE}"
    lines = ["⏰ your reminders:"]
    for r in reminders:
        when = f"{stamp(r.due)} ({stamp(r.due, 'R')})" if r.due else f"*{r.when_text}*, waiting for your timezone"
        lines.append(f"`#{r.id}` {_task(r)} on {when}")
    lines.append("-# `!reminders cancel <id>` to remove one")
    return "\n".join(lines)


def fire(reminder: Reminder, now: datetime) -> str:
    text = f"⏰ <@{reminder.user_id}> reminder: {reminder.task}" if reminder.task else f"⏰ <@{reminder.user_id}> reminder!"
    meta = [f"set {stamp(reminder.created, 'R')}"]
    if reminder.jump_url:
        meta.append(f"[original message]({reminder.jump_url})")
    due = reminder.due
    if due and (now - due).total_seconds() > LATE_AFTER_S:
        meta.append(f"late, was due {stamp(due, 'f')} while I was offline")
    return f"{text}\n-# {' · '.join(meta)}"
