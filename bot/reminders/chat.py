"""Glue between Discord messages and the reminder service.
[CA][IV].

Used by the router for addressed plain-language messages ("@bot remind me to
... friday") and by the ``!remindme`` / ``!timezone`` commands, so both paths
build the same ``Origin`` and give the same answers.
"""

from __future__ import annotations

from typing import Any

import discord

from . import messages
from .intent import parse_reminder_request, parse_timezone_statement
from .service import Origin, ReminderService, Status
from .timezones import AmbiguousTimezoneError, resolve_timezone


def _referenced_url(message: Any) -> str | None:
    reference = getattr(message, "reference", None)
    if reference is None or not getattr(reference, "message_id", None):
        return None
    resolved = getattr(reference, "resolved", None)
    if isinstance(resolved, discord.Message):
        return resolved.jump_url
    guild_id = getattr(reference, "guild_id", None) or "@me"
    return f"https://discord.com/channels/{guild_id}/{reference.channel_id}/{reference.message_id}"


def origin_from(message: Any) -> Origin:
    """Where a reminder made from ``message`` should ping back."""
    return Origin(
        user_id=int(message.author.id),
        channel_id=getattr(getattr(message, "channel", None), "id", None),
        guild_id=getattr(getattr(message, "guild", None), "id", None),
        jump_url=getattr(message, "jump_url", None),
        referenced_url=_referenced_url(message),
    )


def resolve_zone_text(zone_text: str) -> tuple[str | None, str]:
    """(zone, '') when recognised, else (None, message explaining why)."""
    try:
        found = resolve_timezone(zone_text)
    except AmbiguousTimezoneError:
        return None, messages.ambiguous_zone(zone_text)
    return (found.name, "") if found else (None, messages.unknown_zone(zone_text))


async def set_zone_reply(service: ReminderService, user_id: int, zone_text: str) -> str:
    """Set the user's zone from free text; the reply covers reminders it unblocked."""
    zone, problem = resolve_zone_text(zone_text)
    return await _apply_zone(service, user_id, zone) if zone else problem


async def _apply_zone(service: ReminderService, user_id: int, zone: str) -> str:
    activated, problems = await service.set_timezone(user_id, zone)
    return messages.zone_set(zone, activated, problems)


async def handle_chat_message(service: ReminderService, message: Any, text: str) -> str | None:
    """Reply text for a reminder/timezone request, or None to let chat answer."""
    request = parse_reminder_request(text)
    if request is not None:
        outcome = await service.create(origin_from(message), request.rest)
        if outcome.status is Status.NO_TIME and not request.explicit:
            return None  # "remind me about the plot" with no time: just chat
        return outcome.render()
    user_id = int(message.author.id)
    awaiting = any(r.due_at is None for r in service.list_for(user_id))
    statement = parse_timezone_statement(text, awaiting_zone=awaiting)
    if statement is None:
        return None
    zone, problem = resolve_zone_text(statement.zone_text)
    if zone is None:
        return problem if statement.explicit else None
    return await _apply_zone(service, user_id, zone)
