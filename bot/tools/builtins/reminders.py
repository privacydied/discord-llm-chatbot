"""Tool: set a reminder for the person talking to the bot.
[CA][IV][SFT].

The router already handles messages that open with "remind me ..." in code.
This tool covers every other phrasing the model recognises ("don't let me
forget to ... on friday", "can u make sure i ... next week"). The model only
supplies the words; the reminder service parses the time with the same rules
as ``!remindme`` and uses the asker's own timezone. A reminder can only ever
ping the person who asked, in the channel they asked in.
"""

from __future__ import annotations

from typing import Any

from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

MAX_ARG_CHARS = 500

DESCRIPTION = (
    "Set a reminder that pings the person you're talking to at a future time, in their own timezone. "
    "Use whenever someone asks to be reminded of something, set a reminder or alarm, or not to let them forget. "
    'Pass `when` in the user\'s own words ("friday 5pm", "in 2 hours", "on may 2027", "tomorrow morning") '
    'or as their local time "YYYY-MM-DD HH:MM". Never convert to UTC yourself.'
)

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "when": {"type": "string", "description": "When to remind them, in their words or as local 'YYYY-MM-DD HH:MM'."},
        "what": {"type": "string", "description": "What to remind them about, short, e.g. 'post a fit pic'."},
        "timezone": {"type": "string", "description": "Only if the user stated their timezone or city in this conversation, e.g. 'London'. Otherwise omit."},
    },
    "required": ["when", "what"],
}

RELAY = "Tell the user in your own voice, and copy this line exactly (Discord renders the timestamps in each reader's local time): "


def _arg(arguments: dict[str, Any], key: str) -> str:
    return str(arguments.get(key) or "").strip()[:MAX_ARG_CHARS]


async def set_reminder(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Create a reminder via the bot's reminder service. Never raises."""
    service = getattr(ctx.bot, "reminder_service", None)
    if service is None or ctx.message is None:
        return ToolResult.failure("reminders aren't available right now")
    when, what = _arg(arguments, "when"), _arg(arguments, "what")
    if not when:
        return ToolResult.failure("no time given: ask the user when they want the reminder")
    from bot.reminders.chat import origin_from
    from bot.reminders.service import Status

    try:
        outcome = await service.create(origin_from(ctx.message), when, task=what, zone_hint=_arg(arguments, "timezone") or None)
    except Exception as exc:  # noqa: BLE001 - tools must not raise [REH]
        logger.error("tool.set_reminder failed: %s", exc, exc_info=True)
        return ToolResult.failure("the reminder couldn't be saved")
    logger.info("tool.set_reminder status=%s", outcome.status)
    if outcome.status is Status.NO_TIME:
        return ToolResult.failure(f"couldn't read a time from {when!r}: ask the user for a clearer time like 'friday 5pm' or 'in 2 hours'")
    if outcome.status in (Status.FAILED, Status.LIMIT):
        return ToolResult.failure(outcome.render())
    return ToolResult.success(RELAY + outcome.render())


SPEC = ToolSpec(
    name="set_reminder",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=set_reminder,
)
