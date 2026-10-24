"""Tool: reply with a voice note. [CA][REH]

"send a voice note" / "say it out loud" used to get "i can't send voice notes"
because voice only happened through !speak, !tts or the per-user/global toggle.
This tool lets the model recognise any phrasing: it marks the reply to the
current message as voice, and the normal send path (TTS -> native Discord voice
message, audio file fallback) speaks the model's next answer.
"""

from __future__ import annotations

from typing import Any

from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

DESCRIPTION = (
    "Send your reply as a voice note (spoken audio) instead of text. Use whenever someone asks you to "
    "send a voice note, voice message, vn, audio, to say something out loud, speak, or hear your voice "
    "-- you CAN do this. Call it, then write your reply normally: that reply is what gets spoken."
)

PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

READY = (
    "Voice note on: your next reply will be spoken and sent as a voice note. Write exactly what you'd say "
    "out loud: plain words, no markdown, links, lists, emojis or stage directions, and don't mention "
    "that it's a voice note or that you're sending audio."
)


def _unavailable_reason(ctx: ToolContext) -> str | None:
    tts = getattr(ctx.bot, "tts_manager", None)
    is_available = getattr(tts, "is_available", None)
    if tts is None or (callable(is_available) and not is_available()):
        return "voice notes aren't working right now (text-to-speech is down)"
    guild_id = getattr(getattr(ctx.message, "guild", None), "id", None)
    if guild_id is not None:
        from bot.server_features import is_server_feature_enabled

        if not is_server_feature_enabled(guild_id, "tts"):
            return "voice notes are switched off in this server"
    return None


async def send_voice_note(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Mark the reply to the current message as a voice note. Never raises."""
    message_id = getattr(ctx.message, "id", None)
    if message_id is None:
        return ToolResult.failure("there's no message to reply to")
    reason = _unavailable_reason(ctx)
    if reason:
        return ToolResult.failure(reason)
    from bot.tts.state import tts_state

    tts_state.request_voice_reply(message_id)
    logger.info("tool.send_voice_note message_id=%s", message_id)
    return ToolResult.success(READY)


SPEC = ToolSpec(
    name="send_voice_note",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=send_voice_note,
    timeout_s=5.0,
)
