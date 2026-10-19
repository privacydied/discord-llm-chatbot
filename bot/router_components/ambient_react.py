"""Ambient emoji reactions: pick one relevant emoji for a message, or none.

The gate (``ambient.should_ambient_react``) decides *whether* to try; this
module decides *what* to react with. One small LLM call returns a single
option from a fixed allowlist (unicode + the guild's own custom emoji) or
NONE. Anything off-list is discarded, so message text cannot steer the bot
into an arbitrary reaction. [SFT][IV]
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("bot.ambient")

# Curated unicode set: reactions a laid-back person in a group chat actually uses. [CMV]
UNICODE_EMOJI: tuple[str, ...] = (
    "💀",
    "😭",
    "🔥",
    "💯",
    "👀",
    "😂",
    "🤣",
    "🫡",
    "🙏",
    "🤝",
    "😮‍💨",
    "🤨",
    "😤",
    "🥀",
    "🧢",
    "🗿",
    "😐",
    "🙄",
    "🤔",
    "👏",
    "❤️",
    "🫠",
    "😬",
    "🤡",
    "🐐",
    "🍿",
    "⚰️",
    "📉",
    "📈",
    "✅",
    "❌",
)
MAX_CUSTOM_EMOJI = 50
MAX_MESSAGE_CHARS = 500
NONE_TOKEN = "NONE"

OUTCOME_FIRED = "fired"
OUTCOME_NONE = "none"
OUTCOME_REJECTED = "rejected"
OUTCOME_ERROR = "error"

SYSTEM_PROMPT = (
    "you pick emoji reactions for a dry, laid-back guy in a discord group chat. "
    "you only react when one emoji clearly fits what the message actually says — "
    "a joke, a take, good or bad news, something wild. generic or unclear messages get NONE. "
    "most messages get NONE. answer with exactly one option copied as written, or NONE. no other text."
)

Generate = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ReactResult:
    outcome: str
    emoji: Any = None  # str for unicode, discord.Emoji for custom
    raw: str = ""


def _norm(token: str) -> str:
    """Comparison key: drop variation selectors and wrapping punctuation."""
    return token.replace("\ufe0f", "").strip().strip("`'\".,!").strip()


def build_candidates(guild: Any) -> dict[str, Any]:
    """Map comparison key -> reaction object for every allowed option."""
    options: dict[str, Any] = {_norm(e): e for e in UNICODE_EMOJI}
    custom = [e for e in (getattr(guild, "emojis", None) or []) if getattr(e, "available", True)]
    for emoji in custom[:MAX_CUSTOM_EMOJI]:
        is_usable = getattr(emoji, "is_usable", None)
        if callable(is_usable) and not is_usable():
            continue
        options[f":{emoji.name.lower()}:"] = emoji
    return options


def build_prompt(content: str, options: dict[str, Any]) -> str:
    unicode_opts = " ".join(o for o in options.values() if isinstance(o, str))
    custom_opts = " ".join(k for k, o in options.items() if not isinstance(o, str))
    lines = [f"message:\n{content[:MAX_MESSAGE_CHARS]}", "", f"options: {unicode_opts}"]
    if custom_opts:
        lines.append(f"server emoji (judge by name): {custom_opts}")
    lines.append(f"\nreply with one option or {NONE_TOKEN}.")
    return "\n".join(lines)


def parse_choice(text: str, options: dict[str, Any]) -> tuple[str, Any]:
    """Return (outcome, emoji). Exact first line wins; else exactly one option mentioned anywhere."""
    text = (text or "").strip()
    if not text:
        return OUTCOME_REJECTED, None
    first = _norm(text.splitlines()[0])
    if first.upper() == NONE_TOKEN:
        return OUTCOME_NONE, None
    key = first.lower()
    for candidate in (key, f":{key.strip(':')}:"):
        if candidate in options:
            return OUTCOME_FIRED, options[candidate]
    if first in options:
        return OUTCOME_FIRED, options[first]
    lowered = text.replace("\ufe0f", "").lower()
    found = {k for k in options if k in lowered}
    if len(found) == 1:
        return OUTCOME_FIRED, options[found.pop()]
    return (OUTCOME_NONE if NONE_TOKEN in text.upper() else OUTCOME_REJECTED), None


def can_react(message: Any) -> bool:
    """Bot needs Add Reactions and Read Message History in the channel."""
    guild = getattr(message, "guild", None)
    channel = getattr(message, "channel", None)
    me = getattr(guild, "me", None)
    if guild is None or channel is None or me is None:
        return False
    try:
        perms = channel.permissions_for(me)
    except (AttributeError, TypeError):
        return False
    return bool(perms.add_reactions and perms.read_message_history)


async def choose_reaction(message: Any, generate: Generate, timeout_s: float) -> ReactResult:
    """Ask the model for one relevant emoji. Never raises."""
    options = build_candidates(getattr(message, "guild", None))
    prompt = build_prompt(getattr(message, "content", "") or "", options)
    try:
        response = await asyncio.wait_for(generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, stream=False), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 — a reaction is optional; any failure means no reaction
        logger.debug("ambient.react model call failed: %s", exc)
        return ReactResult(OUTCOME_ERROR, raw=str(exc)[:200])
    raw = str((response or {}).get("text") or "")
    try:
        from bot.vl.postprocess import sanitize_model_output

        raw = sanitize_model_output(raw)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ambient.react sanitize failed: %s", exc)
    outcome, emoji = parse_choice(raw, options)
    return ReactResult(outcome, emoji, raw[:200])


async def react_to_message(message: Any, generate: Generate, timeout_s: float) -> ReactResult:
    """Choose and apply a reaction. Discord errors downgrade to OUTCOME_ERROR."""
    result = await choose_reaction(message, generate, timeout_s)
    if result.outcome != OUTCOME_FIRED:
        return result
    try:
        await message.add_reaction(result.emoji)
    except Exception as exc:  # noqa: BLE001 — Forbidden/NotFound/HTTPException: message gone or perms changed
        logger.debug("ambient.react add_reaction failed: %s", exc)
        return ReactResult(OUTCOME_ERROR, result.emoji, str(exc)[:200])
    return result
