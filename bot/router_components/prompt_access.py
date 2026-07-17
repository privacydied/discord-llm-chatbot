"""Prompt-access helpers for Router decomposition."""

from __future__ import annotations

from typing import Any

from bot.utils.logging import get_logger

logger = get_logger(__name__)


def get_system_prompt(bot: Any, key: str, default: str | None = None) -> str | None:
    """Safely read prompt templates from bot.system_prompts.

    Best-effort by contract.  `system_prompts` is duck-typed: any object with
    a `.get` may be supplied, including custom/lazy loaders that raise
    arbitrary exceptions.  Every failure yields `default`, because a prompt
    lookup must never take down message handling -- callers pass `default`
    precisely so there is always something to fall back to.
    """
    try:
        prompts = getattr(bot, "system_prompts", None) or {}
        getter = getattr(prompts, "get", None)
        if callable(getter):
            return getter(key, default)
        return default
    except Exception as exc:
        # Deliberately broad: the getter is caller-supplied, so the exception
        # type is unbounded.  Logged rather than swallowed silently.
        logger.debug("system_prompt lookup failed for key=%r: %s", key, exc)
        return default
