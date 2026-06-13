"""Prompt-access helpers for Router decomposition."""

from __future__ import annotations

from typing import Any


def get_system_prompt(bot: Any, key: str, default: str | None = None) -> str | None:
    """Safely read prompt templates from bot.system_prompts."""
    try:
        prompts = getattr(bot, "system_prompts", None) or {}
        getter = getattr(prompts, "get", None)
        if callable(getter):
            return getter(key, default)
        return default
    except Exception:
        return default
