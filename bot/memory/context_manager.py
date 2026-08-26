import asyncio
import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

import discord

from bot.config import load_config as _load_config
from bot.utils.env import get_bool

logger = logging.getLogger(__name__)


def _load_int_config(key: str, default: int) -> int:
    """Load an integer config value that respects LOW_RESOURCE_MODE via load_config()."""
    try:
        cfg = _load_config()
        val = cfg.get(key)
        if val is not None:
            return int(val)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.debug(f"Failed to load int config {key}: {e}")
    return default


def _load_bool_config(key: str, default: bool) -> bool:
    """Load a bool config value, respecting the config dict."""
    try:
        cfg = _load_config()
        val = cfg.get(key)
        if val is not None:
            return bool(val)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.debug(f"Failed to load bool config {key}: {e}")
    return default


def _check_context_perms(filepath: str, harden: bool) -> tuple[int | None, bool]:
    """Stat (and optionally chmod) the context file. Blocking; call via a thread.

    Returns (current_mode, hardened). ``hardened`` is True only when this call
    actually changed the permissions.
    """
    st = os.stat(filepath)
    mode = stat.S_IMODE(st.st_mode)
    if mode != 0o600 and harden:
        os.chmod(filepath, 0o600)
        return mode, True
    return mode, False


class ContextManager:
    """Manages ephemeral conversation context, storing recent messages in a JSON file
    or in-memory, with separation for guilds/channels and DMs.
    """

    def __init__(
        self,
        bot: discord.Client,
        filepath: str = "runtime/context.json",
        max_messages: int = 10,
    ) -> None:
        """Initializes the ContextManager.

        Args:
            filepath (str): The path to the JSON file for context storage.
            max_messages (int): The maximum number of messages to store per context.

        """
        self.bot = bot
        self.filepath = filepath
        self.max_messages = int(os.getenv("MAX_CONTEXT_MESSAGES", max_messages))
        self.in_memory_only = get_bool("IN_MEMORY_CONTEXT_ONLY", False)
        # Context trimming limits
        self.max_chars_per_message = _load_int_config("CONTEXT_MAX_CHARS_PER_MESSAGE", 2000)
        self.max_total_chars = _load_int_config("CONTEXT_MAX_TOTAL_CHARS", 8000)
        self.ignore_continuation_chunks = _load_bool_config("CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS", True)
        self.memory: dict[str, Any] = {}
        # Dirty-flag debounce: append() used to fire a fire-and-forget disk write
        # (full-file atomic rewrite + fsync) on every single message. Reads always
        # come from self.memory in-process, never from disk, so nothing needs the
        # write to land synchronously -- only a periodic flush (see
        # bot/tasks.py's context_autosave task) and flush-on-shutdown need it to
        # land at all. [PA]
        self._dirty = False
        self._load()
        logger.info(f"ContextManager initialized. In-memory only: {self.in_memory_only}, Max messages: {self.max_messages}, Max chars/msg: {self.max_chars_per_message}, Max total chars: {self.max_total_chars}")

    def _get_source_keys(self, message: discord.Message) -> tuple[str, str | None]:
        """Determines the primary and secondary keys for context storage from a message.

        For guild messages, returns (guild_id_key, channel_id_key).
        For DMs, returns (dm_user_id_key, None).

        Args:
            message (discord.Message): The Discord message object.

        Returns:
            Tuple[str, Optional[str]]: A tuple containing the primary and secondary keys.

        """
        if message.guild:
            return f"guild_{message.guild.id}", f"channel_{message.channel.id}"
        return f"dm_{message.author.id}", None

    def _load(self) -> None:
        """Loads the context from the JSON file into memory."""
        if self.in_memory_only:
            logger.info("Context storage is in-memory only. Skipping file load.")
            return

        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, encoding="utf-8") as f:
                    self.memory = json.load(f)
                logger.info(f"Successfully loaded context from {self.filepath}")
            else:
                logger.info(f"Context file not found at {self.filepath}. Starting with empty context.")
        except (OSError, json.JSONDecodeError) as e:
            logger.exception(f"Failed to load or parse context file at {self.filepath}. Using in-memory fallback. Error: {e}")
            self.memory = {}

    async def _save(self) -> None:
        """Saves the current in-memory context to the JSON file atomically.

        This is part of the storage backend, which could be replaced by Redis, etc.
        Security: Ensure the context file has restrictive permissions (e.g., 600).
        Privacy: DM conversations (keys starting with 'dm_') are NOT saved to disk.
        """
        if self.in_memory_only:
            return

        # Filter out DM conversations to enforce privacy
        guild_context_only = {k: v for k, v in self.memory.items() if not k.startswith("dm_")}

        try:
            from bot.atomic_json import write_json_atomic

            await write_json_atomic(Path(self.filepath), guild_context_only)
            # Post-write permission check (best-effort). stat/chmod are blocking
            # syscalls and the context file may live on a network mount, so they
            # run in a thread rather than on the event loop. [PA][RM]
            try:
                harden = get_bool("STRICT_CONTEXT_PERMS", False)
                mode, hardened = await asyncio.to_thread(
                    _check_context_perms,
                    self.filepath,
                    harden,
                )
                if hardened:
                    logger.info(
                        "Hardened context file permissions to 600",
                        extra={
                            "subsys": "context",
                            "event": "context.perms.hardened",
                        },
                    )
                elif mode is not None and mode != 0o600:
                    logger.warning(
                        "Context file permissions are not 600 (mode=%o). Consider: chmod 600 %s",
                        mode,
                        self.filepath,
                    )
            except (OSError, ValueError, AttributeError, TypeError) as e:
                # Never fail route on permission checks
                logger.debug(f"Permission check failed: {e}")
        except OSError as e:
            logger.exception(f"Failed to save context to {self.filepath}. Error: {e}")

    def append(self, message: discord.Message) -> None:
        """Appends a message to the appropriate context history."""
        primary_key, secondary_key = self._get_source_keys(message)

        # Truncate content to per-message char limit
        raw_content = message.content or ""
        if len(raw_content) > self.max_chars_per_message:
            raw_content = raw_content[: self.max_chars_per_message]

        entry = {
            "author_id": str(message.author.id),
            "content": raw_content,
            "timestamp": message.created_at.isoformat(),
        }

        if secondary_key:  # Guild message
            self.memory.setdefault(primary_key, {}).setdefault(secondary_key, []).append(entry)
            self.memory[primary_key][secondary_key] = self.memory[primary_key][secondary_key][-self.max_messages :]
        else:  # DM
            self.memory.setdefault(primary_key, []).append(entry)
            self.memory[primary_key] = self.memory[primary_key][-self.max_messages :]

        # Mark dirty instead of writing to disk on every single message [PA].
        # A periodic task (bot/tasks.py context_autosave) and shutdown both call
        # flush_if_dirty(); in-process reads always hit self.memory directly so
        # there's no correctness gap, only a bounded window of at-most-N-seconds
        # of context that could be lost on an unclean crash (same tradeoff
        # already accepted for user/server profile autosave).
        self._dirty = True

    async def flush_if_dirty(self) -> bool:
        """Persist to disk if there are unsaved changes; no-op otherwise.

        Returns:
            True if a save was performed, False if there was nothing to flush.

        """
        if not self._dirty or self.in_memory_only:
            return False
        self._dirty = False
        await self._save()
        return True

    def get_context(self, message: discord.Message) -> list[dict[str, str]]:
        """Retrieves the context for a given message's source."""
        primary_key, secondary_key = self._get_source_keys(message)

        if secondary_key:  # Guild message
            return self.memory.get(primary_key, {}).get(secondary_key, [])
        # DM
        return self.memory.get(primary_key, [])

    async def get_context_string(self, message: discord.Message) -> str:
        """Retrieves and formats the context into a string for the LLM prompt."""
        context_history = self.get_context(message)
        if not context_history:
            return ""

        # Filter out continuation chunks if configured
        filtered = []
        for entry in context_history:
            if self.ignore_continuation_chunks:
                content = entry.get("content", "")
                stripped = content.strip()
                if stripped in ("...", "…", "more", "**(continues)**", "(more)", "more..."):
                    continue
            filtered.append(entry)

        # Build from newest to oldest to prioritize recent messages,
        # then reverse to chronological order. This drops oldest when
        # total char cap is exceeded.
        lines = []
        total_chars = 0

        # Cache for user details
        user_cache: dict[str, str] = {}

        for entry in reversed(filtered):
            content = entry.get("content", "")
            author_id = entry.get("author_id")
            username = user_cache.get(author_id)

            if not username:
                try:
                    user = await message.guild.fetch_member(author_id) if message.guild else await self.bot.fetch_user(author_id)
                    username = user.display_name
                    user_cache[author_id] = username
                except (discord.NotFound, discord.HTTPException):
                    username = f"User ({author_id})"

            line = f"[{username}]: {content}"
            line_chars = len(line)
            if total_chars + line_chars > self.max_total_chars and lines:
                break

            lines.append(line)
            total_chars += line_chars

        return "\n".join(reversed(lines))
