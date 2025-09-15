import json
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
import stat
from bot.utils.env import get_bool

import discord

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages ephemeral conversation context, storing recent messages in a JSON file
    or in-memory, with separation for guilds/channels and DMs.
    """

    def __init__(
        self,
        bot: discord.Client,
        filepath: str = "runtime/context.json",
        max_messages: int = 10,
    ):
        """
        Initializes the ContextManager.

        Args:
            filepath (str): The path to the JSON file for context storage.
            max_messages (int): The maximum number of messages to store per context.
        """
        self.filepath = filepath
        self.max_messages = int(os.getenv("MAX_CONTEXT_MESSAGES", max_messages))
        self.in_memory_only = get_bool("IN_MEMORY_CONTEXT_ONLY", False)
        self.bot = bot
        self.memory: Dict[str, Any] = {}
        self._load()
        logger.info(
            f"ContextManager initialized. In-memory only: {self.in_memory_only}, Max messages: {self.max_messages}"
        )

    def _get_source_keys(self, message: discord.Message) -> Tuple[str, Optional[str]]:
        """
        Determines the primary and secondary keys for context storage from a message.

        For guild messages, returns (guild_id_key, channel_id_key).
        For DMs, returns (dm_user_id_key, None).

        Args:
            message (discord.Message): The Discord message object.

        Returns:
            Tuple[str, Optional[str]]: A tuple containing the primary and secondary keys.
        """
        if message.guild:
            return f"guild_{message.guild.id}", f"channel_{message.channel.id}"
        else:
            return f"dm_{message.author.id}", None

    def _load(self):
        """Loads the context from the JSON file into memory."""
        if self.in_memory_only:
            logger.info("Context storage is in-memory only. Skipping file load.")
            return

        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
                logger.info(f"Successfully loaded context from {self.filepath}")
            else:
                logger.info(
                    f"Context file not found at {self.filepath}. Starting with empty context."
                )
        except (json.JSONDecodeError, IOError) as e:
            logger.error(
                f"Failed to load or parse context file at {self.filepath}. Using in-memory fallback. Error: {e}"
            )
            self.memory = {}

    def _save(self):
        """Saves the current in-memory context to the JSON file.

        This is part of the storage backend, which could be replaced by Redis, etc.
        Security: Ensure the context file has restrictive permissions (e.g., 600).
        Privacy: DM conversations (keys starting with 'dm_') are NOT saved to disk.
        """
        if self.in_memory_only:
            return

        # Filter out DM conversations to enforce privacy
        guild_context_only = {
            k: v for k, v in self.memory.items() if not k.startswith("dm_")
        }

        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(guild_context_only, f, indent=2)
            # Post-write permission check (best-effort)
            try:
                st = os.stat(self.filepath)
                mode = stat.S_IMODE(st.st_mode)
                if mode != 0o600:
                    if get_bool("STRICT_CONTEXT_PERMS", False):
                        os.chmod(self.filepath, 0o600)
                        logger.info(
                            "Hardened context file permissions to 600",
                            extra={
                                "subsys": "context",
                                "event": "context.perms.hardened",
                            },
                        )
                    else:
                        logger.warning(
                            "Context file permissions are not 600 (mode=%o). Consider: chmod 600 %s",
                            mode,
                            self.filepath,
                        )
            except Exception:
                # Never fail route on permission checks
                pass
        except IOError as e:
            logger.error(f"Failed to save context to {self.filepath}. Error: {e}")

    def append(self, message: discord.Message):
        """Appends a message to the appropriate context history."""
        primary_key, secondary_key = self._get_source_keys(message)

        entry = {
            "author_id": str(message.author.id),
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
        }

        if secondary_key:  # Guild message
            self.memory.setdefault(primary_key, {}).setdefault(
                secondary_key, []
            ).append(entry)
            self.memory[primary_key][secondary_key] = self.memory[primary_key][
                secondary_key
            ][-self.max_messages :]
        else:  # DM
            self.memory.setdefault(primary_key, []).append(entry)
            self.memory[primary_key] = self.memory[primary_key][-self.max_messages :]

        self._save()

    def get_context(self, message: discord.Message) -> List[Dict[str, str]]:
        """Retrieves the context for a given message's source."""
        primary_key, secondary_key = self._get_source_keys(message)

        if secondary_key:  # Guild message
            return self.memory.get(primary_key, {}).get(secondary_key, [])
        else:  # DM
            return self.memory.get(primary_key, [])

    async def get_context_string(self, message: discord.Message) -> str:
        """Retrieves and formats the context into a string for the LLM prompt."""
        context_history = self.get_context(message)
        if not context_history:
            return ""

        # Cache for user details to avoid redundant API calls within a single format operation
        user_cache: Dict[str, str] = {}

        lines = []
        for entry in context_history:
            author_id = entry.get("author_id")
            username = user_cache.get(author_id)

            if not username:
                try:
                    user = (
                        await message.guild.fetch_member(author_id)
                        if message.guild
                        else await self.bot.fetch_user(author_id)
                    )
                    username = user.display_name
                    user_cache[author_id] = username
                except (discord.NotFound, discord.HTTPException):
                    username = f"User ({author_id})"

            lines.append(f"[{username}]: {entry.get('content', '')}")

        return "\n".join(lines)
