"""Enhanced contextual conversation manager for Discord bot.
Handles multi-user conversation tracking across threads and DMs with privacy controls.
"""

import asyncio
import contextlib
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import discord
from cryptography.fernet import Fernet

from bot.config import load_config as _load_config
from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Bounded LRU cache size for decrypted-content memoization. Fernet decrypt is
# AES+HMAC (real sync CPU); the same ciphertext is decrypted repeatedly within
# and across requests inside the history window, so a small bounded cache keyed
# by ciphertext keeps repeat decrypts cheap without growing unboundedly. [PA]
DECRYPT_CACHE_MAX_ENTRIES = 512


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
    """Load a bool config value from config dict."""
    try:
        cfg = _load_config()
        val = cfg.get(key)
        if val is not None:
            return bool(val)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.debug(f"Failed to load bool config {key}: {e}")
    return default


@dataclass
class MessageEntry:
    """Represents a stored message with all required metadata."""

    user_id: str
    channel_id: str
    thread_id: str | None
    timestamp: str
    role: str  # 'user' or 'bot'
    content: str
    guild_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageEntry":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ContextResponse:
    """Response envelope for context retrieval."""

    response_text: str
    used_history: list[dict[str, Any]]
    fallback: bool


class EnhancedContextManager:
    """Enhanced contextual conversation manager that tracks multi-user conversations
    across threads and DMs with privacy controls and encryption.
    """

    def __init__(
        self,
        bot: discord.Client,
        filepath: str = "runtime/enhanced_context.json",
        history_window: int | None = None,
        max_token_limit: int = 4000,
        encryption_key: bytes | None = None,
    ) -> None:
        """Initialize the enhanced context manager.

        Args:
            bot: Discord bot instance
            filepath: Path to context storage file
            history_window: Max messages per user/channel (from HISTORY_WINDOW env var)
            max_token_limit: Max tokens for context to avoid prompt bloat
            encryption_key: Key for encrypting stored content

        """
        self.bot = bot
        self.filepath = filepath
        self.history_window = history_window or int(os.getenv("HISTORY_WINDOW", "10"))
        self.max_token_limit = max_token_limit
        self.in_memory_only = os.getenv("IN_MEMORY_CONTEXT_ONLY", "false").lower() == "true"

        # Context trimming limits — respect LOW_RESOURCE_MODE via load_config()
        self.max_chars_per_message = _load_int_config("CONTEXT_MAX_CHARS_PER_MESSAGE", 2000)
        self.max_total_chars = _load_int_config("CONTEXT_MAX_TOTAL_CHARS", 8000)
        self.ignore_continuation_chunks = _load_bool_config("CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS", True)
        # Override message count cap with config if available
        self.max_messages = _load_int_config("CONTEXT_MAX_MESSAGES", self.history_window)

        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate or load encryption key
            key_file = f"{filepath}.key"
            if os.path.exists(key_file):
                with open(key_file, "rb") as f:
                    self.cipher = Fernet(f.read())
            else:
                key = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(key)
                os.chmod(key_file, 0o600)  # Restrict permissions
                self.cipher = Fernet(key)

        # Bounded LRU memo for _decrypt_content, keyed by ciphertext. Avoids
        # repeated AES+HMAC work when the same entry is decrypted multiple
        # times per request and across requests within the history window. [PA]
        self._decrypt_cache: OrderedDict[str, str] = OrderedDict()

        # Storage: {context_key: List[MessageEntry]}
        self.memory: dict[str, list[MessageEntry]] = {}

        # Privacy opt-out tracking: {user_id: bool}
        self.privacy_opt_outs: dict[str, bool] = {}

        # Concurrency protection for thread-safe operations
        self._memory_lock = asyncio.Lock()

        # Dirty-flag debounce: append_message() used to do a full-file atomic
        # rewrite + fsync of *every* tracked guild/channel/thread's history on
        # every single message (and again on every bot reply). Reads always come
        # from self.memory in-process, so the write can be deferred to a
        # periodic flush (bot/tasks.py's context_autosave task) and
        # flush-on-shutdown instead of landing synchronously per message. [PA]
        self._dirty = False

        self._load()
        logger.info(f"✔ Enhanced context manager initialized [history_window={self.history_window}, encryption=enabled]")

    def _get_context_key(self, message: discord.Message) -> str:
        """Generate context key for message storage."""
        if isinstance(message.channel, discord.DMChannel):
            return f"dm_{message.author.id}"
        if hasattr(message.channel, "parent") and message.channel.parent:
            # Thread in a channel
            return f"guild_{message.guild.id}_thread_{message.channel.id}"
        # Regular channel
        return f"guild_{message.guild.id}_channel_{message.channel.id}"

    def _get_user_context_key(self, user_id: str, channel_id: str, guild_id: str | None = None) -> str:
        """Generate user-specific context key."""
        if guild_id:
            return f"guild_{guild_id}_channel_{channel_id}_user_{user_id}"
        return f"dm_{user_id}"

    def _encrypt_content(self, content: str) -> str:
        """Encrypt message content."""
        try:
            return self.cipher.encrypt(content.encode()).decode()
        except Exception as e:
            logger.exception(f"❌ Encryption failed: {e}")
            return content  # Fallback to unencrypted

    def _decrypt_content(self, encrypted_content: str) -> str:
        """Decrypt message content (memoized by ciphertext, bounded LRU). [PA]."""
        cached = self._decrypt_cache.get(encrypted_content)
        if cached is not None:
            self._decrypt_cache.move_to_end(encrypted_content)
            return cached

        try:
            plaintext = self.cipher.decrypt(encrypted_content.encode()).decode()
        except (ValueError, TypeError, RuntimeError) as e:
            logger.debug(f"Decryption failed, assuming unencrypted: {e}")
            plaintext = encrypted_content  # Assume it's unencrypted

        self._decrypt_cache[encrypted_content] = plaintext
        if len(self._decrypt_cache) > DECRYPT_CACHE_MAX_ENTRIES:
            self._decrypt_cache.popitem(last=False)  # evict oldest [RM]
        return plaintext

    def _load(self) -> None:
        """Load context from storage file."""
        if self.in_memory_only:
            logger.info("Context storage is in-memory only")
            return

        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, encoding="utf-8") as f:
                    data = json.load(f)

                # Load message entries
                for key, entries in data.get("messages", {}).items():
                    self.memory[key] = [MessageEntry.from_dict(entry) for entry in entries]

                # Load privacy opt-outs
                self.privacy_opt_outs = data.get("privacy_opt_outs", {})

                logger.info(f"✔ Loaded enhanced context from {self.filepath}")
            else:
                logger.info(f"Context file not found at {self.filepath}, starting fresh")
        except Exception as e:
            logger.exception(f"❌ Failed to load context: {e}")
            self.memory = {}
            self.privacy_opt_outs = {}

    async def _save(self) -> None:
        """Save context to storage file with privacy filtering (atomic writes)."""
        if self.in_memory_only:
            return

        from bot.atomic_json import write_json_atomic

        try:
            # Filter out DM conversations to enforce privacy
            filtered_messages = {k: [entry.to_dict() for entry in v] for k, v in self.memory.items() if not k.startswith("dm_")}

            data = {
                "messages": filtered_messages,
                "privacy_opt_outs": self.privacy_opt_outs,
                "metadata": {
                    "version": "1.0",
                    "last_updated": datetime.now(UTC).isoformat(),
                },
            }

            await write_json_atomic(Path(self.filepath), data)

            # Post-write permission hardening. chmod is a blocking syscall and the
            # context file may live on a network mount, so keep it off the loop. [PA]
            with contextlib.suppress(Exception):
                await asyncio.to_thread(os.chmod, self.filepath, 0o600)

        except Exception as e:
            logger.exception("Failed to save context: %s", e)

    def set_privacy_opt_out(self, user_id: str, opt_out: bool = True) -> None:
        """Set privacy opt-out for a user."""
        self.privacy_opt_outs[str(user_id)] = opt_out
        # Fire-and-forget async save (sync caller)
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            task = loop.create_task(self._save())
            task.add_done_callback(
                lambda t: (
                    logger.warning(
                        f"Enhanced context save failed: {t.exception()}",
                        exc_info=t.exception(),
                    )
                    if not t.cancelled() and t.exception()
                    else None
                ),
            )
        except RuntimeError:
            # No event loop — skip save (should not happen in normal bot op)
            pass
        logger.info(f"Privacy opt-out set for user {user_id}: {opt_out}")

    def is_privacy_opted_out(self, user_id: str) -> bool:
        """Check if user has opted out of context tracking."""
        return self.privacy_opt_outs.get(str(user_id), False)

    async def append_message(self, message: discord.Message, role: str = "user") -> None:
        """Append a message to context storage.

        Args:
            message: Discord message object
            role: 'user' or 'bot'

        """
        if self.is_privacy_opted_out(str(message.author.id)):
            logger.debug(f"Skipping context storage for opted-out user {message.author.id}")
            return

        async with self._memory_lock:
            context_key = self._get_context_key(message)

            # Truncate content to per-message char limit
            raw_content = message.content or ""
            if len(raw_content) > self.max_chars_per_message:
                raw_content = raw_content[: self.max_chars_per_message]

            # Create message entry
            entry = MessageEntry(
                user_id=str(message.author.id),
                channel_id=str(message.channel.id),
                thread_id=str(message.channel.id) if hasattr(message.channel, "parent") and message.channel.parent else None,
                timestamp=message.created_at.isoformat(),
                role=role,
                content=self._encrypt_content(raw_content),
                guild_id=str(message.guild.id) if message.guild else None,
            )

            # Add to memory
            if context_key not in self.memory:
                self.memory[context_key] = []

            self.memory[context_key].append(entry)

            # Trim to history window
            if len(self.memory[context_key]) > self.history_window:
                self.memory[context_key] = self.memory[context_key][-self.history_window :]

            # Mark dirty instead of writing to disk on every message/reply [PA].
            # A periodic task (bot/tasks.py context_autosave) and shutdown both
            # call flush_if_dirty(); in-process reads always hit self.memory
            # directly, so there's no read-consistency gap -- only a bounded
            # window of at-most-N-seconds of context that could be lost on an
            # unclean crash (same tradeoff already accepted for profile autosave).
            if not self.in_memory_only:
                self._dirty = True

            logger.debug(f"✔ Message stored [context={context_key}, role={role}, user={message.author.id}]")

    async def flush_if_dirty(self) -> bool:
        """Persist to disk if there are unsaved changes; no-op otherwise.

        Returns:
            True if a save was performed, False if there was nothing to flush.

        """
        if not self._dirty or self.in_memory_only:
            return False
        async with self._memory_lock:
            if not self._dirty:  # re-check under lock (a flush may have raced in)
                return False
            self._dirty = False
            await self._save()
        return True

    def _estimate_token_count(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 characters)."""
        return len(text) // 4

    def _get_recent_participants(self, context_key: str, lookback_messages: int = 20) -> set[str]:
        """Get users who recently participated in a conversation."""
        if context_key not in self.memory:
            return set()

        recent_messages = self.memory[context_key][-lookback_messages:]
        return {entry.user_id for entry in recent_messages}

    def get_context_for_user(self, message: discord.Message, include_cross_user: bool = True) -> list[MessageEntry]:
        """Get conversation context for a specific user, optionally including cross-user context.

        Args:
            message: Discord message to get context for
            include_cross_user: Whether to include other users' messages in shared channels

        Returns:
            List of MessageEntry objects

        """
        context_key = self._get_context_key(message)

        if context_key not in self.memory:
            return []

        # Get all messages for this context
        all_messages = self.memory[context_key]

        if not include_cross_user or isinstance(message.channel, discord.DMChannel):
            # DM or user requested no cross-user context
            user_messages = [entry for entry in all_messages if entry.user_id == str(message.author.id) or entry.role == "bot"]
            return user_messages[-self.history_window :]

        # For shared channels/threads, include recent cross-user context
        recent_participants = self._get_recent_participants(context_key)

        # Filter messages from recent participants and bot
        relevant_messages = [entry for entry in all_messages if (entry.user_id in recent_participants or entry.role == "bot") and not self.is_privacy_opted_out(entry.user_id)]

        return relevant_messages[-self.history_window :]

    def format_context_string(self, entries: list[MessageEntry], max_tokens: int | None = None) -> str:
        """Format context entries into a conversation string.

        Args:
            entries: List of MessageEntry objects
            max_tokens: Maximum tokens to include (uses self.max_token_limit if None)

        Returns:
            Formatted context string

        """
        if not entries:
            return ""

        max_tokens = max_tokens or self.max_token_limit
        lines = []
        total_tokens = 0
        total_chars = 0

        # Decrypt each entry exactly once, then reuse the plaintext for both
        # continuation filtering and formatting below. [PA]
        decrypted_entries = [(entry, self._decrypt_content(entry.content)) for entry in entries]

        # Filter continuation chunks
        filtered = []
        for entry, content in decrypted_entries:
            if self.ignore_continuation_chunks:
                stripped = content.strip()
                if stripped in ("...", "…", "more", "**(continues)**", "(more)", "more..."):
                    continue
            filtered.append((entry, content))

        # Process entries in reverse to prioritize recent messages
        for entry, content in reversed(filtered):
            try:
                # content already decrypted once above; reuse it

                # Format line
                if entry.role == "bot":
                    line = f"[bot]: {content}"
                else:
                    # Try to get username, fallback to user ID
                    try:
                        user = self.bot.get_user(int(entry.user_id))
                        username = user.display_name if user else f"User({entry.user_id})"
                    except (ValueError, AttributeError, TypeError):
                        username = f"User({entry.user_id})"

                    line = f"[{username}]: {content}"

                # Check token limit
                line_tokens = self._estimate_token_count(line)
                if total_tokens + line_tokens > max_tokens and lines:
                    logger.debug(f"Context truncated at {total_tokens} tokens")
                    break

                # Check total char limit
                line_chars = len(line)
                if total_chars + line_chars > self.max_total_chars and lines:
                    logger.debug(f"Context truncated at {total_chars} total chars")
                    break

                lines.append(line)
                total_tokens += line_tokens
                total_chars += line_chars

            except Exception as e:
                logger.exception(f"❌ Error formatting context entry: {e}")
                continue

        # Reverse back to chronological order
        return "\n".join(reversed(lines))

    async def get_contextual_response(
        self,
        message: discord.Message,
        response_text: str,
        include_cross_user: bool = True,
    ) -> ContextResponse:
        """Generate a contextual response with conversation history.

        Args:
            message: Discord message being responded to
            response_text: The bot's response text
            include_cross_user: Whether to include cross-user context

        Returns:
            ContextResponse with formatted response and metadata

        """
        fallback = False
        used_history = []

        try:
            # Get conversation context
            context_entries = self.get_context_for_user(message, include_cross_user)

            if context_entries:
                # Format context string
                context_str = self.format_context_string(context_entries)

                # Prepare used_history for response (redact if privacy opted out)
                used_history = []
                for entry in context_entries:
                    if not self.is_privacy_opted_out(entry.user_id):
                        # Decrypt once, reuse for the length check and preview slice. [PA]
                        decrypted = self._decrypt_content(entry.content)
                        preview = decrypted[:100] + "..." if len(decrypted) > 100 else decrypted
                        history_entry = {
                            "user_id": entry.user_id,
                            "role": entry.role,
                            "timestamp": entry.timestamp,
                            "content_preview": preview,
                        }
                        used_history.append(history_entry)

                logger.debug(f"✔ Context retrieved [entries={len(context_entries)}, tokens≈{self._estimate_token_count(context_str)}]")

        except Exception as e:
            logger.exception(f"❌ Context retrieval failed: {e}")
            fallback = True
            response_text = f"Got you — y'all wild. {response_text}"

        return ContextResponse(response_text=response_text, used_history=used_history, fallback=fallback)

    async def reset_context(self, message: discord.Message) -> None:
        """Reset conversation context for a channel/DM."""
        context_key = self._get_context_key(message)
        if context_key in self.memory:
            del self.memory[context_key]
            await self._save()
            logger.info(f"Context reset for {context_key}")

    def get_stats(self) -> dict[str, Any]:
        """Get context manager statistics."""
        total_messages = sum(len(entries) for entries in self.memory.values())
        return {
            "total_contexts": len(self.memory),
            "total_messages": total_messages,
            "privacy_opt_outs": len(self.privacy_opt_outs),
            "history_window": self.history_window,
            "encryption_enabled": True,
            "in_memory_only": self.in_memory_only,
        }
