"""
Enhanced contextual conversation manager for Discord bot.
Handles multi-user conversation tracking across threads and DMs with privacy controls.
"""

import json
import os
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet

import discord

from bot.util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MessageEntry:
    """Represents a stored message with all required metadata."""

    user_id: str
    channel_id: str
    thread_id: Optional[str]
    timestamp: str
    role: str  # 'user' or 'bot'
    content: str
    guild_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageEntry":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ContextResponse:
    """Response envelope for context retrieval."""

    response_text: str
    used_history: List[Dict[str, Any]]
    fallback: bool


class EnhancedContextManager:
    """
    Enhanced contextual conversation manager that tracks multi-user conversations
    across threads and DMs with privacy controls and encryption.
    """

    def __init__(
        self,
        bot: discord.Client,
        filepath: str = "runtime/enhanced_context.json",
        history_window: Optional[int] = None,
        max_token_limit: int = 4000,
        encryption_key: Optional[bytes] = None,
    ):
        """
        Initialize the enhanced context manager.

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
        self.in_memory_only = (
            os.getenv("IN_MEMORY_CONTEXT_ONLY", "false").lower() == "true"
        )

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

        # Storage: {context_key: List[MessageEntry]}
        self.memory: Dict[str, List[MessageEntry]] = {}

        # Privacy opt-out tracking: {user_id: bool}
        self.privacy_opt_outs: Dict[str, bool] = {}

        # Concurrency protection for thread-safe operations
        self._memory_lock = asyncio.Lock()

        self._load()
        logger.info(
            f"✔ Enhanced context manager initialized [history_window={self.history_window}, encryption=enabled]"
        )

    def _get_context_key(self, message: discord.Message) -> str:
        """Generate context key for message storage."""
        if isinstance(message.channel, discord.DMChannel):
            return f"dm_{message.author.id}"
        elif hasattr(message.channel, "parent") and message.channel.parent:
            # Thread in a channel
            return f"guild_{message.guild.id}_thread_{message.channel.id}"
        else:
            # Regular channel
            return f"guild_{message.guild.id}_channel_{message.channel.id}"

    def _get_user_context_key(
        self, user_id: str, channel_id: str, guild_id: Optional[str] = None
    ) -> str:
        """Generate user-specific context key."""
        if guild_id:
            return f"guild_{guild_id}_channel_{channel_id}_user_{user_id}"
        else:
            return f"dm_{user_id}"

    def _encrypt_content(self, content: str) -> str:
        """Encrypt message content."""
        try:
            return self.cipher.encrypt(content.encode()).decode()
        except Exception as e:
            logger.error(f"❌ Encryption failed: {e}")
            return content  # Fallback to unencrypted

    def _decrypt_content(self, encrypted_content: str) -> str:
        """Decrypt message content."""
        try:
            return self.cipher.decrypt(encrypted_content.encode()).decode()
        except Exception as e:
            logger.debug(f"Decryption failed, assuming unencrypted: {e}")
            return encrypted_content  # Assume it's unencrypted

    def _load(self) -> None:
        """Load context from storage file."""
        if self.in_memory_only:
            logger.info("Context storage is in-memory only")
            return

        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Load message entries
                for key, entries in data.get("messages", {}).items():
                    self.memory[key] = [
                        MessageEntry.from_dict(entry) for entry in entries
                    ]

                # Load privacy opt-outs
                self.privacy_opt_outs = data.get("privacy_opt_outs", {})

                logger.info(f"✔ Loaded enhanced context from {self.filepath}")
            else:
                logger.info(
                    f"Context file not found at {self.filepath}, starting fresh"
                )
        except Exception as e:
            logger.error(f"❌ Failed to load context: {e}")
            self.memory = {}
            self.privacy_opt_outs = {}

    def _save(self) -> None:
        """Save context to storage file with privacy filtering."""
        if self.in_memory_only:
            return

        try:
            # Filter out DM conversations for privacy
            filtered_messages = {
                k: [entry.to_dict() for entry in v]
                for k, v in self.memory.items()
                if not k.startswith("dm_")
            }

            data = {
                "messages": filtered_messages,
                "privacy_opt_outs": self.privacy_opt_outs,
                "metadata": {
                    "version": "1.0",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
            }

            # Ensure directory exists
            os.makedirs(
                os.path.dirname(self.filepath)
                if os.path.dirname(self.filepath)
                else ".",
                exist_ok=True,
            )

            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Set restrictive permissions
            os.chmod(self.filepath, 0o600)

        except Exception as e:
            logger.error(f"❌ Failed to save context: {e}")

    def set_privacy_opt_out(self, user_id: str, opt_out: bool = True) -> None:
        """Set privacy opt-out for a user."""
        self.privacy_opt_outs[str(user_id)] = opt_out
        self._save()
        logger.info(f"✔ Privacy opt-out set for user {user_id}: {opt_out}")

    def is_privacy_opted_out(self, user_id: str) -> bool:
        """Check if user has opted out of context tracking."""
        return self.privacy_opt_outs.get(str(user_id), False)

    async def append_message(self, message: discord.Message, role: str = "user"):
        """
        Append a message to context storage.

        Args:
            message: Discord message object
            role: 'user' or 'bot'
        """
        if self.is_privacy_opted_out(str(message.author.id)):
            logger.debug(
                f"Skipping context storage for opted-out user {message.author.id}"
            )
            return

        async with self._memory_lock:
            context_key = self._get_context_key(message)

            # Create message entry
            entry = MessageEntry(
                user_id=str(message.author.id),
                channel_id=str(message.channel.id),
                thread_id=str(message.channel.id)
                if hasattr(message.channel, "parent") and message.channel.parent
                else None,
                timestamp=message.created_at.isoformat(),
                role=role,
                content=self._encrypt_content(message.content or ""),
                guild_id=str(message.guild.id) if message.guild else None,
            )

            # Add to memory
            if context_key not in self.memory:
                self.memory[context_key] = []

            self.memory[context_key].append(entry)

            # Trim to history window
            if len(self.memory[context_key]) > self.history_window:
                self.memory[context_key] = self.memory[context_key][
                    -self.history_window :
                ]

            # Save to disk if not in memory-only mode
            if not self.in_memory_only:
                self._save()

            logger.debug(
                f"✔ Message stored [context={context_key}, role={role}, user={message.author.id}]"
            )

    def _estimate_token_count(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 characters)."""
        return len(text) // 4

    def _get_recent_participants(
        self, context_key: str, lookback_messages: int = 20
    ) -> Set[str]:
        """Get users who recently participated in a conversation."""
        if context_key not in self.memory:
            return set()

        recent_messages = self.memory[context_key][-lookback_messages:]
        return {entry.user_id for entry in recent_messages}

    def get_context_for_user(
        self, message: discord.Message, include_cross_user: bool = True
    ) -> List[MessageEntry]:
        """
        Get conversation context for a specific user, optionally including cross-user context.

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
            user_messages = [
                entry
                for entry in all_messages
                if entry.user_id == str(message.author.id) or entry.role == "bot"
            ]
            return user_messages[-self.history_window :]

        # For shared channels/threads, include recent cross-user context
        recent_participants = self._get_recent_participants(context_key)

        # Filter messages from recent participants and bot
        relevant_messages = [
            entry
            for entry in all_messages
            if (entry.user_id in recent_participants or entry.role == "bot")
            and not self.is_privacy_opted_out(entry.user_id)
        ]

        return relevant_messages[-self.history_window :]

    def format_context_string(
        self, entries: List[MessageEntry], max_tokens: Optional[int] = None
    ) -> str:
        """
        Format context entries into a conversation string.

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

        # Process entries in reverse to prioritize recent messages
        for entry in reversed(entries):
            try:
                # Decrypt content
                content = self._decrypt_content(entry.content)

                # Format line
                if entry.role == "bot":
                    line = f"[bot]: {content}"
                else:
                    # Try to get username, fallback to user ID
                    try:
                        user = self.bot.get_user(int(entry.user_id))
                        username = (
                            user.display_name if user else f"User({entry.user_id})"
                        )
                    except Exception:
                        username = f"User({entry.user_id})"

                    line = f"[{username}]: {content}"

                # Check token limit
                line_tokens = self._estimate_token_count(line)
                if total_tokens + line_tokens > max_tokens and lines:
                    logger.debug(f"Context truncated at {total_tokens} tokens")
                    break

                lines.append(line)
                total_tokens += line_tokens

            except Exception as e:
                logger.error(f"❌ Error formatting context entry: {e}")
                continue

        # Reverse back to chronological order
        return "\n".join(reversed(lines))

    async def get_contextual_response(
        self,
        message: discord.Message,
        response_text: str,
        include_cross_user: bool = True,
    ) -> ContextResponse:
        """
        Generate a contextual response with conversation history.

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
                        history_entry = {
                            "user_id": entry.user_id,
                            "role": entry.role,
                            "timestamp": entry.timestamp,
                            "content_preview": self._decrypt_content(entry.content)[
                                :100
                            ]
                            + "..."
                            if len(self._decrypt_content(entry.content)) > 100
                            else self._decrypt_content(entry.content),
                        }
                        used_history.append(history_entry)

                logger.debug(
                    f"✔ Context retrieved [entries={len(context_entries)}, tokens≈{self._estimate_token_count(context_str)}]"
                )

        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}")
            fallback = True
            response_text = f"Got you — y'all wild. {response_text}"

        return ContextResponse(
            response_text=response_text, used_history=used_history, fallback=fallback
        )

    def reset_context(self, message: discord.Message) -> None:
        """Reset conversation context for a channel/DM."""
        context_key = self._get_context_key(message)
        if context_key in self.memory:
            del self.memory[context_key]
            self._save()
            logger.info(f"✔ Context reset for {context_key}")

    def get_stats(self) -> Dict[str, Any]:
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
