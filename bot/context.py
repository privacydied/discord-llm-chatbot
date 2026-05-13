"""
Conversation context management for tracking message history and state.
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import discord
from discord import Message

# Import config
from .config import load_config


def _load_int_config(key: str, default: int) -> int:
    """Load an integer config value respecting LOW_RESOURCE_MODE via load_config()."""
    try:
        cfg = load_config()
        val = cfg.get(key)
        if val is not None:
            return int(val)
    except Exception:
        pass
    return default


def _get_config_value(key: str, default: Any) -> Any:
    """Read a config value fresh on each call (avoids stale module-level cache)."""
    try:
        return load_config().get(key, default)
    except Exception:
        return default


# Conversation store for tracking message history
conversation_store = defaultdict(list)
CONTEXT_MAXLEN = _load_int_config("CONTEXT_MAX_MESSAGES", 30)
# Context trimming limits — respect LOW_RESOURCE_MODE via load_config()
CONTEXT_MAX_CHARS_PER_MESSAGE = _load_int_config("CONTEXT_MAX_CHARS_PER_MESSAGE", 2000)
CONTEXT_MAX_TOTAL_CHARS = _load_int_config("CONTEXT_MAX_TOTAL_CHARS", 8000)
last_message_time = {}


def context_key(message: Message) -> str:
    """Generate a key for storing conversation context."""
    if isinstance(message.channel, discord.DMChannel):
        return f"dm_{message.author.id}"
    return f"guild_{message.guild.id}_channel_{message.channel.id}"


def reset_context(message: Message, config=None):
    """Reset the conversation context for a channel or DM."""
    key = context_key(message)
    conversation_store[key] = []

    # Update last message time to now to prevent immediate reset
    last_message_time[key] = time.time()


def get_context(message: Message) -> List[Dict[str, Any]]:
    """Get the conversation context for a message."""
    key = context_key(message)
    return conversation_store.get(key, []).copy()


def update_last_active(message: Message):
    """Update the last active time for a conversation."""
    key = context_key(message)
    last_message_time[key] = time.time()


def should_reset_context(message: Message) -> bool:
    """
    Determine if the conversation context should be reset.

    Context is reset if:
    1. It's been more than CONTEXT_RESET_AFTER seconds since the last message
    2. The message starts with '!reset' (handled by command handler)
    """
    key = context_key(message)
    last_time = last_message_time.get(key, 0)

    # If it's been too long since the last message, reset context
    if time.time() - last_time > _get_config_value("CONTEXT_RESET_AFTER", 3600):
        return True

    return False


def add_to_context(message: Message, role: str, content: str, **kwargs):
    """Add a message to the conversation context."""
    key = context_key(message)

    # Clean up old messages
    current_time = time.time()
    conversation_store[key] = [msg for msg in conversation_store.get(key, []) if current_time - msg.get("timestamp", 0) <= _get_config_value("CONTEXT_TTL", 900)]

    # Truncate content to per-message char limit
    if len(content) > CONTEXT_MAX_CHARS_PER_MESSAGE:
        content = content[:CONTEXT_MAX_CHARS_PER_MESSAGE]

    # Add new message
    message_data = {
        "role": role,
        "content": content,
        "timestamp": current_time,
        **kwargs,
    }

    conversation_store[key].append(message_data)

    # Enforce total chars cap: drop oldest messages until under limit
    while len(conversation_store[key]) > 0:
        total = sum(len(m.get("content", "")) for m in conversation_store[key])
        if total <= CONTEXT_MAX_TOTAL_CHARS:
            break
        conversation_store[key] = conversation_store[key][1:]

    # Trim to max length
    if len(conversation_store[key]) > CONTEXT_MAXLEN:
        conversation_store[key] = conversation_store[key][-CONTEXT_MAXLEN:]

    # Update last message time
    update_last_active(message)

    return message_data


def get_conversation_history(message: Message, max_messages: int = 10) -> List[Dict[str, Any]]:
    """Get recent conversation history for a channel or DM."""
    key = context_key(message)
    return conversation_store.get(key, [])[-max_messages:]


def get_conversation_context(user_id: str, guild_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get conversation context by user_id and guild_id."""
    # Generate context key based on user_id and guild_id
    if guild_id:
        # For guild channels, we need to determine the specific channel
        # For now, use a general guild-user key pattern
        key = f"guild_{guild_id}_user_{user_id}"
    else:
        # For DM channels
        key = f"dm_{user_id}"

    # Get the conversation history
    context = conversation_store.get(key, [])

    # Clean up old messages based on TTL
    current_time = time.time()
    context = [msg for msg in context if current_time - msg.get("timestamp", 0) <= _get_config_value("CONTEXT_TTL", 900)]

    # Truncate each message to per-message char limit
    for msg in context:
        c = msg.get("content", "")
        if len(c) > CONTEXT_MAX_CHARS_PER_MESSAGE:
            msg["content"] = c[:CONTEXT_MAX_CHARS_PER_MESSAGE]

    # Update the store with cleaned context
    conversation_store[key] = context

    return context.copy()


def get_last_user_message(key: str) -> Optional[Dict[str, Any]]:
    """Get the last user message in a conversation."""
    messages = conversation_store.get(key, [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None
