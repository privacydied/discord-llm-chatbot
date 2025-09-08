"""
Compatibility package for tests expecting `bot.types` to be a module that
provides Command/ParsedCommand/ResponseMessage/InputModality/OutputModality
and a `money` submodule exposing Money.

Prefer importing canonical symbols directly in production code, but keep this
package to avoid breaking legacy tests. [CA][REH]
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

# Re-export InputModality from the dedicated modality module to maintain
# backward compatibility for legacy imports (e.g., tests importing from bot.types)
from ..modality import InputModality  # noqa: F401


class Command(Enum):
    """Enumeration of all supported bot commands."""
    CHAT = auto()       # General chat, default command
    PING = auto()       # A simple command for testing
    HELP = auto()       # Show help message
    SEARCH = auto()     # Online web search
    TTS = auto()        # Toggle TTS on/off for the user
    TTS_ALL = auto()    # Admin-only global TTS toggle
    SPEAK = auto()      # Single TTS response then revert to text
    SAY = auto()        # Say a message in TTS
    MEMORY_ADD = auto() # Add a memory
    MEMORY_DEL = auto() # Delete a memory
    MEMORY_SHOW = auto()# Show memories
    MEMORY_WIPE = auto()# Wipe memories
    RAG = auto()        # RAG system commands
    RAG_BOOTSTRAP = auto() # Bootstrap RAG knowledge base
    RAG_SEARCH = auto() # Search RAG knowledge base
    RAG_STATUS = auto() # Show RAG system status
    ALERT = auto()      # Admin DM alert system
    IMG = auto()        # Image generation command
    IGNORE = auto()     # A command to signify that the message should be ignored


@dataclass
class ParsedCommand:
    """Represents a parsed command with its type and cleaned content."""
    command: Command
    cleaned_content: str


@dataclass
class ResponseMessage:
    """Data class to hold response content."""
    text: Optional[str] = None
    audio_path: Optional[Path] = None


class OutputModality(Enum):
    """Defines the type of output the bot should produce.

    This mirrors the enum used for router output while avoiding circular imports
    during module initialization. [REH]
    """
    TEXT = auto()
    TTS = auto()


__all__ = [
    "Command",
    "ParsedCommand",
    "ResponseMessage",
    "InputModality",
    "OutputModality",
    "money",
]
