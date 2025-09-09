from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TYPE_CHECKING

# Re-export InputModality from the dedicated modality module to maintain
# backward compatibility for legacy imports (e.g., tests importing from bot.types)
from .modality import InputModality

if TYPE_CHECKING:
    pass


class Command(Enum):
    """Enumeration of all supported bot commands."""

    CHAT = auto()  # General chat, default command
    PING = auto()  # A simple command for testing
    HELP = auto()  # Show help message
    SEARCH = auto()  # Online web search
    TTS = auto()  # Toggle TTS on/off for the user
    TTS_ALL = auto()  # Admin-only global TTS toggle
    SPEAK = auto()  # Single TTS response then revert to text
    SAY = auto()  # Say a message in TTS
    MEMORY_ADD = auto()  # Add a memory
    MEMORY_DEL = auto()  # Delete a memory
    MEMORY_SHOW = auto()  # Show memories
    MEMORY_WIPE = auto()  # Wipe memories
    RAG = auto()  # RAG system commands
    RAG_BOOTSTRAP = auto()  # Bootstrap RAG knowledge base
    RAG_SEARCH = auto()  # Search RAG knowledge base
    RAG_STATUS = auto()  # Show RAG system status
    ALERT = auto()  # Admin DM alert system
    IMG = auto()  # Image generation command
    IGNORE = auto()  # A command to signify that the message should be ignored


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

    This is a lightweight shim for legacy imports (tests and older code paths)
    that expect `OutputModality` to be importable from `bot.types`.
    It mirrors the enum defined in `bot.router` without importing it directly,
    to avoid circular imports during module initialization.
    """

    TEXT = auto()
    TTS = auto()


# Explicit export list for clarity and legacy compatibility
__all__ = [
    "Command",
    "ParsedCommand",
    "ResponseMessage",
    "InputModality",
    "OutputModality",
]
