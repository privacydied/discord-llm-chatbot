from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

class Command(Enum):
    """Enumeration of all supported bot commands."""
    CHAT = auto()      # General chat, default command
    PING = auto()      # A simple command for testing
    TTS = auto()
    SAY = auto()
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
