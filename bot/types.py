from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ResponseMessage:
    """Data class to hold response content."""
    text: Optional[str] = None
    audio_path: Optional[Path] = None
