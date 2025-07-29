from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from discord import Embed, File


@dataclass
class BotAction:
    content: str = ""
    embeds: List[Embed] = field(default_factory=list)
    files: List[File] = field(default_factory=list)
    audio_path: Optional[str] = None
    error: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_payload(self):
        return bool(self.content or self.embeds or self.files or self.audio_path)

    # Backward compatibility:
    @property
    def text(self):
        return self.content

    @text.setter
    def text(self, v):
        self.content = v

    @property
    def embed(self):
        return self.embeds[0] if self.embeds else None

    @property
    def file(self):
        return self.files[0] if self.files else None


# Transitional alias for compatibility
ResponseMessage = BotAction
