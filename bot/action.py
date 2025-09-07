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
    
    def __post_init__(self):
        """Apply final safety net to sanitize content before sending to Discord."""
        if self.content:
            self.content = self._apply_final_sanitization(self.content)
    
    def _apply_final_sanitization(self, content: str) -> str:
        """
        Final safety net to remove any chain-of-thought leakage before Discord send.
        This is a belt-and-suspenders approach that should rarely trigger.
        """
        try:
            from .vl.postprocess import sanitize_model_output, has_reasoning_content
            
            # Only sanitize if content contains reasoning patterns
            if has_reasoning_content(content):
                import os
                from .util.logging import get_logger
                
                logger = get_logger("bot.action.safety_net")
                logger.warning(f"Final safety net triggered - sanitizing content with reasoning leakage")
                
                sanitized = sanitize_model_output(content)
                
                # Log if significant sanitization occurred
                if len(sanitized) < len(content) * 0.8:  # More than 20% removed
                    logger.info(f"Safety net removed {len(content) - len(sanitized)} chars of reasoning content")
                
                return sanitized
            
            return content
            
        except Exception as e:
            # If sanitization fails, return original content (fail-safe)
            try:
                from .util.logging import get_logger
                logger = get_logger("bot.action.safety_net")
                logger.error(f"Final sanitization failed, using original content: {e}")
            except:
                pass
            return content

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
