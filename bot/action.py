from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .public_output import sanitize_embed_collection_for_public
from .utils.output_sanitizer import strip_leading_mode_preamble as _strip_leading_mode_preamble

if TYPE_CHECKING:
    from discord import Embed, File


@dataclass
class BotAction:
    content: str = ""
    embeds: list[Embed] = field(default_factory=list)
    files: list[File] = field(default_factory=list)
    audio_path: str | None = None
    error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Apply final safety net to sanitize content before sending to Discord."""
        self.content = self._apply_final_sanitization(self.content)
        self.embeds = sanitize_embed_collection_for_public(self.embeds)

    def _apply_final_sanitization(self, content: str) -> str:
        """Final safety net to remove any chain-of-thought leakage before Discord send.
        This is a belt-and-suspenders approach that should rarely trigger.
        """
        from .public_output import (
            extract_public_reply_text,
            has_reasoning_leakage,
            sanitize_public_text,
        )
        from .utils.logging import get_logger
        from .vl.postprocess import has_reasoning_content, sanitize_model_output

        logger = get_logger("bot.action.safety_net")

        content = _strip_leading_mode_preamble(content)
        if has_reasoning_leakage(content):
            logger.warning("Final safety net triggered - sanitizing content with reasoning leakage")
            return extract_public_reply_text(content)

        # Second layer: legacy VL postprocess
        vl_handled = False
        if has_reasoning_content(content):
            logger.warning("VL safety net triggered - sanitizing content with reasoning patterns")
            try:
                sanitized = sanitize_model_output(content)
                if len(sanitized) < len(content) * 0.8:
                    logger.info("VL safety net removed %d chars", len(content) - len(sanitized))
                    vl_handled = True
                    content = sanitized
            except Exception as e:
                logger.exception(f"VL sanitization failed: {e}")

        if not vl_handled:
            # Third layer: strip internal labels, aggregation markers, etc.
            content = sanitize_public_text(content)

        return content

    @property
    def has_payload(self):
        return bool(self.content or self.embeds or self.files or self.audio_path)

    # Backward compatibility:
    @property
    def text(self):
        return self.content

    @text.setter
    def text(self, v) -> None:
        self.content = v

    @property
    def embed(self):
        return self.embeds[0] if self.embeds else None

    @property
    def file(self):
        return self.files[0] if self.files else None


# Transitional alias for compatibility
ResponseMessage = BotAction
