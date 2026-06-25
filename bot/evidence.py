"""Evidence bundling for context-aware replies.

Collects and formats all available evidence (text, media, transcripts) into a
coherent context block for prompt injection. Includes breadcrumb logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum characters per source in composed context
MAX_SOURCE_CHARS = 2000

# Global token budget for composed context
DEFAULT_TOKEN_BUDGET = 3500

# Maximum sections to include before truncation
MAX_SECTIONS = 7


@dataclass
class EvidenceBundle:
    """Self-contained evidence block for prompt injection."""

    # Text sources
    caption_text: str | None = None
    quoted_text: str | None = None

    # Media-derived sources
    media_transcript: str | None = None
    media_vision_notes: str | None = None
    media_alt_text: str | None = None
    media_ocr_text: str | None = None

    # Linkbacks for traceability
    primary_tweet_id: str | None = None
    selected_tweet_id: str | None = None

    # Additional context inserted at runtime
    extra_sections: dict[str, str] = field(default_factory=dict)

    # Runtime flags
    stt_no_speech: bool = False

    def compose(
        self,
        token_budget: int | None = None,
        max_sections: int = MAX_SECTIONS,
        section_limit: int = MAX_SOURCE_CHARS,
    ) -> str:
        """Compose evidence into a single context block for prompt injection.

        Args:
            token_budget: Maximum tokens (approx chars/4) for the entire block
            max_sections: Maximum number of source sections to include
            section_limit: Character limit per individual section

        Returns:
            Formatted context string ready for prompt injection
        """
        if token_budget is None:
            token_budget = DEFAULT_TOKEN_BUDGET

        # Collect all available sources with their labels
        sources: list[tuple[str, str]] = []

        if self.caption_text:
            sources.append(("caption", self.caption_text[:section_limit]))

        if self.quoted_text:
            sources.append(("quoted text", self.quoted_text[:section_limit]))

        if self.media_transcript:
            sources.append(("transcript", self.media_transcript[:section_limit]))

        if self.media_vision_notes:
            sources.append(("vision notes", self.media_vision_notes[:section_limit]))

        if self.media_alt_text:
            sources.append(("alt text", self.media_alt_text[:section_limit]))

        if self.media_ocr_text:
            sources.append(("OCR text", self.media_ocr_text[:section_limit]))

        for key, val in self.extra_sections.items():
            if val:
                sources.append((key, val[:section_limit]))

        # Build composed text
        parts = []
        for label, text in sources:
            if text:
                parts.append(f"[{label}]\n{text.strip()}")

        # If over token budget, truncate from the end (least important last)
        composed = "\n\n".join(parts)
        max_chars = token_budget * 4  # rough 4 chars per token
        if len(composed) > max_chars:
            composed = composed[:max_chars].rsplit("\n", 1)[0] + "\n... [truncated]"

        # Prepare breadcrumb data
        if sources:
            lengths = []
            for label, text in sources:
                lengths.append(f"{label}:{len(text)}")
            lens = ", ".join(lengths)
            kept_labels = [label for label, _ in sources]
        else:
            lens = "empty"
            kept_labels = []

        # Breadcrumb logging
        logger = logging.getLogger(__name__)
        kept_sections = ", ".join(kept_labels) if kept_labels else "none"
        try:
            logger.info(
                "context.assembled",
                extra={
                    "event": "context.assembled",
                    "detail": {
                        "chars": len(composed),
                        "lengths": lens,
                        "stt_no_speech": bool(getattr(self, "stt_no_speech", False)),
                        "primary": self.primary_tweet_id or "",
                        "selected": (self.selected_tweet_id or self.primary_tweet_id or ""),
                        "sections_kept": kept_sections,
                        "total_sections": len(sources),
                        "token_budget": token_budget,
                        "section_limit": section_limit,
                    },
                },
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Failed to log evidence compose breadcrumb: {e}")

        return composed

    def has_evidence(self) -> bool:
        """Check if bundle contains any evidence."""
        return bool(
            self.caption_text
            or self.quoted_text
            or self.media_transcript
            or self.media_alt_text
            or self.media_ocr_text
            or self.media_vision_notes
            or self.extra_sections
        )
