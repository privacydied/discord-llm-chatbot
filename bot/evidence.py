"""Evidence bundle data structures for multimodal provenance stitching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return "\n".join(part.strip() for part in str(value).splitlines() if part.strip()).strip()


def _trim_text(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[: max_chars - 3] + "..."


@dataclass(slots=True)
class EvidenceSection:
    """Single evidence slice with type classification and provenance metadata."""

    kind: str
    title: str
    body: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[str, str, str]:
        return self.kind, self.title, self.body

    def trimmed(self, limit: int) -> "EvidenceSection":  # [CMV]
        return EvidenceSection(
            kind=self.kind,
            title=self.title,
            body=_trim_text(self.body, limit),
            provenance=dict(self.provenance),
        )


@dataclass(slots=True)
class EvidenceBundle:
    """Structured collection of all textual evidence for LLM consumption."""

    source_platform: str = ""
    source_url: str = ""
    primary_tweet_id: Optional[str] = None  # Anchor for deterministic media selection
    selected_tweet_id: Optional[str] = None  # Actual media host tweet id (may equal primary)
    caption_text: str = ""
    quoted_text: str = ""  # For quote tweets and retweets
    media_transcript: str = ""
    media_vision_notes: str = ""
    media_ocr_text: str = ""
    media_alt_text: str = ""
    extra_sections: List[EvidenceSection] = field(default_factory=list)
    # Local flag for telemetry and routing decisions when STT yields no/low speech [REH]
    stt_no_speech: bool = False

    def add_section(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        section = EvidenceSection(
            kind=kind,
            title=title,
            body=_normalize_text(body),
            provenance=provenance or {},
        )
        if section.body:
            self.extra_sections.append(section)

    def merge_text(self, attr: str, value: Optional[str]) -> None:
        if not value:
            return
        normalized = _normalize_text(value)
        if not normalized:
            return
        current = getattr(self, attr)
        if current:
            combined = self._dedupe_concat(current, normalized)
            setattr(self, attr, combined)
        else:
            setattr(self, attr, normalized)

    def _dedupe_concat(self, left: str, right: str) -> str:
        left_norm = left.lower()
        right_norm = right.lower()
        if right_norm in left_norm:
            return left
        if left_norm in right_norm:
            return right
        return f"{left}\n\n{right}"

    def _sections_similar(self, section1: EvidenceSection, section2: EvidenceSection) -> bool:
        """Check if two sections are near-duplicates for deduplication."""
        if section1.kind == section2.kind:
            return False  # Same kind should not be deduped

        # Only dedup different kinds that might overlap (e.g., OCR vs caption)
        text1 = _normalize_text(section1.body)
        text2 = _normalize_text(section2.body)

        # Simple similarity check: if one contains the other or they're very similar
        shorter, longer = (text1, text2) if len(text1) <= len(text2) else (text2, text1)
        if len(shorter) < 10:  # Too short to dedup
            return False

        # Check if shorter is substantially contained in longer
        return shorter in longer and len(shorter) / len(longer) > 0.8

    def get_evidence_sections(self) -> List[EvidenceSection]:
        """Get ordered evidence sections for prompt composition."""
        sections = []

        # Fixed order: CAPTION, QUOTED, TRANSCRIPT/VISION, OCR, ALT
        if self.caption_text.strip():
            sections.append(
                EvidenceSection(
                    kind="caption",
                    title="Tweet Caption",
                    body=self.caption_text.strip(),
                    provenance={"source": "tweet_text"},
                )
            )

        if self.quoted_text.strip():
            sections.append(
                EvidenceSection(
                    kind="quoted",
                    title="Quoted Text",
                    body=self.quoted_text.strip(),
                    provenance={"source": "quoted_tweet"},
                )
            )

        # Choose between transcript and vision (prioritize transcript if present)
        media_body = ""
        media_title = ""
        if self.media_transcript.strip():
            media_body = self.media_transcript.strip()
            media_title = "Audio Transcript"
        elif self.media_vision_notes.strip():
            media_body = self.media_vision_notes.strip()
            media_title = "Visual Analysis"

        if media_body:
            sections.append(
                EvidenceSection(
                    kind="media",
                    title=media_title,
                    body=media_body,
                    provenance={"source": "media_processing"},
                )
            )

        if self.media_ocr_text.strip():
            sections.append(
                EvidenceSection(
                    kind="ocr",
                    title="OCR Text",
                    body=self.media_ocr_text.strip(),
                    provenance={"source": "ocr_extraction"},
                )
            )

        if self.media_alt_text.strip():
            sections.append(
                EvidenceSection(
                    kind="alt",
                    title="Alt Text",
                    body=self.media_alt_text.strip(),
                    provenance={"source": "alt_text"},
                )
            )

        # Add any extra sections
        sections.extend(self.extra_sections)

        # Deduplication: if sections are near-duplicates, keep the longer one
        if len(sections) >= 2:
            deduped = []
            for section in sections:
                # Check against already added sections
                is_duplicate = False
                for existing in deduped:
                    if self._sections_similar(section, existing):
                        # Keep the longer one
                        if len(section.body) > len(existing.body):
                            # Replace existing with current
                            existing.body = section.body
                        is_duplicate = True
                        break
                if not is_duplicate:
                    deduped.append(section)
            sections = deduped

        return sections

    def compose_prompt_text(self, *, token_budget: int = 0, section_limit: int = 0) -> str:
        """
        Compose evidence into final prompt text with token guard and ordering.
        Args:
            token_budget: Maximum total characters (0 for unlimited)
            section_limit: Max sections per kind (0 for unlimited)

        Returns:
            Formatted evidence string for LLM consumption
        """
        sections = self.get_evidence_sections()

        if not sections:
            return ""

        # Apply section limits if specified
        if section_limit > 0:
            filtered = []
            kind_counts = {}
            for sec in sections:
                kind = sec.kind
                if kind not in kind_counts:
                    kind_counts[kind] = 0
                if kind_counts[kind] < section_limit:
                    filtered.append(sec)
                    kind_counts[kind] += 1
            sections = filtered

        # Build text with token guard
        parts = []
        total_chars = 0

        for section in sections:
            section_text = f"[{section.title}]\n{section.body}\n"
            section_chars = len(section_text)

            # Check if adding this section would exceed budget
            if token_budget > 0 and total_chars + section_chars > token_budget:
                if parts:  # Don't trim if this is the first section
                    break
                # If only section, trim it
                section = section.trimmed(token_budget)
                section_text = f"[{section.title}]\n{section.body}\n"
                parts.append(section_text)
                break

            parts.append(section_text)
            total_chars += section_chars

        composed = "\n".join(parts).strip()

        # Logging [CDiP]
        try:
            from .utils.logging import get_logger

            logger = get_logger(__name__)
            kept_sections = [sec.kind for sec in sections[: len(parts)]]
            # Per-section length snapshot for telemetry
            lens = {
                "caption": len(self.caption_text or ""),
                "quoted": len(self.quoted_text or ""),
                "transcript": len(self.media_transcript or ""),
                "vision": len(self.media_vision_notes or ""),
                "ocr": len(self.media_ocr_text or ""),
                "alt": len(self.media_alt_text or ""),
            }
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
                        "total_sections": len(sections),
                        "token_budget": token_budget,
                        "section_limit": section_limit,
                    },
                },
            )
        except Exception:
            pass

        return composed

    def has_evidence(self) -> bool:
        """Check if bundle contains any evidence."""
        return bool(self.caption_text or self.quoted_text or self.media_transcript or self.media_alt_text or self.media_ocr_text or self.media_vision_notes or self.extra_sections)
