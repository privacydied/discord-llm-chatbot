"""Conservative memory ingestion and retrieval guards."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class MemoryCategory(str, Enum):
    recurring_instruction = "recurring_instruction"
    user_preference = "user_preference"
    project_fact = "project_fact"
    server_fact = "server_fact"
    temporary_context = "temporary_context"
    conversation_decision = "conversation_decision"


class MemoryScope(str, Enum):
    user = "user"
    guild = "guild"
    channel = "channel"
    thread = "thread"


class MemoryDecision:
    __slots__ = (
        "allowed",
        "reason",
        "category",
        "confidence",
        "scope",
        "source_message_id",
    )

    def __init__(
        self,
        *,
        allowed: bool,
        reason: str,
        category: MemoryCategory | None = None,
        confidence: float = 0.0,
        scope: MemoryScope | None = None,
        source_message_id: str | None = None,
    ) -> None:
        self.allowed = allowed
        self.reason = reason
        self.category = category
        self.confidence = confidence
        self.scope = scope
        self.source_message_id = source_message_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "category": self.category.value if self.category else None,
            "confidence": self.confidence,
            "scope": self.scope.value if self.scope else None,
            "source_message_id": self.source_message_id,
        }


@dataclass
class MemoryIngestionContext:
    source_user_id: str | None = None
    source_channel_id: str | None = None
    source_guild_id: str | None = None
    source_message_id: str | None = None
    is_explicit_command: bool = False
    guild_only: bool = False
    raw_text: str | None = None


_FUTURE_DIRECTIVE_TOKENS = (
    r"from now on\b.*",
    r"going forward\b.*",
    r"henceforth\b.*",
    r"remember that\b.*",
    r"use this for future\b.*",
    r"call this\b.*",
    r"for future use\b.*",
)
_QUOTED_OR_EXTERNAL_TOKENS = (
    r"^>",
    r"^```",
    r"quoted from",
    r"forwarded from",
    r"email body",
    r"document body",
    r"ocr",
    r"transcribed from",
)
_DENIED_INFERRED_CONTENT_FRAGMENTS = (
    r"\bA/B\b",
    r"\bMODE:\b",
    r"\bdiagnostic\b",
    r"\bbait\b",
    r"\bjoke\b",
    r"\blol\b",
    r"\bstfu\b",
    r"\bidiot\b",
)
_DIAGNOSTIC_ARTIFACT_PATTERNS = (
    re.compile(r"^[A-Z]/[A-Z](/[A-Z])*$", re.MULTILINE),
    re.compile(r"^(AUTO|MANUAL|MODE):", re.MULTILINE),
    re.compile(r"\bstats:\b", re.IGNORECASE),
)


def _compile_patterns(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


_FUTURE_DIRECTIVE_PATTERNS = _compile_patterns(_FUTURE_DIRECTIVE_TOKENS)
_QUOTED_OR_EXTERNAL_PATTERNS = _compile_patterns(_QUOTED_OR_EXTERNAL_TOKENS)
_DENIED_FRAGMENT_PATTERNS = _compile_patterns(_DENIED_INFERRED_CONTENT_FRAGMENTS)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _classify_category(text: str, context: MemoryIngestionContext) -> MemoryCategory | None:
    lowered = text.lower()
    if _matches_any(lowered, _FUTURE_DIRECTIVE_PATTERNS):
        return MemoryCategory.recurring_instruction
    if lowered.startswith("always "):
        return MemoryCategory.recurring_instruction
    server_indicators = (
        "server rule",
        "in this server",
        "in #",
        "this guild",
        "channel rule",
    )
    if context.guild_only and any(indicator in lowered for indicator in server_indicators):
        return MemoryCategory.server_fact
    decision_indicators = ("decided to", "we decided", "agreed to", "picked")
    if any(lowered.startswith(indicator) for indicator in decision_indicators):
        return MemoryCategory.conversation_decision
    temporary_indicators = ("only for today", "temporary", "until", "for now", "just this once")
    if any(indicator in lowered for indicator in temporary_indicators):
        return MemoryCategory.temporary_context
    preference_indicators = ("i prefer", "i like", "i want", "i need", "use short", "use terse")
    if any(lowered.startswith(indicator) for indicator in preference_indicators):
        return MemoryCategory.user_preference
    if context.guild_only and any(lowered.startswith(indicator) for indicator in ("server uses", "server runs", "server requires")):
        return MemoryCategory.server_fact
    path_indicators = ("canonical repo", "repo path", "path is", "uses path", "working directory")
    if any(indicator in lowered for indicator in path_indicators):
        return MemoryCategory.project_fact
    return None


def should_auto_store_memory(
    candidate: dict[str, Any],
    context: MemoryIngestionContext,
    *,
    min_confidence: float = 0.8,
    min_importance: float = 0.7,
    min_temporary_importance: float = 0.6,
    max_chars: int = 140,
) -> MemoryDecision:
    text = (context.raw_text or candidate.get("content") or candidate.get("document") or "")
    if not isinstance(text, str) or not text.strip():
        return MemoryDecision(allowed=False, reason="empty_text")

    trimmed = text.strip()
    if len(trimmed) > max_chars:
        return MemoryDecision(allowed=False, reason="too_long")

    if any(pattern.search(trimmed) for pattern in _DIAGNOSTIC_ARTIFACT_PATTERNS):
        return MemoryDecision(allowed=False, reason="diagnostic_artifact")

    if _matches_any(trimmed, _DENIED_FRAGMENT_PATTERNS):
        return MemoryDecision(allowed=False, reason="denied_inferred_content")

    if _matches_any(trimmed, _QUOTED_OR_EXTERNAL_PATTERNS):
        return MemoryDecision(allowed=False, reason="quoted_or_external_content")

    importance = float(candidate.get("importance", 0.0) or 0.0)
    confidence = float(candidate.get("confidence", 0.0) or 0.0)
    category = _classify_category(trimmed, context)
    if category == MemoryCategory.recurring_instruction and not _matches_any(trimmed, _FUTURE_DIRECTIVE_PATTERNS):
            return MemoryDecision(
                allowed=False,
                reason="recurring_instruction_requires_explicit_future_marker",
                category=category,
                confidence=confidence,
            )

    is_temporary = category == MemoryCategory.temporary_context
    threshold = min_temporary_importance if is_temporary else min_importance
    if importance < threshold or confidence < min_confidence:
        return MemoryDecision(
            allowed=False,
            reason="below_threshold",
            category=category,
            confidence=confidence,
        )

    if category == MemoryCategory.recurring_instruction:
        scope = MemoryScope.user
    elif category == MemoryCategory.server_fact:
        if not context.guild_only:
            return MemoryDecision(
                allowed=False,
                reason="server_fact_without_guild_scope",
                category=category,
                confidence=confidence,
            )
        scope = MemoryScope.guild
    elif category in {MemoryCategory.user_preference, MemoryCategory.project_fact}:
        scope = MemoryScope.user
    else:
        scope = MemoryScope.user

    return MemoryDecision(
        allowed=True,
        reason="allowed",
        category=category,
        confidence=confidence,
        scope=scope,
        source_message_id=context.source_message_id,
    )


_MIN_LEXICAL_OVERLAP_TOKENS = 1


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def select_memories_for_prompt(
    user_message: str,
    candidates: Sequence[dict[str, Any]],
    *,
    max_items: int = 3,
    min_relevance: float = 0.7,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    query_tokens = _tokenize(user_message)
    scored: list[tuple[float, dict[str, Any]]] = []

    for item in candidates:
        metadata = item.get("metadata", {}) or {}
        confidence = float(metadata.get("confidence", 0.0))
        source = str(metadata.get("source", ""))
        explicit = source == "explicit_memory_command"
        semantic = float(item.get("semantic_score", 0.0))
        document = str(item.get("document", ""))
        tokens = _tokenize(document)
        overlap = len(query_tokens & tokens) if query_tokens else 0
        lexical = 1.0 if overlap >= _MIN_LEXICAL_OVERLAP_TOKENS else 0.0
        relevance = (0.4 * semantic) + (0.6 * lexical)
        if not explicit and confidence < 0.8:
            continue
        if relevance < min_relevance:
            logger.debug(
                "memory.rejected",
                extra={
                    "event": "memory.rejected",
                    "memory_id": item.get("memory_id"),
                    "reason": "below_relevance_threshold",
                    "relevance": relevance,
                },
            )
            continue
        scored.append((relevance, item))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    selected = [item for _, item in scored[:max_items]]
    return selected

# Compatibility alias for tests expecting earlier internal name
_looks_like_diagnostic = any
