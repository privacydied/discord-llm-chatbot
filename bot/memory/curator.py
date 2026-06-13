"""Curates which interactions should become durable memories."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

# Pattern constants extracted to a dedicated module to keep this file smaller.
# Importing from curator_patterns also re-exports is_public_safe for callers.
from .curator_patterns import (  # noqa: F401 – re-export for external importers
    _GENERAL_CLAIM_TERMS,
    _INFERRED_DENYLIST,
    _INTERNAL_PATTERNS,
    _PROJECT_FACT_CUES,
    _PROJECT_FACT_SIGNALS,
    _SECRET_PATTERNS,
    _SENSITIVE_ATTRIBUTE_PATTERNS,
    _WORLD_CLAIM_TERMS,
    is_public_safe,
)
from .persistent_store import MemoryRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryCandidate:
    memory_id: str
    user_id: str
    guild_id: str | None
    channel_id: str | None
    thread_id: str | None
    source_message_id: str | None
    context_type: str
    text: str
    summary: str
    importance: float
    confidence: float
    source: str
    expires_at: str | None
    metadata_json: str = "{}"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_accessed_at: str | None = None
    deleted_at: str | None = None
    chroma_id: str | None = None

    def to_record(self) -> MemoryRecord:
        return MemoryRecord(
            memory_id=self.memory_id,
            user_id=self.user_id,
            guild_id=self.guild_id,
            channel_id=self.channel_id,
            thread_id=self.thread_id,
            source_message_id=self.source_message_id,
            context_type=self.context_type,
            text=self.text,
            summary=self.summary,
            importance=self.importance,
            confidence=self.confidence,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_accessed_at=self.last_accessed_at,
            expires_at=self.expires_at,
            source=self.source,
            deleted_at=self.deleted_at,
            chroma_id=self.chroma_id,
            metadata_json=self.metadata_json,
        )


class CuratedMemoryCurator:
    """Heuristic curation for durable memories.

    Design goals:
    - Explicit memories: permissive.
    - Inferred memories: conservative; avoid noisy project/debug chatter.
    """

    def __init__(
        self,
        default_ttl_days: int = 180,
        temp_ttl_days: int = 14,
        min_importance: float = 0.55,
    ) -> None:
        self.default_ttl_days = int(default_ttl_days)
        self.temp_ttl_days = int(temp_ttl_days)
        self.min_importance = float(min_importance)

    # ---------------- explicit (user asked to remember) ----------------

    def build_explicit_candidate(
        self,
        *,
        user_id: str,
        text: str,
        guild_id: str | None = None,
        channel_id: str | None = None,
        thread_id: str | None = None,
        source_message_id: str | None = None,
        context_type: str = "user_preference",
        source: str = "explicit_memory_command",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate | None:
        text = self._normalize(text)
        if not text:
            return None
        if len(text) > 2000:
            return None  # too long
        if self._looks_sensitive(text):
            return None
        if self._looks_internal(text):
            return None

        summary = self._summarize(text, context_type)
        importance = max(self.min_importance, 0.9 if source == "explicit_memory_command" else 0.75)
        confidence = 0.96 if source == "explicit_memory_command" else 0.9
        expires_at = self._expiration_for(context_type)

        payload = metadata or {}
        return MemoryCandidate(
            memory_id=str(uuid4()),
            user_id=str(user_id),
            guild_id=str(guild_id) if guild_id is not None else None,
            channel_id=str(channel_id) if channel_id is not None else None,
            thread_id=str(thread_id) if thread_id is not None else None,
            source_message_id=str(source_message_id) if source_message_id is not None else None,
            context_type=context_type,
            text=text,
            summary=summary,
            importance=importance,
            confidence=confidence,
            source=source,
            expires_at=expires_at,
            metadata_json=self._metadata_json(payload),
        )

    # ---------------- inferred (auto from conversation) ----------------

    def curate_inferred_candidate(
        self,
        *,
        user_id: str,
        text: str,
        guild_id: str | None = None,
        channel_id: str | None = None,
        thread_id: str | None = None,
        source_message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate | None:
        text = self._normalize(text)
        if not text:
            return self._log_inferred_decision(
                user_id,
                guild_id,
                channel_id,
                thread_id,
                source_message_id,
                reason="too_short",
            )
        if len(text) < 16:
            return self._log_inferred_decision(
                user_id,
                guild_id,
                channel_id,
                thread_id,
                source_message_id,
                reason="too_short",
            )

        if self._looks_sensitive(text):
            return self._log_inferred_decision(
                user_id,
                guild_id,
                channel_id,
                thread_id,
                source_message_id,
                reason="sensitive",
            )

        if self._looks_internal(text):
            return self._log_inferred_decision(
                user_id,
                guild_id,
                channel_id,
                thread_id,
                source_message_id,
                reason="internal_noise",
            )

        if self._looks_denied_inferred(text.lower()):
            return self._log_inferred_decision(
                user_id,
                guild_id,
                channel_id,
                thread_id,
                source_message_id,
                reason="denylist_blocked",
            )

        rejection_reason = self._rejection_reason_for_inferred(text)
        if rejection_reason is not None:
            return self._log_inferred_decision(
                user_id,
                guild_id,
                channel_id,
                thread_id,
                source_message_id,
                reason=rejection_reason,
            )

        context_type, is_durable = self._classify_inferred(text)
        if not is_durable:
            return self._log_inferred_decision(
                user_id,
                guild_id,
                channel_id,
                thread_id,
                source_message_id,
                context_type=context_type,
                reason="reject_not_project_fact",
            )

        importance = self._importance_inferred(text, context_type)

        # For temporary_context, enforce stricter threshold
        if context_type == "temporary_context":
            if importance < 0.6:
                return self._log_inferred_decision(
                    user_id,
                    guild_id,
                    channel_id,
                    thread_id,
                    source_message_id,
                    context_type=context_type,
                    reason="weak_temporary_context",
                )
        # Non-temp memories must meet min_importance
        elif importance < self.min_importance:
            return self._log_inferred_decision(
                user_id,
                guild_id,
                channel_id,
                thread_id,
                source_message_id,
                context_type=context_type,
                reason="below_importance_threshold",
            )

        confidence = self._confidence(text, context_type)
        summary = self._summarize(text, context_type)
        expires_at = self._expiration_for(context_type)

        candidate = MemoryCandidate(
            memory_id=str(uuid4()),
            user_id=str(user_id),
            guild_id=str(guild_id) if guild_id is not None else None,
            channel_id=str(channel_id) if channel_id is not None else None,
            thread_id=str(thread_id) if thread_id is not None else None,
            source_message_id=str(source_message_id) if source_message_id is not None else None,
            context_type=context_type,
            text=text,
            summary=summary,
            importance=importance,
            confidence=confidence,
            source="inferred_curated",
            expires_at=expires_at,
            metadata_json=self._metadata_json(metadata or {}),
        )

        self._log_inferred_decision(
            user_id,
            guild_id,
            channel_id,
            thread_id,
            source_message_id,
            context_type=context_type,
            reason="accepted",
            importance=importance,
            confidence=confidence,
            ttl_days=self._ttl_for(context_type),
        )

        return candidate

    # ---------------- internal helpers: normalization & filters ----------------

    def _normalize(self, text: str) -> str:
        from bot.config import load_config as _curator_load_config

        _cc = _curator_load_config()
        max_chars = int(_cc.get("MEMORY_MAX_TEXT_CHARS", 500))
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text.strip())
        return text[: min(2000, max_chars)]

    def _looks_sensitive(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _SECRET_PATTERNS)

    def _looks_internal(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _INTERNAL_PATTERNS)

    @staticmethod
    def _looks_denied_inferred(lower: str) -> bool:
        """Return True when inferred content matches the denylist."""
        return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _INFERRED_DENYLIST)

    def _rejection_reason_for_inferred(self, text: str) -> str | None:
        lower = text.lower()
        if self._looks_quoted_or_external_content(lower):
            return "reject_quoted_or_external_content"
        if self._looks_sensitive_generalization(lower):
            return "reject_sensitive_generalization"
        if self._looks_world_claim(lower):
            return "reject_world_claim"
        return None

    @staticmethod
    def _has_project_anchor(lower: str) -> bool:
        return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _PROJECT_FACT_SIGNALS)

    @staticmethod
    def _has_project_fact_cue(lower: str) -> bool:
        return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _PROJECT_FACT_CUES)

    def _looks_project_scoped_fact(self, lower: str) -> bool:
        if not self._has_project_anchor(lower):
            return False
        return self._has_project_fact_cue(lower)

    @staticmethod
    def _looks_quoted_or_external_content(lower: str) -> bool:
        if any(
            token in lower
            for token in [
                "according to",
                "article",
                "webpage",
                "website",
                "blog",
                "source:",
            ]
        ):
            return True
        if "http://" in lower or "https://" in lower or "www." in lower:
            return True
        return bool(re.search(r"[\"“”][^\"“”]{12,}[\"“”]", lower))

    def _looks_sensitive_generalization(self, lower: str) -> bool:
        if self._has_project_anchor(lower):
            return False
        if not any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _SENSITIVE_ATTRIBUTE_PATTERNS):
            return False
        if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _GENERAL_CLAIM_TERMS):
            return True
        return bool(any(trigger in lower for trigger in [" prefer ", " likes ", " like ", " love ", " loves ", " am ", "'m ", " have ", " is ", " are "]))

    def _looks_world_claim(self, lower: str) -> bool:
        if self._has_project_anchor(lower):
            return False
        if not any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _WORLD_CLAIM_TERMS):
            return False
        return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _GENERAL_CLAIM_TERMS)

    # ---------------- internal helpers: inferred classification ----------------

    def _classify_inferred(self, text: str) -> tuple[str, bool]:
        """Return (context_type, is_durable) for inferred memories.

        - Must be more selective than before.
        - 'today', 'this week', 'for now', 'temporarily' only => temporary_context, not durable.
        - Avoid storing normal project/debug chatter.
        """
        lower = text.lower()

        # 1) Strong user preference patterns
        if self._is_user_preference(lower):
            return "user_preference", True

        # 2) Strong recurring instructions (always/never)
        if self._is_recurring_instruction(lower):
            return "recurring_instruction", True

        # 3) Explicit 'remember that...' with clear object
        if self._is_strong_remember(lower):
            return "recurring_instruction", True

        # 4) Conversation decisions: we decided / the decision is
        if self._is_conversation_decision(lower):
            return "conversation_decision", True

        # 5) Correction of bot behavior or a remembered rule
        if self._is_bot_correction(lower):
            return "correction", True

        # 6) Project-scoped durable facts: for <project>, in <project>, etc.
        if self._is_project_fact(lower):
            return "project_fact", True

        # 7) Server/guild rules (not just any mention of server/guild)
        if self._is_server_fact(lower):
            return "server_fact", True

        # 8) Relationship note / inside joke
        rel = self._is_relationship_note(lower)
        joke = self._is_inside_joke(lower)
        if rel:
            return "relationship_note", True
        if joke:
            return "inside_joke", True

        # 9) Temporary context: only for short-lived guidance
        if self._is_temporary_context(lower):
            return "temporary_context", False

        # Default: not durable
        return "conversation_decision", False

    # ---------------- pattern matchers for inferred classification ----------------

    @staticmethod
    def _is_user_preference(lower: str) -> bool:
        """Strong preference signals only."""
        # Direct: "I prefer", "my preference is", etc.
        if re.search(r"\b(i prefer|my preference|i'd prefer|i would prefer)\b", lower):
            return True
        # Strong preference without filler: "prefer short replies"
        return bool(bool(re.search(r"(?:^|\s)prefer\b(?:\s+\w+){1,6}", lower)))

    @staticmethod
    def _is_recurring_instruction(lower: str) -> bool:
        """Always/never instructions that clearly target bot behavior.

        Must be a clear future-facing directive to the bot, e.g.:
          "always reply shorter", "never say that again", "from now on..."

        Reject:
          "i never did", "she never goes", "my friends never..."
        """
        # "from now on" / "going forward" / "henceforth" — strong future markers
        if bool(re.search(r"\bfrom\s+now\s+on\b", lower)):
            return True
        if bool(re.search(r"\bgoing\s+forward\b", lower)):
            return True
        if bool(re.search(r"\bhenceforth\b", lower)):
            return True

        # "always" with bot addressee or followed by content
        if bool(re.search(r"\byou\s+(?:should\s+|must\s+|have\s+to\s+)?always\b", lower)):
            return True
        if bool(re.search(r"\balways\b(?:\s+\w+){1,}", lower)):
            # "always" followed by at least one word — but reject "i always"
            if not bool(re.search(r"\b(?:i|my|she|he|they)\b.*\balways\b", lower)):
                return True

        # "you should never", "you must never", "you never" — bot-directed
        if bool(re.search(r"\byou\s+(?:should\s+|must\s+|have\s+to\s+)?never\b", lower)):
            return True
        # "don't do/use/say X again"
        if bool(re.search(r"\bdon't\s+(?:do|say|use|assume)\b.*\bagain\b", lower)):
            return True
        # "never do X again"
        return bool(bool(re.search(r"\bnever\s+(?:do|say|use|assume)\b.*\bagain\b", lower)))

    @staticmethod
    def _is_strong_remember(lower: str) -> bool:
        """Only treat 'remember' as durable when there is a clear object.

        Store:
          - "remember that I prefer X"
          - "remember this for discord-bot: X"
          - "remember, X is the rule"

        Do NOT store:
          - "remember when..."
          - "I don't remember"
          - "do you remember this?"
        """
        # Exclude weak patterns first
        if any(p in lower for p in ["don't remember", "i don't remember", "you remember"]):
            return False
        if any(lower.startswith(p) for p in ["remember when", "do you remember"]):
            return False

        # Positive: "remember that", "remember this", "remember, ..."
        if bool(re.search(r"\bremember\s+(?:that|this|it)\b", lower)):
            return True
        return bool(bool(re.search(r"\bremember,\s+\w", lower)))

    @staticmethod
    def _is_conversation_decision(lower: str) -> bool:
        """Decisions framed as group/project choices."""
        # "we decided", "the decision is"
        if bool(re.search(r"\bwe decided\b", lower)):
            return True
        return bool(bool(re.search(r"\bthe decision (?:is|for)\b", lower)))

    @staticmethod
    def _is_bot_correction(lower: str) -> bool:
        """Only treat as correction when it corrects bot behavior or a rule,
        not just any mention of 'correct'/'wrong' in code talk.

        Store:
          - "you were wrong about X; the correct rule is Y"
          - "that's wrong, ..."
          - "actually, ..."
          - "don't say/do that again"
          - "remember, X is Y not Z"

        Reject:
          - "this code is wrong"
          - "wrong output"
          - "correct this function"
        """
        # Direct corrections to bot:
        if bool(re.search(r"\byou (?:were|are) (?:wrong|incorrect)\b", lower)):
            return True
        if bool(re.search(r"\bthat's wrong\b", lower)):
            return True
        if bool(re.search(r"\bthe correct rule is\b", lower)):
            return True

        # "actually, ..." as correction:
        if bool(re.search(r"^\s*actually,\s+", lower)):
            return True

        # "don't say/do that again":
        if bool(re.search(r"\bdon't (?:do|say|use)\b.*again\b", lower)):
            return True

        # "remember, X is Y not Z" style:
        return bool(bool(re.search(r"\bremember,?\s+.*(?:is|should be).*not\b", lower)))

    def _is_project_fact(self, lower: str) -> bool:
        """Only treat as project_fact when the content is clearly about this bot,
        repo, codebase, runtime behavior, or a concrete subsystem.

        Accept only project-scoped statements such as:
          - "the bot should only reply when mentioned"
          - "RAG uses ChromaDB"
          - "memory service persists curated memories"
          - "typing indicator should wrap routed messages"

        Reject generic social/world claims or vague project chatter.
        """
        if bool(re.search(r"\b(?:working on|fixing|debugging)\b", lower)):
            return False
        if bool(re.search(r"\b(?:this|that|the)\s+project\b", lower)):
            return False
        if bool(re.search(r"\b(?:the\s+)?code\b.*\b(?:wrong|broken|buggy)\b", lower)):
            return False
        if bool(re.search(r"\b(?:annoying|frustrating|bad)\b", lower)) and "project" in lower:
            return False

        return self._looks_project_scoped_fact(lower)

    @staticmethod
    def _is_server_fact(lower: str) -> bool:
        """Only treat as server_fact when it's a clear rule/fact about the server,
        not just any mention of 'server' or 'guild'.
        """
        # Exclude generic mentions:
        if bool(re.search(r"\b(?:working on|fixing|debugging)\b", lower)):
            return False

        # Patterns:
        if bool(re.search(r"\b(this server|the server)\s+(should|must|uses|is)\b", lower)):
            return True
        if bool(re.search(r"\b(for this server|for the server),\s+", lower)):
            return True
        # guild-specific rule:
        return bool(bool(re.search(r"\b(this guild|the guild)\s+(should|must|uses|is)\b", lower)))

    @staticmethod
    def _is_relationship_note(lower: str) -> bool:
        # "our relationship", "we're close"
        return bool(bool(re.search(r"\b(?:our\s+)?relationship\b", lower)))

    @staticmethod
    def _is_inside_joke(lower: str) -> bool:
        return bool(bool(re.search(r"\b(?:inside joke|running joke)\b", lower)))

    @staticmethod
    def _is_temporary_context(lower: str) -> bool:
        """Only treat as temporary when it's clearly short-lived guidance.
        Do not treat 'today' as durable on its own; it must be combined with
        a weak/temporary phrase.
        """
        # Strong temp markers:
        if any(p in lower for p in ["for now", "temporarily", "this week"]):
            return True
        # 'today' only when paired with temporary-scope language:
        return bool("today" in lower and any(p in lower for p in ["for now", "until", "just for today", "temporarily"]))

    # ---------------- importance & confidence (inferred) ----------------

    def _importance_inferred(self, text: str, context_type: str) -> float:
        lower = text.lower()
        score = 0.55

        if context_type in {"user_preference", "recurring_instruction"}:
            score += 0.25
        elif context_type in {
            "project_fact",
            "server_fact",
            "conversation_decision",
            "correction",
        }:
            score += 0.20
        elif context_type in {"relationship_note", "inside_joke"}:
            score += 0.10
        elif context_type == "temporary_context":
            score += 0.05

        # Small boosts for strong language
        if any(w in lower for w in ["always", "never"]):
            score += 0.05
        if any(w in lower for w in ["important", "remember"]):
            score += 0.05

        return max(0.0, min(1.0, score))

    def _confidence(self, text: str, context_type: str) -> float:
        score = 0.72
        if context_type == "correction":
            score += 0.1
        if context_type in {
            "user_preference",
            "recurring_instruction",
            "project_fact",
            "server_fact",
        }:
            score += 0.1
        if len(text) > 80:
            score += 0.03
        return max(0.0, min(1.0, score))

    # ---------------- summary & expiration ----------------

    def _summarize(self, text: str, context_type: str) -> str:
        prefix_map = {
            "user_preference": "Prefers",
            "project_fact": "Project fact:",
            "server_fact": "Server fact:",
            "conversation_decision": "Decision:",
            "recurring_instruction": "Instruction:",
            "relationship_note": "Relationship note:",
            "inside_joke": "Inside joke:",
            "correction": "Correction:",
            "temporary_context": "Temporary context:",
        }
        prefix = prefix_map.get(context_type, "Memory:")
        summary = text.strip().rstrip(".")
        if len(summary) > 220:
            summary = summary[:217].rstrip() + "..."
        return f"{prefix} {summary}"

    def _expiration_for(self, context_type: str) -> str | None:
        days = self._ttl_for(context_type)
        if days <= 0:
            return None
        return (datetime.now(UTC) + timedelta(days=days)).isoformat()

    def _ttl_for(self, context_type: str) -> int:
        if context_type == "temporary_context":
            return self.temp_ttl_days
        return self.default_ttl_days

    @staticmethod
    def _metadata_json(metadata: dict[str, Any]) -> str:
        try:
            return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return "{}"

    # ---------------- structured decision logging ----------------

    @staticmethod
    def _log_inferred_decision(
        user_id: str,
        guild_id: str | None,
        channel_id: str | None,
        thread_id: str | None,
        source_message_id: str | None,
        context_type: str | None = None,
        reason: str = "rejected",
        importance: float | None = None,
        confidence: float | None = None,
        ttl_days: int | None = None,
    ) -> None:
        """Log an inferred memory decision in a structured way.

        Does not log raw message content to avoid leaking secrets.
        """
        extra = {
            "subsys": "memory",
            "event": "inferred_memory_decision",
            "user_id": user_id,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "thread_id": thread_id,
            "source_message_id": source_message_id,
            "reason": reason,
        }
        if context_type is not None:
            extra["context_type"] = context_type
        if importance is not None:
            extra["importance"] = importance
        if confidence is not None:
            extra["confidence"] = confidence
        if ttl_days is not None:
            extra["ttl_days"] = ttl_days

        if reason == "accepted":
            logger.info("Inferred memory accepted", extra=extra)
        else:
            logger.debug("Inferred memory rejected", extra=extra)


# Legacy compatibility wrapper for existing callers that relied on _DURABLE_HINTS.
# This is intentionally kept small and not used by new inferred logic.
_DURABLE_HINTS: dict[str, str] = {
    "prefer": "user_preference",
    "prefers": "user_preference",
    "call me": "user_preference",
    "always": "recurring_instruction",
    "never": "recurring_instruction",
    "remember": "recurring_instruction",
    "decided": "conversation_decision",
    "decision": "conversation_decision",
    "correct": "correction",
    "wrong": "correction",
    "project": "project_fact",
    "working on": "project_fact",
    "server": "server_fact",
    "guild": "server_fact",
    "relationship": "relationship_note",
    "inside joke": "inside_joke",
    "for now": "temporary_context",
    "temporarily": "temporary_context",
    "this week": "temporary_context",
    "today": "temporary_context",
}
