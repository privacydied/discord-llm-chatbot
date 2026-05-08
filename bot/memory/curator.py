"""Curates which interactions should become durable memories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from .persistent_store import MemoryRecord

_SECRET_PATTERNS = [
    r"\bpassword\b",
    r"\bpassphrase\b",
    r"\bapi[-_ ]?key\b",
    r"\bsecret\b",
    r"\btoken\b",
    r"\bauthorization\b",
    r"\bbearer\b",
    r"\bprivate key\b",
    r"-----BEGIN [A-Z ]+-----",
    r"\bghp_[A-Za-z0-9]{20,}\b",
    r"\bsk-[A-Za-z0-9]{20,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
]

_INTERNAL_PATTERNS = [
    r"chain of thought",
    r"think step by step",
    r"internal prompt",
    r"tool trace",
    r"function call",
    r"hidden reasoning",
    r"system prompt",
]

_DURABLE_HINTS = {
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


@dataclass(slots=True)
class MemoryCandidate:
    memory_id: str
    user_id: str
    guild_id: Optional[str]
    channel_id: Optional[str]
    thread_id: Optional[str]
    source_message_id: Optional[str]
    context_type: str
    text: str
    summary: str
    importance: float
    confidence: float
    source: str
    expires_at: Optional[str]
    metadata_json: str = "{}"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed_at: Optional[str] = None
    deleted_at: Optional[str] = None
    chroma_id: Optional[str] = None

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
    """Heuristic curation for durable memories."""

    def __init__(
        self,
        default_ttl_days: int = 180,
        temp_ttl_days: int = 14,
        min_importance: float = 0.55,
    ):
        self.default_ttl_days = int(default_ttl_days)
        self.temp_ttl_days = int(temp_ttl_days)
        self.min_importance = float(min_importance)

    def build_explicit_candidate(
        self,
        *,
        user_id: str,
        text: str,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        context_type: str = "user_preference",
        source: str = "explicit_memory_command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryCandidate]:
        text = self._normalize(text)
        if not text or self._looks_sensitive(text) or self._looks_internal(text):
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

    def curate_inferred_candidate(
        self,
        *,
        user_id: str,
        text: str,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryCandidate]:
        text = self._normalize(text)
        if not text or len(text) < 12:
            return None
        if self._looks_sensitive(text) or self._looks_internal(text):
            return None
        if not self._looks_durable(text):
            return None

        context_type = self._classify(text)
        importance = self._importance(text, context_type)
        if importance < self.min_importance and context_type != "correction":
            return None

        confidence = self._confidence(text, context_type)
        summary = self._summarize(text, context_type)
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
            source="inferred_curated",
            expires_at=expires_at,
            metadata_json=self._metadata_json(payload),
        )

    def _normalize(self, text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip())
        return text[:2000]

    def _looks_sensitive(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pattern, lower, flags=re.I) for pattern in _SECRET_PATTERNS)

    def _looks_internal(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pattern, lower, flags=re.I) for pattern in _INTERNAL_PATTERNS)

    def _looks_durable(self, text: str) -> bool:
        lower = text.lower()
        if any(hint in lower for hint in _DURABLE_HINTS):
            return True
        return bool(re.search(r"\b(i prefer|my preference|we decided|i corrected|i work on|i am working on|i always|i never)\b", lower))

    def _classify(self, text: str) -> str:
        lower = text.lower()
        for hint, context_type in _DURABLE_HINTS.items():
            if hint in lower:
                return context_type
        return "conversation_decision"

    def _importance(self, text: str, context_type: str) -> float:
        lower = text.lower()
        score = 0.55
        if context_type in {"user_preference", "recurring_instruction"}:
            score += 0.25
        elif context_type in {"project_fact", "server_fact", "conversation_decision", "correction"}:
            score += 0.20
        elif context_type in {"relationship_note", "inside_joke"}:
            score += 0.10
        elif context_type == "temporary_context":
            score += 0.05
        if "always" in lower or "never" in lower:
            score += 0.05
        if "important" in lower or "remember" in lower:
            score += 0.05
        return max(0.0, min(1.0, score))

    def _confidence(self, text: str, context_type: str) -> float:
        score = 0.72
        if context_type == "correction":
            score += 0.1
        if context_type in {"user_preference", "recurring_instruction", "server_fact"}:
            score += 0.1
        if len(text) > 80:
            score += 0.03
        return max(0.0, min(1.0, score))

    def _summarize(self, text: str, context_type: str) -> str:
        summary = text
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
        summary = summary.strip().rstrip(".")
        if len(summary) > 220:
            summary = summary[:217].rstrip() + "..."
        return f"{prefix} {summary}"

    def _expiration_for(self, context_type: str) -> Optional[str]:
        if context_type == "temporary_context":
            days = self.temp_ttl_days
        else:
            days = self.default_ttl_days
        if days <= 0:
            return None
        return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()

    def _metadata_json(self, metadata: Dict[str, Any]) -> str:
        try:
            return __import__("json").dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return "{}"
