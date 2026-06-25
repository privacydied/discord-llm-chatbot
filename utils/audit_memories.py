"""Audit existing curated memories and optionally quarantine suspicious entries.

This script never permanently deletes data. In apply mode it sets a review state
that can be inspected before any destructive action.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.memory.persistent_store import PersistentMemoryStore  # noqa: E402

logger = logging.getLogger("audit_memories")

_DEFAULT_SQLITE = PROJECT_ROOT / "data" / "memory.db"
_DIAGNOSTIC_TOKENS = ("MODE:", "A/B test", "diagnostic", "A/B/MODE", "AB=")
_FUTURE_TOKENS = (
    "from now on",
    "going forward",
    "henceforth",
    "remember that",
    "use this for future",
    "call this",
    "for future use",
    "always",
    "never",
)
_RECURRING_LABEL = "recurring_instruction"


@dataclass
class MemoryIssue:
    memory_id: str
    reason: str
    category: str
    confidence: float | None
    decision_reason: str | None
    scope: str | None
    created_by: str | None
    preview: str


@dataclass
class QuarantineResult:
    dry_run: bool
    reviewed: int = 0
    quarantined: int = 0
    issues: list[MemoryIssue] = field(default_factory=list)


def _parse_metadata(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _preview(text: str, limit: int = 120) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


def _lower(s: object) -> str:
    return str(s or "").lower()


def _is_diagnostic(text: str) -> bool:
    lower = _lower(text)
    return any(token.lower() in lower for token in _DIAGNOSTIC_TOKENS)


def _looks_like_recurring_instruction(text: str) -> bool:
    lower = _lower(text)
    return any(token in lower for token in _FUTURE_TOKENS)


def _iter_records(store: PersistentMemoryStore, user_id: str | None = None) -> Iterable:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(store.list_memories(user_id=user_id, limit=500))


def _quarantine_record(store: PersistentMemoryStore, issue: MemoryIssue) -> bool:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(store.quarantine_memory(issue.memory_id))


def audit_memories(sqlite_path: Path, *, user_id: str | None = None, min_confidence: float = 0.7, max_preview_chars: int = 120) -> QuarantineResult:
    store = PersistentMemoryStore(sqlite_path)
    result = QuarantineResult(dry_run=True)
    for record in _iter_records(store, user_id=user_id):
        metadata = _parse_metadata(getattr(record, "metadata", None))
        text = getattr(record, "content", "") or getattr(record, "text", "") or ""
        category = str(getattr(record, "category", "") or metadata.get("category", ""))
        confidence = getattr(record, "confidence", None)
        if confidence is None:
            confidence = metadata.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        decision_reason = metadata.get("decision_reason")
        scope = metadata.get("scope")
        created_by = getattr(record, "source", None) or metadata.get("created_by")
        preview = _preview(text, limit=max_preview_chars)
        issues: list[str] = []

        if category == _RECURRING_LABEL and not _looks_like_recurring_instruction(text):
            issues.append("recurring_instruction_without_future_language")
        if confidence is not None and confidence < min_confidence:
            issues.append("low_confidence")
        if not metadata:
            issues.append("missing_metadata")
        if not decision_reason:
            issues.append("missing_decision_reason")
        if _is_diagnostic(text):
            issues.append("diagnostic_artifact")
        if len(text) > 2000:
            issues.append("overly_long")
        if category == _RECURRING_LABEL and not scope:
            issues.append("missing_scope_for_recurring_instruction")

        for reason in issues:
            result.issues.append(
                MemoryIssue(
                    memory_id=str(getattr(record, "memory_id", "")),
                    reason=reason,
                    category=str(category),
                    confidence=confidence,
                    decision_reason=str(decision_reason) if decision_reason is not None else None,
                    scope=str(scope) if scope is not None else None,
                    created_by=str(created_by) if created_by is not None else None,
                    preview=preview,
                )
            )
        result.reviewed += 1

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and optionally quarantine suspicious memories.")
    parser.add_argument("--sqlite-path", type=Path, default=_DEFAULT_SQLITE, help="Path to memory SQLite database.")
    parser.add_argument("--user-id", default=None, help="Limit audit to a single user id.")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Flag memories below this confidence.")
    parser.add_argument("--max-preview-chars", type=int, default=120, help="Preview length for reports.")
    parser.add_argument("--quarantine", action="store_true", help="Apply quarantine actions instead of dry-run.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.sqlite_path.exists():
        logger.error("sqlite path not found: %s", args.sqlite_path)
        return 2

    result = audit_memories(
        args.sqlite_path,
        user_id=args.user_id,
        min_confidence=args.min_confidence,
        max_preview_chars=args.max_preview_chars,
    )

    if args.quarantine and result.issues:
        result.dry_run = False
        store = PersistentMemoryStore(args.sqlite_path)
        for issue in result.issues:
            try:
                changed = _quarantine_record(store, issue)
            except Exception as exc:  # pragma: no cover - safety fallback
                logger.warning("quarantine failed for %s: %s", issue.memory_id, exc)
                changed = False
            if changed:
                result.quarantined += 1

    if args.json:
        payload = {
            "reviewed": result.reviewed,
            "quarantined": result.quarantined,
            "issues": [
                {
                    "memory_id": issue.memory_id,
                    "reason": issue.reason,
                    "category": issue.category,
                    "confidence": issue.confidence,
                    "decision_reason": issue.decision_reason,
                    "scope": issue.scope,
                    "created_by": issue.created_by,
                    "preview": issue.preview,
                }
                for issue in result.issues
            ],
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(f"Reviewed {result.reviewed} memories. Flagged {len(result.issues)} issues.")
        if result.quarantined:
            print(f"Quarantined {result.quarantined} memories.")
        for issue in result.issues[:50]:
            print(f"- {issue.memory_id}: {issue.reason} | {issue.category} | conf={issue.confidence} | {issue.preview}")

    return 0 if result.issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
