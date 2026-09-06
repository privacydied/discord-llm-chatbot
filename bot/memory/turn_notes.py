"""Shared contract for conversation-turn derived results.

A "turn note" is the textual understanding a specialized route derived from
media during a turn: an STT transcript, a VL description, extracted URL text,
a tweet payload, a document extraction. Notes attach to the *user turn* that
carried the media so later turns ("the video attached", "translate that")
resolve against recent chronological context instead of long-term memory.

This module is intentionally dependency-free (stdlib only): ``bot/memory``
must not depend on backend implementations, and route code in ``bot/router.py``
maps its own modality enums to the kind strings defined here. One shared
contract -- not per-route append logic. [CA]
"""

from __future__ import annotations

import os
import re

# Kind labels for derived results. Routes map their modality to one of these.
KIND_X_VIDEO = "x_video"
KIND_VIDEO = "video"
KIND_AUDIO = "audio"
KIND_IMAGE = "image"
KIND_URL = "url"
KIND_SCREENSHOT = "screenshot"
KIND_DOCUMENT = "document"

_KNOWN_KINDS = frozenset(
    {
        KIND_X_VIDEO,
        KIND_VIDEO,
        KIND_AUDIO,
        KIND_IMAGE,
        KIND_URL,
        KIND_SCREENSHOT,
        KIND_DOCUMENT,
    }
)

# Router modality names (InputModality.name) -> note kind. Keyed by name
# string so this module never imports the modality enum. [CA]
MODALITY_KIND_MAP = {
    "VIDEO_URL": KIND_VIDEO,
    "AUDIO_VIDEO_FILE": KIND_AUDIO,
    "SINGLE_IMAGE": KIND_IMAGE,
    "MULTI_IMAGE": KIND_IMAGE,
    "GENERAL_URL": KIND_URL,
    "SCREENSHOT_URL": KIND_SCREENSHOT,
    "PDF_DOCUMENT": KIND_DOCUMENT,
    "PDF_OCR": KIND_DOCUMENT,
}

# Evidence-part headers emitted by the attachments-multimodal flow, mapped to
# note kinds. Unrecognized headers fall back to KIND_DOCUMENT.
_EVIDENCE_HEADER_KINDS = (
    ("TRANSCRIPT", KIND_AUDIO),
    ("DOCUMENT", KIND_DOCUMENT),
    ("IMAGE ANALYSIS", KIND_IMAGE),
    ("TXT FILE", KIND_DOCUMENT),
)

_HEADER_RE = re.compile(r"^\[([^\]:]+)(?::\s*([^\]]*))?\]\s*(.*)$", re.DOTALL)

_URL_RE = re.compile(r"https?://[^\s<>\")\]]+")


def _max_note_chars() -> int:
    try:
        return max(200, int(os.getenv("CONTEXT_DERIVED_MAX_CHARS_PER_NOTE", "1500")))
    except (TypeError, ValueError):
        return 1500


def _max_notes_per_turn() -> int:
    try:
        return max(1, int(os.getenv("CONTEXT_DERIVED_MAX_NOTES_PER_TURN", "8")))
    except (TypeError, ValueError):
        return 8


def MAX_NOTE_CHARS() -> int:
    """Public accessor for the per-note truncation cap (env-tunable)."""
    return _max_note_chars()


def MAX_NOTES_PER_TURN() -> int:
    """Public accessor for the per-turn note count cap (env-tunable)."""
    return _max_notes_per_turn()


def normalize_kind(kind: str | None) -> str:
    """Normalize a caller-supplied kind to a known label (fallback: url)."""
    label = (kind or "").strip().lower()
    return label if label in _KNOWN_KINDS else KIND_URL


def compact_text(text: str, limit: int | None = None) -> str:
    """Deterministically truncate derived text, preserving head and tail.

    Keeps the first ~70% and last ~30% so both the media opening and the
    closing (what "the last part" refers to) survive compaction. [PA]
    """
    cap = limit if limit is not None and limit > 0 else _max_note_chars()
    body = (text or "").strip()
    if len(body) <= cap:
        return body
    head_len = int(cap * 0.7)
    tail_len = max(0, cap - head_len - 12)
    return f"{body[:head_len]}...[{len(body) - cap} chars omitted]...{body[len(body) - tail_len :]}" if tail_len else f"{body[:head_len]}...[{len(body) - cap} chars omitted]"


def build_note(kind: str | None, label: str | None, text: str | None) -> dict | None:
    """Build one capped derived-note dict, or None when there is nothing to keep."""
    body = (text or "").strip()
    if not body:
        return None
    return {
        "kind": normalize_kind(kind),
        "label": (label or "").strip()[:300],
        "text": compact_text(body),
    }


def note_from_aggregator_result(modality_name: str | None, item_name: str | None, result_text: str | None) -> dict | None:
    """Map one successful ResultAggregator entry to a turn note.

    Skips empty results and failure-shaped placeholders ("...timed out...",
    "Failed: ...") which carry no referent value for later turns. [REH]
    """
    body = (result_text or "").strip()
    if not body:
        return None
    lowered = body.lower()
    if lowered.startswith(("❌", "⏱️", "⚠️ screenshot captured but")):
        return None
    kind = MODALITY_KIND_MAP.get((modality_name or "").upper(), KIND_URL)
    return build_note(kind, item_name, body)


def note_from_evidence_part(part: str | None) -> dict | None:
    """Map one `[HEADER] body` evidence part to a turn note."""
    body = (part or "").strip()
    if not body:
        return None
    match = _HEADER_RE.match(body)
    if not match:
        return build_note(KIND_DOCUMENT, "", body)
    header, header_label, rest = match.group(1).strip().upper(), (match.group(2) or "").strip(), (match.group(3) or "").strip()
    kind = KIND_DOCUMENT
    for prefix, mapped in _EVIDENCE_HEADER_KINDS:
        if header.startswith(prefix):
            kind = mapped
            break
    return build_note(kind, header_label or header, rest)


def extract_urls(text: str | None) -> list[str]:
    """Extract URL identity descriptors from raw message text (bounded)."""
    found = _URL_RE.findall(text or "")
    seen: list[str] = []
    for url in found:
        cleaned = url.rstrip(".,;!?)'\"")
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
        if len(seen) >= 10:
            break
    return seen


def enrich_items_with_media_notes(bot: object, trigger_message: object, items: object) -> dict[str, str]:
    """Exact-ID derived-note lookup for local context builders.

    Thread-tail and reply-chain blocks are built from live Discord messages
    (raw text only). This maps each packaged item's Discord message ID to the
    capped multimodal results stored for that turn, so builders can render
    transcripts / VL descriptions under the raw text. Duck-typed throughout:
    returns {} whenever the manager is absent (tests, bots without enhanced
    context). Never raises. [CA][REH]
    """
    try:
        manager = getattr(bot, "enhanced_context_manager", None)
        if manager is None:
            return {}
        resolve_key = getattr(manager, "resolve_key", None)
        get_notes = getattr(manager, "get_media_notes_for", None)
        if not callable(resolve_key) or not callable(get_notes):
            return {}
        key = resolve_key(trigger_message)
        ids: list[str] = []
        for it in items or []:
            mid = str(getattr(it, "id", "") or "")
            if mid and mid not in ids:
                ids.append(mid)
        if not ids:
            return {}
        notes = get_notes(ids, key)
        return dict(notes or {})
    except Exception:  # noqa: BLE001 - duck-typed manager methods; contract is Never raises (see docstring)
        return {}
