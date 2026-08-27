"""Conversational image-edit routing helpers (mention/reply img2img). [CA]

Resolves a source image (attachment / reply / URL) and classifies whether an
addressed message expresses an image-EDIT instruction rather than an
analysis/question, so the router can send it down the existing img2img
orchestrator path instead of VL analysis. Pure/IO-light functions live here so
they can be unit tested without spinning up the full Router. [CSD]

v1 uses keyword heuristics (`classify_edit_intent`). A learned/LLM intent
classifier could later replace that function's body without changing its
signature or the router wiring that calls it.
"""

from __future__ import annotations

import json
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from bot.modality import ImageRef, collect_image_urls_from_message
from bot.utils.file_utils import download_robust_image
from bot.utils.logging import get_logger

if TYPE_CHECKING:
    from discord import Message

logger = get_logger(__name__)

# Fallback trigger phrases used only when configs/vision_policy.json can't be
# read; kept in sync with the spec's example edit verbs. [CMV]
_DEFAULT_EDIT_TRIGGER_PHRASES: tuple[str, ...] = (
    "edit",
    "modify",
    "change",
    "alter",
    "fix",
    "improve",
    "erase",
    "remove",
    "inpaint",
    "outpaint",
    "extend",
    "variation",
    "upscale",
    "enhance",
    "give him",
    "give her",
    "give it",
    "give this",
    "make him",
    "make her",
    "make it",
    "make this",
    "add a",
    "add some",
    "put a",
    "put some",
    "turn this into",
    "turn it into",
    "make it look like",
    "photoshop",
    "swap",
    "replace",
)

# Analysis/question intent always wins ties (safe default per spec). [SFT]
_ANALYSIS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(what|who|where|when|why|how)\b", re.IGNORECASE),
    re.compile(r"\b(describe|analy[sz]e|explain|identify|caption)\b", re.IGNORECASE),
    re.compile(r"\btell me about\b", re.IGNORECASE),
    re.compile(r"\bis this\b", re.IGNORECASE),
    re.compile(r"\?\s*$"),
)


@lru_cache(maxsize=4)
def _load_policy_trigger_phrases(policy_path: str) -> tuple[str, ...]:
    """Load `intent_patterns.image_editing.trigger_phrases` from vision policy [REH]."""
    try:
        with open(policy_path, encoding="utf-8") as f:
            policy = json.load(f)
        phrases = policy.get("intent_patterns", {}).get("image_editing", {}).get("trigger_phrases", [])
        cleaned = tuple(str(p).strip().lower() for p in phrases if str(p).strip())
        return cleaned or _DEFAULT_EDIT_TRIGGER_PHRASES
    except Exception as exc:
        logger.debug(f"vision_policy edit-phrase load failed, using defaults: {exc}")
        return _DEFAULT_EDIT_TRIGGER_PHRASES


@lru_cache(maxsize=8)
def _edit_trigger_phrases(policy_path: str, extra_keywords: str) -> tuple[str, ...]:
    """Merged trigger phrases, longest first so the most specific one is reported."""
    base = set(_load_policy_trigger_phrases(policy_path)) | set(_DEFAULT_EDIT_TRIGGER_PHRASES)
    extra = {kw.strip().lower() for kw in (extra_keywords or "").split(",") if kw.strip()}
    return tuple(sorted(base | extra, key=len, reverse=True))


@dataclass(frozen=True)
class EditIntentResult:
    """Outcome of the edit-vs-analysis heuristic classifier."""

    is_edit: bool
    matched_phrase: str | None = None


@lru_cache(maxsize=512)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole-word matcher for one trigger phrase. [IV]

    Substring matching is what made ``edit`` fire on "credit", "reddit" and
    "editor", ``fix`` on "prefix", and ``change`` on "exchange" -- turning an
    ordinary sentence into an image-generation job. Phrases may contain spaces
    ("give this"), so the boundaries are asserted around the whole phrase.
    """
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)


def classify_edit_intent(
    text: str,
    policy_path: str = "configs/vision_policy.json",
    extra_keywords: str = "",
) -> EditIntentResult:
    """Classify addressed message text as an edit instruction or not.

    Analysis-style questions ("what is this", "describe...", trailing "?")
    are treated as NOT an edit instruction even if an edit verb also appears,
    matching the spec's "ambiguous -> prefer VL analysis" default. [SFT]

    Only ever call this with text the requester actually typed: adopted text
    (a quoted parent message, an ingested .txt) is context, not a command.
    """
    if not text or not text.strip():
        return EditIntentResult(False)

    lowered = text.lower().strip()

    if any(pattern.search(lowered) for pattern in _ANALYSIS_PATTERNS):
        return EditIntentResult(False)

    # Phrases arrive longest-first: "make it look like" is reported over "make it".
    for phrase in _edit_trigger_phrases(policy_path, extra_keywords):
        if _phrase_pattern(phrase).search(lowered):
            return EditIntentResult(True, phrase)

    return EditIntentResult(False)


@dataclass(frozen=True)
class ResolvedEditImage:
    """A downloaded source image ready to feed into an img2img request."""

    data: bytes
    content_type: str | None
    source: str  # "current" | "reply" | "url"


async def _download_ref(ref: ImageRef, source: str, max_size_mb: int) -> ResolvedEditImage | None:
    """Download an ImageRef to bytes, enforcing MAX_ATTACHMENT_SIZE_MB. [RM][IV]"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = tmp.name
    try:
        ok = await download_robust_image(ref, tmp_path, max_size_mb=max_size_mb)
        if not ok:
            return None
        data = Path(tmp_path).read_bytes()
        if not data:
            return None
        return ResolvedEditImage(data=data, content_type=ref.content_type, source=source)
    finally:
        with suppress(OSError):
            Path(tmp_path).unlink()


async def resolve_edit_source_image(message: Message, max_size_mb: int) -> ResolvedEditImage | None:
    """Resolve the source image for a conversational edit, in priority order:

    1. an attachment/image-embed on the triggering message
    2. an attachment/image-URL on the message being REPLIED TO
    3. a bare image URL typed in the triggering message's text
    """
    cur_refs = collect_image_urls_from_message(message) or []
    if cur_refs:
        resolved = await _download_ref(cur_refs[0], "current", max_size_mb)
        if resolved:
            return resolved

    ref_message = None
    if getattr(message, "reference", None):
        try:
            ref_message = await message.channel.fetch_message(message.reference.message_id)
        except Exception as exc:
            logger.debug(f"edit_route: fetch reference message failed: {exc}")

    if ref_message:
        ref_refs = collect_image_urls_from_message(ref_message) or []
        if ref_refs:
            resolved = await _download_ref(ref_refs[0], "reply", max_size_mb)
            if resolved:
                return resolved

    from .input_harvest import extract_urls_strict, is_direct_image_url

    for url in extract_urls_strict(getattr(message, "content", "") or ""):
        if is_direct_image_url(url):
            resolved = await _download_ref(ImageRef(url=url), "url", max_size_mb)
            if resolved:
                return resolved

    return None
