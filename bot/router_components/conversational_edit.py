"""Conversational image-edit routing helpers (mention/reply img2img). [CA]

Resolves a source image (attachment / reply / URL) and classifies whether an
addressed message expresses an image-EDIT instruction rather than an
analysis/question, so the router can send it down the existing img2img
orchestrator path instead of VL analysis. Pure/IO-light functions live here so
they can be unit tested without spinning up the full Router. [CSD]

Two trigger forms are supported:
1. Explicit mention trigger: ``@Bot edit: <prompt>`` or ``@Bot edit <prompt>``
   — the ``edit`` keyword followed by optional colon is a direct instruction.
2. Heuristic keyword trigger: ``@Bot give him a beard`` — classified by
   ``classify_edit_intent`` using a policy-driven phrase list.

v1 uses keyword heuristics (`classify_edit_intent`). A learned/LLM intent
classifier could later replace that function's body without changing its
signature or the router wiring that calls it.
"""

from __future__ import annotations

import json
import re
import shlex
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


# Pattern for the explicit mention trigger: "edit:" or "edit" as the first token
# after the bot mention has been stripped. Captures everything after as the prompt.
# Matches: "edit: make him chinese", "edit make him chinese", "EDIT: foo".
# Does NOT match: "edited by", "credit me", "editor picks" (word boundary on left).
_EXPLICIT_EDIT_RE = re.compile(
    r"""^\s*edit(?::|\s+)\s*(.+)$""",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplicitEditInvocation:
    """Parsed result of an explicit ``@Bot edit: <prompt>`` trigger."""

    prompt: str  # everything after "edit:" / "edit "


def parse_explicit_edit_trigger(text: str) -> ExplicitEditInvocation | None:
    """Detect the explicit mention-triggered edit form.

    After the bot mention is stripped, a message like ``edit: make him chinese``
    or ``edit make him chinese`` is an unambiguous edit instruction — no
    heuristic needed. Returns ``None`` when the text does not match this form
    so the caller can fall back to ``classify_edit_intent``. [CA]
    """
    if not text:
        return None
    match = _EXPLICIT_EDIT_RE.match(text.strip())
    if not match:
        return None
    prompt = match.group(1).strip()
    if not prompt:
        return None
    return ExplicitEditInvocation(prompt=prompt)


# --- Flag parsing for the explicit "edit:" trigger -------------------------
#
# Mirrors /imgedit's option set (bot/commands/vision_commands.py:216-222) so
# the mention-triggered route isn't a stripped-down version of the slash
# command. Bounds are identical to /imgedit's app_commands.Range validators. [IV][CMV]

STEPS_RANGE = (10, 50)
STRENGTH_RANGE = (0.1, 1.0)
GUIDANCE_RANGE = (1.0, 20.0)
_KNOWN_PROVIDERS = ("together", "novita", "auto")

EDIT_FLAGS_HELP_TEXT = (
    "**Image edit usage**\n"
    "`@Bot edit: <prompt>` or `@Bot edit <prompt>` — attach an image, reply to "
    "one, or include a direct image URL.\n"
    "Optional flags (same ranges as `/imgedit`):\n"
    "`-seed <int>` `-steps <10-50>` `-strength <0.1-1.0>` `-guidance <1-20>` "
    "`-negative <text>` `-provider <together|novita|auto>` `-use <model>`\n"
    "Example: `@Bot edit: make him a superhero -steps 20 -strength 0.6`"
)

# Flags that consume the following token as their value.
_VALUE_FLAGS = frozenset({"-seed", "-steps", "-strength", "-guidance", "-negative", "-provider", "-use"})


@dataclass(frozen=True)
class ParsedEditFlags:
    """Result of extracting ``-flag value`` pairs out of an edit prompt.

    ``errors`` holds human-readable validation failures (e.g. an out-of-range
    ``-steps``); the caller should surface these instead of submitting a job
    when non-empty. [REH]
    """

    prompt: str
    seed: int | None = None
    steps: int | None = None
    strength: float | None = None
    guidance: float | None = None
    negative: str | None = None
    provider: str | None = None
    model: str | None = None
    help_requested: bool = False
    errors: tuple[str, ...] = ()


def _parse_flag_value(flag: str, raw: str, errors: list[str]) -> tuple[str, object | None]:
    """Validate one flag's value against /imgedit's bounds. Returns (field_name, value)."""
    if flag == "-seed":
        try:
            return "seed", int(raw)
        except ValueError:
            errors.append(f"`-seed` must be a whole number, got `{raw}`.")
            return "seed", None
    if flag == "-steps":
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"`-steps` must be a whole number, got `{raw}`.")
            return "steps", None
        if not (STEPS_RANGE[0] <= value <= STEPS_RANGE[1]):
            errors.append(f"`-steps` must be between {STEPS_RANGE[0]} and {STEPS_RANGE[1]}, got `{value}`.")
            return "steps", None
        return "steps", value
    if flag == "-strength":
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"`-strength` must be a number, got `{raw}`.")
            return "strength", None
        if not (STRENGTH_RANGE[0] <= value <= STRENGTH_RANGE[1]):
            errors.append(f"`-strength` must be between {STRENGTH_RANGE[0]} and {STRENGTH_RANGE[1]}, got `{value}`.")
            return "strength", None
        return "strength", value
    if flag == "-guidance":
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"`-guidance` must be a number, got `{raw}`.")
            return "guidance", None
        if not (GUIDANCE_RANGE[0] <= value <= GUIDANCE_RANGE[1]):
            errors.append(f"`-guidance` must be between {GUIDANCE_RANGE[0]} and {GUIDANCE_RANGE[1]}, got `{value}`.")
            return "guidance", None
        return "guidance", value
    if flag == "-provider":
        low = raw.strip().lower()
        if low not in _KNOWN_PROVIDERS:
            errors.append(f"`-provider` must be one of {', '.join(_KNOWN_PROVIDERS)}, got `{raw}`.")
            return "provider", None
        return "provider", low
    if flag == "-negative":
        return "negative", raw
    return "model", raw  # -use


def parse_edit_flags(text: str) -> ParsedEditFlags:
    """Extract ``-seed``/``-steps``/``-strength``/``-guidance``/``-negative``/
    ``-provider``/``-use``/``-h`` flags out of an explicit edit prompt.

    Uses ``shlex`` so quoted multi-word values (e.g. ``-negative "blurry, low
    quality"``) work; falls back to a plain whitespace split on unbalanced
    quotes so a typo doesn't swallow the whole prompt. Unrecognized tokens are
    rejoined, in order, as the cleaned prompt. [IV]
    """
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()

    errors: list[str] = []
    prompt_parts: list[str] = []
    values: dict[str, object] = {}
    help_requested = False

    i = 0
    while i < len(tokens):
        token = tokens[i]
        low = token.lower()
        if low in ("-h", "--help"):
            help_requested = True
            i += 1
            continue
        if low in _VALUE_FLAGS:
            if i + 1 >= len(tokens):
                errors.append(f"`{token}` needs a value.")
                i += 1
                continue
            field, value = _parse_flag_value(low, tokens[i + 1], errors)
            if value is not None:
                values[field] = value
            i += 2
            continue
        prompt_parts.append(token)
        i += 1

    return ParsedEditFlags(
        prompt=" ".join(prompt_parts).strip(),
        help_requested=help_requested,
        errors=tuple(errors),
        **values,
    )


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
