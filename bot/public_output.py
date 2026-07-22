"""Public output sanitizer - last-mile safety layer before Discord send.

Ensures only public-facing assistant text is sent to Discord.
Blocks internal reasoning, chain-of-thought, mode-gate commentary, etc.
Also strips internal labels, aggregation markers, prompt fragments, and
debug/routing identifiers that should never reach the public channel.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from .utils.logging import get_logger
from .utils.output_sanitizer import strip_leading_mode_preamble

if TYPE_CHECKING:
    import discord

logger = get_logger(__name__)

# Patterns that indicate internal reasoning leakage
REASONING_LEAK_PATTERNS = [
    # Existing patterns
    r"^\s*Okay\s*,?\s*the\s+user",
    r"^\s*The\s+user\s+shared",
    r"^\s*First\s*,?\s*I\s+need\s+to",
    r"^\s*I\s+need\s+to\s+figure\s+out",
    r"Checking\s+the\s+MODE\s*GATE",
    r"MODE\s*GATE",
    r"POLITICAL\s*MODE",
    r"NORMAL\s*MODE",
    # CoT narration of the mode-decision procedure (observed leaking from
    # reasoning models restating the system prompt's gate logic) [REH]
    r"\bMODE\s*=",
    r"let\s+me\s+re-?read",
    r"→\s*MODE\b",
    r"EXPLICIT_LENS_REQUEST",
    r"POLITICS_CORE_TOPIC",
    r"chain-of-thought",
    r"hidden\s+reasoning",
    r"scratchpad",
    r"^\s*analysis\s*:",
    r"^\s*reasoning\s*:",
    r"<thinking>",
    r"</thinking>",
    r"<reasoning>",
    r"</reasoning>",
    r"<scratchpad>",
    r"</scratchpad>",
    # Additional patterns for comprehensive coverage [REH]
    r"I\s+should\s+analyze",
    r"Let\s+me\s+analyze",
    r"I\s+will\s+analyze",
    r"I\s+need\s+to\s+analyze",
    r"Based\s+on\s+the\s+above",
    r"According\s+to\s+the\s+rules",
    r"As\s+an\s+AI\s+(assistant|language\s+model)",
    r"^\s*thought\s*:\s*",
    r"^\s*plan\s*:\s*",
    r"^\s*steps?\s*:\s*",
    r"<tool_call>",
    r"^\s*\]\s*$",  # standalone ] on its own line only
    r"<analysis>",
    r"</analysis>",
    # Internal prompt/context wrapper labels
    r"\bsystem\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\binternal\s+prompt\b",
    # v2: tool-call / JSON-like leakage
    r"^\s*\{\s*\"tool\"\s*:",
    r"^\s*\{\s*\"type\"\s*:\s*\"(code|function)\"",
    r"^\s*\[\s*\{\s*\"name\"\s*:",
    # v2: internal routing/status fragments
    r"^\s*(dispatch|route|router|pipeline|ingest|backpressure|queue_status|health)\s*[:=]\s*(?:internal|status|ok|fail|timeout)",
    # v2: analysis/final/commentary role leakage
    r"^\s*(final\s+answer|analysis\s+summary|commentary\s+only)",
    # v2: raw prompt scaffolding markers
    r"^\s*<system>",
    r"^\s*</system>",
    r"^\s*<instruction>",
    r"^\s*</instruction>",
]

# Compiled regex for faster matching
_reasoning_pattern = re.compile(
    "|".join(f"({p})" for p in REASONING_LEAK_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)

SAFE_FALLBACK_MESSAGE = "I couldn't produce a usable reply for that. Try rephrasing it."

_UNSAFE_FALLBACK_RE = re.compile(
    r"^\s*I\s+didn't\s+receive\s+a\s+usable\s+response"
    r"(?:\s+from\s+the\s+model(?:\s+just\s+now)?)?"
    r"\.\s*Could\s+you\s+rephrase(?:\s+or\s+add\s+a\s+bit\s+more\s+detail\s+about\s+.*)?\s*$",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Compiled patterns for sanitize_public_text
# ---------------------------------------------------------------------------

# a) Internal aggregation headers: "### [1/3] ..."
# Matches at line start (multiline) AND mid-line (no-MULTILINE) to catch
# headers quoted inside fallback text like: ...about "### [1/3] ✅ Image: ..."?
_AGGREGATION_HEADER_RE = re.compile(
    r"^\s*###\s*\[\d+/\d+\]\s*[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)

# b) "### Original Message Text:" header lines
_ORIGINAL_MSG_HEADER_RE = re.compile(
    r"^\s*###\s*Original\s+Message\s+Text\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

# c) Standalone [n/m] at line start that look like internal routing.
#    Matched when followed by screenshot-style aggregation content, emoji,
#    or modality words. Does NOT match [1/3] mid-sentence in user content
#    like "Add [1/3] cup".
_INTERNAL_ROUTING_LABEL_RE = re.compile(
    r"^\s*\[\d+/\d+\]\s+[^\n—-]{1,80}\s*[—-]\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+UTC\b.*$",
    re.MULTILINE,
)
_INTERNAL_ROUTING_LABEL_SIMPLE_RE = re.compile(
    r"^\s*\[\d+/\d+\]\s+(?:[✅❌⚠️🔍📋]|(?:Image|URL|Screenshot|Video|Audio|File|Message)\b).*$",
    re.IGNORECASE | re.MULTILINE,
)

# d) Internal section markers
_INTERNAL_SECTION_MARKERS_RE = re.compile(
    r"^\s*(?:VISUAL_FACTS|vl\s+prompt\s+output|Internal\s+intent)\s*:.*$"
    r"|\[Tweet\s+Caption\]",
    re.IGNORECASE | re.MULTILINE,
)

# e) Lines that are purely internal identifiers
_INTERNAL_IDENTIFIER_RE = re.compile(
    r"^\s*_process_multimodal_message_internal\s*$"
    r"|^\s*VL_DEBUG_FLOW\s*$",
    re.MULTILINE,
)

# e2) Inline route/debug labels that point to internal pipeline stages
_INTERNAL_ROUTE_DEBUG_LABEL_RE = re.compile(
    r"^\s*(?:route|debug|internal)\s*[:=]\s*(?:_process_multimodal_message_internal|attachments|vision|analysis|fallback|status|progress|message|image|video|audio|url)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

# f) System/developer/internal prompt/message labels as section headers
_PROMPT_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:system|developer|internal)\s+(?:prompt|message)\s*:\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)

# g) Internal-format timestamps: "digits — YYYY-MM-DD HH:MM UTC"
_INTERNAL_TIMESTAMP_RE = re.compile(
    r"^\s*\d+\s*[—\-–]\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+UTC\s*$",
    re.MULTILINE,
)

# h) <analysis>...</analysis> tags with content
_ANALYSIS_TAG_RE = re.compile(
    r"<analysis>.*?</analysis>",
    re.DOTALL | re.IGNORECASE,
)

# Standalone <analysis> or </analysis> leftover tags (in case not paired)
_ANALYSIS_LEFTOVER_RE = re.compile(
    r"</?analysis>",
    re.IGNORECASE,
)

# Whitespace collapse: 3+ newlines -> 2, strip trailing per-line whitespace
_MULTI_BLANK_LINE_RE = re.compile(r"\n\s*\n\s*\n+")
_TRAILING_LINE_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def _compute_text_hash(text: str) -> str:
    """Compute a short hash of text for logging (not the full content)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Minimum characters that must survive line-level leak stripping for the
# salvaged reply to be sent instead of the safe fallback. [CMV]
_MIN_SALVAGE_CHARS = 40

_REASONING_BLOCK_RE = re.compile(
    r"<(thinking|reasoning|scratchpad|analysis)>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_leaking_lines(text: str) -> str:
    """Remove the reasoning-leak REGION, preserving the rest of the reply. [REH]

    Reasoning leaks are contiguous: chain-of-thought runs as one block with
    pattern-matching lines interleaved among narration lines that match
    nothing ('So MODE = ... ? Wait, let me re-read...'). Dropping only the
    matching lines shipped the narration between them. Instead, remove the
    whole span from the FIRST leak-matching line through the LAST one — lines
    before and after the region (the actual reply) survive.
    """
    text = _REASONING_BLOCK_RE.sub("", text)
    lines = text.splitlines()
    match_idx = [i for i, line in enumerate(lines) if _reasoning_pattern.search(line)]
    if not match_idx:
        return text.strip()
    kept = lines[: match_idx[0]] + lines[match_idx[-1] + 1 :]
    return "\n".join(kept).strip()


def _matches_reasoning_pattern(text: str) -> tuple[bool, str]:
    """Check if text matches any reasoning leak pattern.
    Returns (matched, matched_pattern_or_empty).
    """
    if not text:
        return False, ""

    match = _reasoning_pattern.search(text)
    if match:
        return True, match.group(0)
    return False, ""


def _matches_unsafe_fallback_pattern(text: str) -> tuple[bool, str]:
    """Detect unsafe fallback text that quotes internal context."""
    if not text:
        return False, ""

    match = _UNSAFE_FALLBACK_RE.search(text)
    if match:
        return True, match.group(0)
    return False, ""


def extract_public_reply_text(
    content: str | None,
    *,
    request_id: str | None = None,
    message_id: str | None = None,
    channel_id: str | None = None,
    guild_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Extract public-facing reply text from model output.

    This is the final safety layer before sending to Discord.
    It:
    1. Handles None/empty input gracefully
    2. Detects and blocks reasoning/CoT leakage
    3. Normalizes whitespace
    4. Returns safe fallback if content is unsafe

    Args:
        content: Raw model output or text candidate
        request_id: Optional request ID for logging
        message_id: Optional Discord message ID for logging
        channel_id: Optional Discord channel ID for logging
        guild_id: Optional Discord guild ID for logging
        provider: Optional provider name for logging
        model: Optional model name for logging

    Returns:
        Safe public text ready for Discord

    """
    # Handle None/empty
    if content is None:
        return SAFE_FALLBACK_MESSAGE

    # Strip basic whitespace but preserve content
    cleaned = content.strip()
    if not cleaned:
        return SAFE_FALLBACK_MESSAGE

    unsafe_fallback, matched_fallback = _matches_unsafe_fallback_pattern(cleaned)
    if unsafe_fallback:
        text_hash = _compute_text_hash(cleaned)
        log_extra = {
            "event": "public_output.unsafe_fallback_blocked",
            "pattern_matched": matched_fallback[:120] if matched_fallback else "",
            "content_length": len(cleaned),
            "content_hash": text_hash,
        }
        if request_id:
            log_extra["request_id"] = request_id
        if message_id:
            log_extra["message_id"] = message_id
        if channel_id:
            log_extra["channel_id"] = channel_id
        if guild_id:
            log_extra["guild_id"] = guild_id
        if provider:
            log_extra["provider"] = provider
        if model:
            log_extra["model"] = model

        logger.warning(
            "Blocked unsafe fallback leak: pattern='%s' hash=%s len=%d",
            matched_fallback[:120] if matched_fallback else "",
            text_hash,
            len(cleaned),
            extra=log_extra,
        )
        return SAFE_FALLBACK_MESSAGE

    # Check for reasoning patterns before any further processing, after stripping
    # any leaked leading MODE label so reasoning-pattern logging reflects the
    # true content that will be sent.
    cleaned = strip_leading_mode_preamble(cleaned)
    has_leak, matched_pattern = _matches_reasoning_pattern(cleaned)

    if has_leak:
        # Salvage first: strip only the leaking lines/blocks. If a substantive,
        # leak-free reply remains, send that instead of nuking the whole
        # response — a reasoning model echoing one system-prompt phrase used to
        # cost users an entire good answer. [REH]
        salvaged = _strip_leaking_lines(cleaned)
        if len(salvaged) >= _MIN_SALVAGE_CHARS and not _matches_reasoning_pattern(salvaged)[0]:
            logger.info(
                "Reasoning leak stripped: pattern='%s' removed=%d kept=%d",
                matched_pattern[:50] if matched_pattern else "",
                len(cleaned) - len(salvaged),
                len(salvaged),
                extra={
                    "event": "public_output.reasoning_stripped",
                    "pattern_matched": matched_pattern[:50] if matched_pattern else "",
                    "removed_chars": len(cleaned) - len(salvaged),
                    "kept_chars": len(salvaged),
                },
            )
            cleaned = salvaged
            has_leak = False

    if has_leak:
        # Log the leak with metadata (not the full content)
        text_hash = _compute_text_hash(cleaned)
        log_extra = {
            "event": "public_output.reasoning_blocked",
            "pattern_matched": matched_pattern[:50] if matched_pattern else "",
            "content_length": len(cleaned),
            "content_hash": text_hash,
        }
        if request_id:
            log_extra["request_id"] = request_id
        if message_id:
            log_extra["message_id"] = message_id
        if channel_id:
            log_extra["channel_id"] = channel_id
        if guild_id:
            log_extra["guild_id"] = guild_id
        if provider:
            log_extra["provider"] = provider
        if model:
            log_extra["model"] = model

        logger.warning(
            "Blocked reasoning leak: pattern='%s' hash=%s len=%d",
            matched_pattern[:50] if matched_pattern else "",
            text_hash,
            len(cleaned),
            extra=log_extra,
        )
        return SAFE_FALLBACK_MESSAGE

    # Normalize excessive whitespace (collapse multiple blank lines)
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    cleaned = cleaned.strip()

    # Final empty check after normalization
    if not cleaned:
        return SAFE_FALLBACK_MESSAGE

    return cleaned


def has_reasoning_leakage(content: str | None) -> bool:
    """Check if content contains reasoning leakage patterns.

    This is a lightweight check for callers that want to handle
    sanitization themselves.

    Args:
        content: Text to check

    Returns:
        True if content contains reasoning leakage

    """
    if not content:
        return False
    matched, _ = _matches_reasoning_pattern(content)
    return matched


def sanitize_public_text(text: str) -> str:
    """Strip internal labels, aggregation markers, prompt fragments, and
    debug/routing identifiers from text before it reaches Discord.

    This is the single public-output sanitizer called at the send boundary.
    It is deterministic, idempotent, and never returns empty unless the
    original was empty (in which case SAFE_FALLBACK_MESSAGE is returned).

    Safe user content like "[1/3] cup of flour" is preserved; only patterns
    that match internal routing/aggregation structure are stripped.
    """
    if not text:
        return ""

    original_len = len(text)
    original_stripped_len = len(text.strip())
    cleaned = strip_leading_mode_preamble(text)

    unsafe_fallback, matched_fallback = _matches_unsafe_fallback_pattern(cleaned)
    if unsafe_fallback:
        logger.warning(
            "Blocked unsafe fallback leak during sanitization: pattern='%s' hash=%s len=%d",
            matched_fallback[:120] if matched_fallback else "",
            _compute_text_hash(cleaned),
            len(cleaned),
            extra={
                "event": "public_output.unsafe_fallback_blocked",
                "pattern_matched": matched_fallback[:120] if matched_fallback else "",
                "content_length": len(cleaned),
                "content_hash": _compute_text_hash(cleaned),
            },
        )
        return SAFE_FALLBACK_MESSAGE

    # a) Strip internal aggregation headers: "### [1/3] ✅ Image: ..."
    cleaned = _AGGREGATION_HEADER_RE.sub("", cleaned)

    # b) Strip "### Original Message Text:" headers
    cleaned = _ORIGINAL_MSG_HEADER_RE.sub("", cleaned)

    # c) Strip standalone [n/m] labels at line start that look like routing
    cleaned = _INTERNAL_ROUTING_LABEL_RE.sub("", cleaned)
    cleaned = _INTERNAL_ROUTING_LABEL_SIMPLE_RE.sub("", cleaned)

    # d) Strip internal section markers (VISUAL_FACTS:, vl prompt output:, etc.)
    cleaned = _INTERNAL_SECTION_MARKERS_RE.sub("", cleaned)

    # e) Strip lines that are purely internal identifiers
    cleaned = _INTERNAL_IDENTIFIER_RE.sub("", cleaned)
    cleaned = _INTERNAL_ROUTE_DEBUG_LABEL_RE.sub("", cleaned)

    # f) Strip system/developer/internal prompt section headers
    cleaned = _PROMPT_SECTION_HEADER_RE.sub("", cleaned)

    # g) Strip internal-format timestamps: "000 — 2026-04-29 23:36 UTC"
    cleaned = _INTERNAL_TIMESTAMP_RE.sub("", cleaned)

    # h) Strip <analysis>...</analysis> blocks and leftover tags
    cleaned = _ANALYSIS_TAG_RE.sub("", cleaned)
    cleaned = _ANALYSIS_LEFTOVER_RE.sub("", cleaned)

    # Collapse whitespace: multiple blank lines → double newline
    cleaned = _MULTI_BLANK_LINE_RE.sub("\n\n", cleaned)
    cleaned = _TRAILING_LINE_WS_RE.sub("", cleaned)
    cleaned = cleaned.strip()

    # If original input was only whitespace, return empty string (not fallback).
    if not cleaned and original_stripped_len == 0:
        return ""

    # If stripping removed all content from originally non-empty text,
    # return the safe fallback instead of an empty string.
    if not cleaned and original_len > 0:
        return SAFE_FALLBACK_MESSAGE

    return cleaned


def sanitize_embed_for_public(embed: discord.Embed) -> discord.Embed:
    """Sanitize all text fields of a Discord Embed for public output.

    Mutates the embed in place and returns it.
    Handles None values gracefully. If embed is None, returns None.
    """
    if embed is None:
        return None

    # Title
    if embed.title:
        embed.title = sanitize_public_text(embed.title) or ""

    # Description
    if embed.description:
        embed.description = sanitize_public_text(embed.description) or ""

    # Fields — discord.py stores them in embed._fields as list of dicts
    # We must clear and re-add because field name/value are read-only after add
    if embed.fields:
        saved_fields = []
        for f in embed.fields:
            saved_fields.append(
                {
                    "name": sanitize_public_text(f.name) or "\u200b",
                    "value": sanitize_public_text(f.value) or "\u200b",
                    "inline": f.inline,
                },
            )
        # Clear existing fields by setting internal _fields to empty
        embed._fields = []
        for sf in saved_fields:
            embed.add_field(
                name=sf["name"],
                value=sf["value"],
                inline=sf["inline"],
            )

    # Footer text
    if embed.footer and embed.footer.text:
        new_footer = sanitize_public_text(embed.footer.text) or ""
        embed.set_footer(
            text=new_footer,
            icon_url=embed.footer.icon_url or None,
        )

    return embed


def sanitize_embed_collection_for_public(
    embeds: list[discord.Embed] | None,
) -> list[discord.Embed]:
    """Sanitize a list of embeds for public Discord output."""
    if not embeds:
        return []
    return [sanitize_embed_for_public(embed) for embed in embeds if embed is not None]


def sanitize_public_message_payload(
    content: str | None = None,
    *,
    embed: discord.Embed | None = None,
    embeds: list[discord.Embed] | None = None,
) -> tuple[str | None, discord.Embed | None, list[discord.Embed]]:
    """Sanitize outbound Discord message payload text immediately before send."""
    sanitized_content = sanitize_public_text(content) if content is not None else None
    sanitized_embed = sanitize_embed_for_public(embed) if embed is not None else None
    sanitized_embeds = sanitize_embed_collection_for_public(embeds)
    return sanitized_content, sanitized_embed, sanitized_embeds
