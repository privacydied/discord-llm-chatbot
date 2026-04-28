"""
Public output sanitizer - last-mile safety layer before Discord send.

Ensures only public-facing assistant text is sent to Discord.
Blocks internal reasoning, chain-of-thought, mode-gate commentary, etc.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Tuple

from .utils.logging import get_logger

logger = get_logger(__name__)

# Patterns that indicate internal reasoning leakage
REASONING_LEAK_PATTERNS = [
    r"^\s*Okay\s*,?\s*the\s+user",
    r"^\s*The\s+user\s+shared",
    r"^\s*First\s*,?\s*I\s+need\s+to",
    r"^\s*I\s+need\s+to\s+figure\s+out",
    r"Checking\s+the\s+MODE\s*GATE",
    r"MODE\s*GATE",
    r"POLITICAL\s*MODE",
    r"NORMAL\s*MODE",
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
]

# Compiled regex for faster matching
_reasoning_pattern = re.compile(
    "|".join(f"({p})" for p in REASONING_LEAK_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)

SAFE_FALLBACK_MESSAGE = "I couldn't produce a clean public answer for that. Try again or give me a bit more context."


def _compute_text_hash(text: str) -> str:
    """Compute a short hash of text for logging (not the full content)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _matches_reasoning_pattern(text: str) -> Tuple[bool, str]:
    """
    Check if text matches any reasoning leak pattern.
    Returns (matched, matched_pattern_or_empty).
    """
    if not text:
        return False, ""

    match = _reasoning_pattern.search(text)
    if match:
        return True, match.group(0)
    return False, ""


def extract_public_reply_text(
    content: Optional[str],
    *,
    request_id: Optional[str] = None,
    message_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    guild_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Extract public-facing reply text from model output.

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

    # Check for reasoning patterns before any further processing
    has_leak, matched_pattern = _matches_reasoning_pattern(cleaned)

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


def has_reasoning_leakage(content: Optional[str]) -> bool:
    """
    Check if content contains reasoning leakage patterns.

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
