"""
VL Output Sanitizer - Removes chain-of-thought leakage and model reasoning.

Enforces clean VL output before passing to Text Flow in the 1-hop pipeline.
"""

import os
import re
from typing import Optional


def sanitize_model_output(text: str) -> str:
    """
    Sanitize VL model output to remove reasoning, rules, and thinking blocks.

    Args:
        text: Raw VL model response

    Returns:
        Cleaned text suitable for Text Flow input
    """
    # Check if sanitization is enabled
    strip_reasoning = os.getenv("VL_STRIP_REASONING", "1").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not strip_reasoning:
        return text.strip()

    # Start with the original text
    cleaned = text.strip()

    # 1. Strip any <think>...</think> blocks (case-insensitive, multiline)
    think_pattern = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
    cleaned = think_pattern.sub("", cleaned)

    # 2. Remove first block of rules/explain steps boilerplate
    rules_patterns = [
        r"^.*?Follow specific rules.*?\n",
        r"^.*?Let me break down.*?\n",
        r"^.*?Don\'t speculate.*?\n",
        r"^.*?Here are the guidelines.*?\n",
        r"^.*?I need to follow.*?\n",
        r"^\s*[-•]\s*.*?rules?.*?\n",
        r"^\s*[-•]\s*.*?guidelines?.*?\n",
        r"^\s*\d+\.\s*.*?rules?.*?\n",
    ]

    for pattern in rules_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # 3. Remove common intros that echo instructions
    intro_patterns = [
        r"^Got your image\s*[—-]\s*here\'s what I see:?\s*",
        r"^I can see .*? image.*?\. Let me analyze.*?\s*",
        r"^Looking at .*? image.*?\s*",
        r"^I\'ll analyze .*? image.*?\s*",
        r"^Here\'s my analysis.*?:\s*",
        r"^Based on .*? guidelines.*?\s*",
    ]

    for pattern in intro_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # 4. Clean up whitespace and line breaks
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)  # Collapse multiple blank lines
    cleaned = cleaned.strip()

    # 5. Cap length to ~1000 chars on sentence boundary
    if len(cleaned) > 1000:
        # Find last sentence boundary before 1000 chars
        truncate_at = 1000
        sentence_endings = [".", "!", "?"]

        # Look backward from position 1000 to find sentence ending
        for i in range(
            min(1000, len(cleaned) - 1), 500, -1
        ):  # Don't go below 500 chars
            if cleaned[i] in sentence_endings and i < len(cleaned) - 1:
                # Make sure next char is whitespace or end of string
                if i + 1 >= len(cleaned) or cleaned[i + 1].isspace():
                    truncate_at = i + 1
                    break

        cleaned = cleaned[:truncate_at].strip()

    # 6. Final cleanup - remove any remaining empty lines at start/end
    cleaned = cleaned.strip()

    return cleaned


def has_reasoning_content(text: str) -> bool:
    """
    Check if text contains reasoning content that should be sanitized.

    Args:
        text: Text to check

    Returns:
        True if text contains <think> blocks or rule explanations
    """
    if not text:
        return False

    # Check for <think> blocks
    if re.search(r"<think>", text, re.IGNORECASE):
        return True

    # Check for common rule/guideline language
    rule_indicators = [
        "follow specific rules",
        "let me break down",
        "don't speculate",
        "here are the guidelines",
        "i need to follow",
        "based on the rules",
        "according to guidelines",
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in rule_indicators)


def sanitize_vl_reply_text(
    text: str, max_chars: Optional[int] = None, strip_reasoning: Optional[bool] = None
) -> str:
    """
    Sanitize VL text for reply-image flow to a concise, natural message.
    - Optionally strip chain-of-thought / planning text (default: on)
    - Keep first 3–5 descriptive lines (bullets or sentences)
    - Hard truncate to max_chars at sentence/space boundary with ellipsis

    Args:
        text: Raw VL model output
        max_chars: Character cap for final output (default 420 if unset)
        strip_reasoning: Whether to remove reasoning/plan text (default True if unset)

    Returns:
        Clean, concise text suitable for inline Discord replies.
    """
    if text is None:
        return ""

    # Defaults from env when not provided
    if max_chars is None:
        try:
            max_chars_env = os.getenv("VL_REPLY_MAX_CHARS", "420")
            max_chars = int(max_chars_env.strip())
        except Exception:
            max_chars = 420
    if strip_reasoning is None:
        strip_reasoning = os.getenv("VL_STRIP_REASONING", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    cleaned = text.strip()

    if strip_reasoning:
        # Remove <think>...</think> and <reasoning>...</reasoning> blocks
        cleaned = re.sub(
            r"<(think|reasoning)>.*?</\\1>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Drop common planning/reasoning lead-ins and boilerplate lines
        drop_line_patterns = [
            r"^\s*Thoughts?\s*:.*$",
            r"^\s*Thinking\s*:.*$",
            r"^\s*Reasoning\s*:.*$",
            r"^\s*I\s+should\s*:.*$",
            r"^\s*Steps?\s*:.*$",
            r"^\s*Plan\s*:.*$",
            r"^\s*Let's\s+think.*$",
            r"^\s*Let us\s+think.*$",
            # Numbered plans like "1. Do X" / "2) Do Y" at the start
            r"^\s*\d+\s*[\.)]\s+.*$",
        ]
        lines = cleaned.splitlines()
        kept_lines = []
        for line in lines:
            drop = False
            for pat in drop_line_patterns:
                if re.match(pat, line, flags=re.IGNORECASE):
                    drop = True
                    break
            if not drop:
                kept_lines.append(line)
        cleaned = "\n".join(kept_lines)

    # Normalize whitespace: collapse excessive blank lines
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned).strip()

    # Prefer descriptive bullets/sentences: keep first up to 5 lines that look like content
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if lines:

        def _is_content_line(s: str) -> bool:
            if not s:
                return False
            if s.startswith(("- ", "• ", "* ")):
                return True
            # Plain sentences ending with punctuation
            return s[-1:] in ".!?" or len(s.split()) >= 6

        filtered = [ln for ln in lines if _is_content_line(ln)]
        # If filtering removed everything, fall back to first few original lines
        chosen = filtered if filtered else lines
        # Keep between 3 and 5 lines when available
        take_n = 5 if len(chosen) >= 5 else (3 if len(chosen) >= 3 else len(chosen))
        cleaned = "\n".join(chosen[:take_n]).strip()

    # Final hard truncate by characters with sentence/space boundary preference
    if max_chars > 0 and len(cleaned) > max_chars:
        # Try to cut at last sentence boundary before limit
        boundary = -1
        for i in range(min(len(cleaned), max_chars), max(0, max_chars - 200), -1):
            if cleaned[i - 1] in ".!?":
                boundary = i
                break
        if boundary == -1:
            # Fallback: last space before limit
            space_idx = cleaned.rfind(" ", 0, max_chars)
            boundary = space_idx if space_idx != -1 else max_chars
        cleaned = cleaned[:boundary].rstrip() + "…"

    return cleaned.strip()
