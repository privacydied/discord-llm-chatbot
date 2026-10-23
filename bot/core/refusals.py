"""Detect model-level refusals so the reply can be retried on another model. [REH]

Free models differ a lot in what they refuse: the same "rank emojis by penis
size" banter gets played along with by one and "nah, i'm not doing that" from
the next. A short reply that is only a refusal is worth one retry on a different
model; long replies that merely contain "i can't" are real answers and pass.
"""

from __future__ import annotations

import re

# Refusals are short. Anything longer is treated as an answer that happens to
# include refusal-ish wording.
MAX_REFUSAL_CHARS = 280

_REFUSAL_RE = re.compile(
    r"""
    i[’']?m\s+sorry,?\s+but\s+i\s+can[’']?t
    | \bi\s+can[’']?t\s+(help|assist)\s+with\s+(that|this)
    | \bi[’']?m\s+not\s+able\s+to\s+(answer|help|do|assist)
    | \bi\s+(cannot|can[’']?t|won[’']?t)\s+(answer|help|assist|do|write|make|create|generate|engage|provide|rank|roast)\b
    | \bi[’']?m\s+not\s+(doing|writing|making|gonna\s+(do|write|make|rank|roast|go\s+there)|going\s+to\s+(do|write|make|help))\b
    | \bnot\s+doing\s+that\b
    | \bi\s+don[’']?t\s+do\s+(that|this|those)\b
    | \blet[’']?s\s+keep\s+it\s+(clean|respectful|cool|civil|pg)\b
    | \bkeep\s+it\s+(clean|respectful|civil)\b
    | \bcan[’']?t\s+do\s+that\b
    | \bnot\s+the\s+move\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_refusal(text: str | None) -> bool:
    """True when ``text`` is a short reply whose point is declining the request."""
    body = (text or "").strip()
    if not body or len(body) > MAX_REFUSAL_CHARS:
        return False
    return bool(_REFUSAL_RE.search(body))
