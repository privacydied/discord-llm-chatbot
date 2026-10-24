"""Detect "is this AI?"-type questions so the detector runs deterministically. [CA]

Left to the model, "@bot is this AI?" on a tweet link could go down the X
syndication route (which ignores quote-tweet media) and the model then answered
from the tweet text without calling ``check_ai_generated``. An explicit AI-check
question is routed straight to the detector instead.

Strict on purpose: the whole message must be the question. "AI", "generated",
"photoshopped", "edited", "cgi" always count; "fake"/"real" alone only with a
picture word or as "real or fake" ("is this real?" usually means "is it true").
"""

from __future__ import annotations

import re

_LEAD = r"(?:(?:yo|ayo|ay|aye|bro|bruh|ronin|pls|please|ok|okay|wait|so|and|hey|lol|lmao|nah|deadass|fr)[\s,]+)*"
_SUBJ = r"(?:(?:is|are|was|r|iz)\s+)?(?:(?:this|that|it|these|those|dis|dat|the)\s+)?"
_NOUN = r"(?:pic|pics|picture|image|img|photo|photos|video|vid|clip|tweet|post|one|shit)"
_AI = r"(?:ai|a\.i\.?|ai[\s-]?gen(?:erated)?|ai[\s-]?made|generated|photoshopped|photoshop|edited|cgi|deepfake|deep\s*fake)"
_END = r"\s*(?:or\s+(?:real|nah|not|fake|what)|or\s+nah)?\s*(?:fr|tho|though|lol|lmao|bro|💀|😭|\?|!|\.)*\s*$"

_AI_RE = re.compile(rf"^\s*{_LEAD}{_SUBJ}(?:{_NOUN}\s+)?(?:a\s+|an\s+)?{_AI}(?:\s+{_NOUN})?{_END}", re.IGNORECASE)
_PIC_FAKE_RE = re.compile(rf"^\s*{_LEAD}{_SUBJ}{_NOUN}\s+(?:fake|real){_END}", re.IGNORECASE)
_REAL_OR_FAKE_RE = re.compile(
    rf"^\s*{_LEAD}{_SUBJ}(?:{_NOUN}\s+)?(?:real\s+or\s+(?:fake|ai|nah|generated)|(?:fake|ai)\s+or\s+real){_END}",
    re.IGNORECASE,
)
_LOOKS_AI_RE = re.compile(rf"^\s*{_LEAD}{_SUBJ}(?:{_NOUN}\s+)?(?:look|looks|lookin|looking)\s+(?:like\s+)?{_AI}{_END}", re.IGNORECASE)
# Looser second rule for natural sentences ("are the images in this tweet ai or
# real", "is the girl in this pic ai"): an AI word + a pointer at the media + a
# question cue. A false hit is cheap -- with no link/image the caller falls
# back to ordinary chat.
_AI_WORD_RE = re.compile(rf"\b{_AI}\b", re.IGNORECASE)
_POINTER_RE = re.compile(
    r"\b(?:this|that|these|those|it|pics?|pictures?|images?|imgs?|photos?|videos?|vids?|clips?|tweets?|posts?|"
    r"person|girl|guy|woman|man|her|him|she|he)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\?|^\s*(?:\w+\s+){0,4}?(?:is|are|check|tell\s+me|if|whether)\b|\breal\s+or\b|\bor\s+real\b", re.IGNORECASE)
LOOSE_MAX_CHARS = 160
_MENTION_RE = re.compile(r"<@[!&]?\d+>")
_URL_RE = re.compile(r"https?://\S+")


def is_ai_check_question(text: str | None) -> bool:
    """True when the message (mentions and links removed) is only an AI-check question."""
    body = _URL_RE.sub(" ", _MENTION_RE.sub(" ", text or ""))
    body = " ".join(body.split())
    if not body:
        return False
    if len(body) <= 80 and any(p.match(body) for p in (_AI_RE, _PIC_FAKE_RE, _REAL_OR_FAKE_RE, _LOOKS_AI_RE)):
        return True
    return len(body) <= LOOSE_MAX_CHARS and bool(_AI_WORD_RE.search(body) and _POINTER_RE.search(body) and _QUESTION_RE.search(body))
