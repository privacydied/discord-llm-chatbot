"""Spot "reply in a voice note"-type asks so the answer is spoken without relying on the model. [CA]

Voice is off by default (TTS_GLOBAL_DEFAULT=false). The send_voice_note tool still
lets the model pick up odd phrasings; this catches the explicit ones in code.
"Say this: ..." is different (read verbatim) and handled by say_command.
"""

from __future__ import annotations

import re

_VOICE = r"(?:voice\s*(?:note|message|memo|msg|clip)s?|vn|vm|audio|voice)"
_PATTERNS = [
    # "reply/answer/respond/talk/tell me in a voice note", "send it as a vn"
    rf"\b(?:reply|respond|answer|talk|tell\s+me|explain|send\s+(?:it|that|this|me\s+that)?|say\s+it)\s+(?:back\s+)?(?:in|as|with|via|through)\s+(?:a\s+|an\s+)?{_VOICE}\b",
    # "send (me) a voice note", "drop a vn", "gimme a voice message", "record a voice note"
    rf"\b(?:send|drop|gimme|give\s+me|leave|record|do)\s+(?:me\s+|us\s+)?(?:a\s+|an\s+)?{_VOICE}\b",
    # "voice note this", "vn me", "voice note it"
    r"\b(?:voice\s*note|vn)\s+(?:this|that|it|me|us)\b",
    # "say it out loud", "speak your answer", "out loud pls"
    r"\b(?:say\s+it|answer|reply|tell\s+me)\s+(?:out\s*loud|aloud)\b",
    r"\buse\s+your\s+voice\b",
    r"\bin\s+(?:a\s+)?(?:voice\s*(?:note|message|memo))\b",
]
_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:don'?t|do\s+not|no|stop|never|without)\s+(?:\w+\s+){0,3}?(?:voice|vn|audio)", re.IGNORECASE)


def wants_voice_reply(text: str | None) -> bool:
    """True when the message explicitly asks for the answer as a voice note."""
    body = " ".join((text or "").split())
    if not body or _NEGATION_RE.search(body):
        return False
    return bool(_RE.search(body))
