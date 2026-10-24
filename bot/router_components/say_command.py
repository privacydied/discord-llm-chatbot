"""Natural-language "say this" -> read the text out loud, word for word. [CA]

``!say <text>`` already speaks any text for anyone. Asking in words ("yo say
this: ...", "@ronin say "..."", replying to a message with "say this") went to
the LLM instead, which treats repeating as endorsing and refuses ("nah i'm not
saying that, that's weird"). This is the same command, spelled naturally, so it
is handled in code like ``!say``: the text is spoken verbatim, no model call.

Deliberately strict: only an explicit marker counts (a colon, quotes, or
"say this/that/it" with nothing after it on a reply). "say why you like x" or
"can you say what happened" stay normal chat. Spoiler bars around the text are
ignored (||"..."||). On a reply, a short bare "say <words>" also counts
("say nigger" replying to the bot, 2026-10-24) unless it opens with a chat or
slang word ("say less", "say type shit", "say hi to x", "say a swear word").
"""

from __future__ import annotations

import re

MAX_SAY_CHARS = 1000

_LEAD = r"(?:(?:yo|ayo|ay|aye|hey|oi|bro|bruh|ronin|pls|plz|please|now|and|just|can\s+(?:you|u)|could\s+(?:you|u)|will\s+(?:you|u))[\s,]+)*"
_VERB = r"(?:says?|read|repeat|speak)"
_OBJ = r"(?:\s+(?:this|that|it|after\s+me|the\s+following))?(?:\s+(?:word\s+(?:by|for)\s+word|exactly|verbatim))?"
_HOW = r"(?:\s+(?:out\s*loud|aloud|in\s+(?:a\s+)?(?:vn|voice\s*note|voice\s*message|voice|audio)))?"

# "say: text", "say this: text", "say this out loud - text"
_COLON_RE = re.compile(rf"^\s*{_LEAD}{_VERB}{_OBJ}{_HOW}\s*[:\-–—]\s*(?P<text>.+)$", re.IGNORECASE | re.DOTALL)
# 'say "text"' / "say this “text”"
_QUOTED_RE = re.compile(
    rf"^\s*{_LEAD}{_VERB}{_OBJ}{_HOW}\s+(?P<q>[\"“”'‘’])(?P<text>.+?)[\"“”'‘’]?\s*[.!?]*\s*$",
    re.IGNORECASE | re.DOTALL,
)
# "say this" / "read that out loud" with nothing after (reply to the text)
_BARE_RE = re.compile(rf"^\s*{_LEAD}{_VERB}\s+(?:this|that|it){_HOW}\s*[.!?]*\s*$", re.IGNORECASE)

# bare "say <words>" on a reply: short, and not opening with a chat/slang word
_BARE_SAY_RE = re.compile(r"^\s*say\s+(?P<text>\S.*)$", re.IGNORECASE | re.DOTALL)
BARE_SAY_MAX_WORDS = 8
_CHAT_FIRST_WORDS = frozenset(
    "a an the some sum something smth somethin anything nothing less type word more no yes yeah nah "
    "hi hello hey bye goodbye goodnight sorry thanks thank what why how who when where which whether if "
    "your ur my his her their our it's its that's ya you u me him them us so".split()
)
_SPOILER_RE = re.compile(r"\|\|")
_MENTION_RE = re.compile(r"<@[!&]?\d+>|<#\d+>")
_WRAP_QUOTES = "\"“”'‘’"


def _tidy(text: str) -> str:
    text = _MENTION_RE.sub("", text or "").strip()
    text = text.lstrip(_WRAP_QUOTES).rstrip(_WRAP_QUOTES).strip()  # also an unclosed quote
    return " ".join(text.split())[:MAX_SAY_CHARS]


def _bare_say(body: str) -> str:
    match = _BARE_SAY_RE.match(body)
    if not match:
        return ""
    words = match.group("text").split()
    if len(words) > BARE_SAY_MAX_WORDS or words[0].lower().strip(",.!?") in _CHAT_FIRST_WORDS:
        return ""
    if re.search(r"\bin\s+[a-z]+\s*[.!?]*$", match.group("text"), re.IGNORECASE):
        return ""  # "say hello in french": a translation ask
    return _tidy(match.group("text"))


def parse_say_command(text: str | None, *, is_reply: bool = False) -> tuple[bool, str]:
    """(is_say_command, inline_text). Empty inline text + True = speak the replied-to message."""
    body = _SPOILER_RE.sub(" ", text or "").strip()
    if not body:
        return False, ""
    for pattern in (_COLON_RE, _QUOTED_RE):
        match = pattern.match(body)
        if match:
            spoken = _tidy(match.group("text"))
            return (True, spoken) if spoken else (False, "")
    if _BARE_RE.match(body):
        return True, ""
    if is_reply:
        spoken = _bare_say(body)
        if spoken:
            return True, spoken
    return False, ""


def referenced_text(message_text: str | None) -> str:
    return _tidy(message_text or "")
