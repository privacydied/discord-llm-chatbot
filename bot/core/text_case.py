"""Persona text shaping: lowercase and line-break collapsing that leave code, URLs and Discord tokens intact."""

import re

# Spans whose case carries meaning and must survive untouched [IV]:
# fenced code, inline code, URLs, and Discord tokens (<@id>, <#id>, <:Emoji:id>, <t:...>).
_PROTECTED_RE = re.compile(
    r"```.*?```"  # fenced code block
    r"|`[^`\n]+`"  # inline code
    r"|https?://\S+"  # bare URL
    r"|<[@#:a!t][^>\s]*>",  # Discord mention / channel / emoji / timestamp token
    re.DOTALL,
)


def lowercase_outside_protected(text: str) -> str:
    """Lowercase ``text`` everywhere except code, URLs, and Discord tokens."""
    if not text:
        return text
    parts: list[str] = []
    last = 0
    for match in _PROTECTED_RE.finditer(text):
        parts.append(text[last : match.start()].lower())
        parts.append(match.group(0))
        last = match.end()
    parts.append(text[last:].lower())
    return "".join(parts)


_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# A line that starts a list item or numbered step keeps its own line. [IV]
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NEWLINES_RE = re.compile(r"[ \t]*\n[\s]*")


def _collapse_prose(chunk: str) -> str:
    """Join lines with a space, except before a list item, which keeps one newline."""

    def join(match: re.Match[str]) -> str:
        rest = chunk[match.end() :]
        return "\n" if _LIST_LINE_RE.match(rest) else " "

    return _NEWLINES_RE.sub(join, chunk)


def collapse_line_breaks_outside_code(text: str) -> str:
    """Make a reply one chat paragraph: line breaks become spaces outside fenced code; lists keep their lines."""
    if not text or "\n" not in text:
        return text
    segments: list[str] = []
    last = 0
    for match in _FENCE_RE.finditer(text):
        segments.append(_collapse_prose(text[last : match.start()]).strip())
        segments.append(match.group(0))
        last = match.end()
    segments.append(_collapse_prose(text[last:]).strip())
    # A code block sits on its own lines; prose around it joins with one newline.
    return "\n".join(seg for seg in segments if seg)
