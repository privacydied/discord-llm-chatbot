"""Persona casing: force replies to lowercase without corrupting case-sensitive spans."""

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
