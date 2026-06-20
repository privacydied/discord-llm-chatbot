import re

_LEADING_MODE_PREAMBLE_RE = re.compile(
    r"""
    \A
    (?:
        [\s\u200b\u200c\u200d\ufeff]*
        [>*_`~\s]*
        mode
        [\s*_`~]*
        [:=\-\u2013\u2014\u2015]
        [\s*_`~]*
        (?:normal|political)
        [\s*_`~.!]*
        (?:\r?\n|$)
    )+
    """,
    re.IGNORECASE | re.VERBOSE,
)


def strip_leading_mode_preamble(text: str) -> str:
    if not text:
        return text
    cleaned = _LEADING_MODE_PREAMBLE_RE.sub("", text)
    return cleaned.lstrip("\r\n \t\u200b\u200c\u200d\ufeff")
