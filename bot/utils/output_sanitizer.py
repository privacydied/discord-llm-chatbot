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
        (?:normal|political|contradiction)
        [\s*_`~.!]*
        (?:\r?\n|$)
    )+
    """,
    re.IGNORECASE | re.VERBOSE,
)

_COMPACT_AB_MODE_RE = re.compile(
    r"""
    \A
    [>*_`~\s]*
    A:\s*(?:true|false)
    [\s]+
    B:\s*(?:true|false)
    [\s]+
    MODE:\s*(?:normal|political|contradiction)
    (?:
        [\s]+
        (.+)
    )?
    [\s]*
    \Z
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _strip_wrappers(line: str) -> str:
    stripped = line.strip()
    stripped = stripped.lstrip(">").strip()
    while stripped.startswith("`") and stripped.endswith("`") and len(stripped) > 2:
        stripped = stripped[1:-1].strip()
    while stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
        stripped = stripped[2:-2].strip()
    stripped = stripped.strip("*_~").strip()
    return stripped


def _is_mode_line(line: str) -> bool:
    return bool(
        re.match(
            r"^mode:\s*(?:normal|political|contradiction)$",
            _strip_wrappers(line),
            re.IGNORECASE,
        )
    )


def _strip_leading_ab_mode_diagnostics(text: str) -> str:
    if not text:
        return text

    lines = text.splitlines()
    stripped_lines = [_strip_wrappers(line) for line in lines]

    if len(stripped_lines) < 3:
        return text

    if not re.match(r"^a:\s*(?:true|false)$", stripped_lines[0], re.IGNORECASE):
        return text
    if not re.match(r"^b:\s*(?:true|false)$", stripped_lines[1], re.IGNORECASE):
        return text

    mode_start = 2
    idx = mode_start
    while idx < len(stripped_lines) and _is_mode_line(stripped_lines[idx]):
        idx += 1

    if idx == mode_start:
        return text

    body_lines = lines[idx:]
    body_text = "\n".join(body_lines)

    if body_text.strip():
        return body_text.lstrip("\r\n \t\u200b\u200c\u200d\ufeff")

    return "\n".join(lines[:2]).rstrip("\r\n \t\u200b\u200c\u200d\ufeff")


def _strip_compact_ab_mode(text: str) -> str:
    m = _COMPACT_AB_MODE_RE.match(text)
    if not m:
        return text

    body = m.group(1)
    if body:
        return body.strip()

    ab_m = re.match(
        r"^[>*_`~\s]*(A:\s*(?:true|false)\s+B:\s*(?:true|false))",
        text,
        re.IGNORECASE,
    )
    if ab_m:
        return ab_m.group(1)

    return text


def strip_leading_mode_preamble(text: str) -> str:
    if not text:
        return text

    result = _strip_compact_ab_mode(text)
    result = _strip_leading_ab_mode_diagnostics(result)
    cleaned = _LEADING_MODE_PREAMBLE_RE.sub("", result)
    return cleaned.lstrip("\r\n \t\u200b\u200c\u200d\ufeff")
