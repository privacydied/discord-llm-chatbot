"""Tool calls a model wrote as plain text instead of using the tool API.
[REH][IV][SFT].

Some free models (nemotron, qwen-style templates) sometimes answer with the
call itself as text, e.g.::

    <function=web_search> <parameter=query> poland crime rate </parameter> </function>

Sent as-is, that markup lands in Discord. This module finds those calls so the
caller can run them, and strips the markup from anything shown to users.
Only registered tool names are returned; arguments still go through
``execute_tool`` validation like any other call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MAX_TEXTUAL_CALLS = 3

# <function=name> <parameter=key> value </parameter> ... </function>
_FUNCTION_RE = re.compile(r"<function=([A-Za-z_][\w.-]{0,63})>(.*?)(?:</function>|$)", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z_]\w{0,63})>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)
# <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)
# Wrapper/stray tags that come with either style.
_STRAY_TAG_RE = re.compile(
    r"</?(?:invoke|function_calls|tool_call|tool_calls)\b[^>]*>|</?parameter[^>]*>|</function>",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TextualCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def _coerce(value: str) -> Any:
    """Parameter values arrive as text; numbers/JSON literals are decoded."""
    text = value.strip()
    if (text and text[0] in "[{0123456789-") or text in ("true", "false", "null"):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def _from_function_tags(content: str) -> list[TextualCall]:
    calls = []
    for match in _FUNCTION_RE.finditer(content):
        args = {key: _coerce(val) for key, val in _PARAMETER_RE.findall(match.group(2))}
        calls.append(TextualCall(match.group(1), args))
    return calls


def _from_json_tags(content: str) -> list[TextualCall]:
    calls = []
    for match in _TOOL_CALL_JSON_RE.finditer(content):
        try:
            data = json.loads(match.group(1))
        except ValueError:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            continue
        args = data.get("arguments") or data.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        calls.append(TextualCall(data["name"], args if isinstance(args, dict) else {}))
    return calls


def find_textual_calls(content: str | None, allowed: set[str] | frozenset[str]) -> list[TextualCall]:
    """Registered tool calls written as text in ``content`` (at most a few)."""
    if not content or "<" not in content:
        return []
    calls = _from_function_tags(content) + _from_json_tags(content)
    return [c for c in calls if c.name in allowed][:MAX_TEXTUAL_CALLS]


def strip_textual_calls(content: str | None) -> str:
    """``content`` with any written-out tool-call markup removed."""
    if not content:
        return ""
    text = _FUNCTION_RE.sub("", content)
    text = _TOOL_CALL_JSON_RE.sub("", text)
    text = _STRAY_TAG_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def has_textual_call_markup(content: str | None) -> bool:
    if not content:
        return False
    return bool(_FUNCTION_RE.search(content) or _TOOL_CALL_JSON_RE.search(content))
