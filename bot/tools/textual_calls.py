"""Tool calls a model wrote as plain text instead of using the tool API.
[REH][IV][SFT].

Some free models (nemotron, qwen-style templates) sometimes answer with the
call itself as text. Formats seen live::

    <function=web_search> <parameter=query> poland crime rate </parameter> </function>
    <function=web_search> <query>uk bmi stats</query> </function> </functions>
    <invoke name="web_search"><parameter name="query">q</parameter></invoke>
    <tool_call>{"name": "web_search", "arguments": {"query": "q"}}</tool_call>
    {"tool": "search", "query": "uk bmi stats"}

Sent as-is, that markup lands in Discord. This module finds those calls so the
caller can run them, and strips the markup from anything shown to users.
Only registered tool names are returned (after mapping a few invented
aliases); arguments still go through ``execute_tool`` validation like any
other call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MAX_TEXTUAL_CALLS = 3

# Names models invent for registered tools. [IV]
_ALIASES = {
    "search": "web_search",
    "search_web": "web_search",
    "websearch": "web_search",
    "google_search": "web_search",
    "browse": "read_webpage",
    "fetch_url": "read_webpage",
    "open_url": "read_webpage",
}

# <function=name> <parameter=key> value </parameter> ... </function>
_FUNCTION_RE = re.compile(r"<function=([A-Za-z_][\w.-]{0,63})>(.*?)(?:</function>|$)", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z_]\w{0,63})>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)
# <query>value</query> inside a <function=...> block.
_BARE_PARAM_RE = re.compile(r"<([A-Za-z_]\w{0,63})>(.*?)</\1>", re.DOTALL)
# <invoke name="name"> <parameter name="key">value</parameter> </invoke>
_INVOKE_RE = re.compile(r"<invoke\s+name=[\"']([A-Za-z_][\w.-]{0,63})[\"']\s*>(.*?)(?:</invoke>|$)", re.DOTALL)
_NAMED_PARAM_RE = re.compile(r"<parameter\s+name=[\"']([A-Za-z_]\w{0,63})[\"']\s*>(.*?)(?:</parameter>|$)", re.DOTALL)
# <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)
# A whole line that is a JSON object naming a tool: {"tool": "search", ...}
_JSON_LINE_RE = re.compile(r"^[ \t]*(\{[^\n]*\"(?:tool|name|function)\"\s*:[^\n]*\})[ \t]*$", re.MULTILINE)
# Wrapper/stray tags that come with any style.
_STRAY_TAG_RE = re.compile(
    r"</?(?:invoke|functions|function_calls|tool_call|tool_calls)\b[^>]*>|</?parameter[^>]*>|</function>",
    re.IGNORECASE,
)
_JSON_NAME_KEYS = ("name", "tool", "function")
_JSON_ARG_KEYS = ("arguments", "parameters", "args", "input")


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
        body = match.group(2)
        pairs = _PARAMETER_RE.findall(body) or _BARE_PARAM_RE.findall(body)
        calls.append(TextualCall(match.group(1), {key: _coerce(val) for key, val in pairs}))
    return calls


def _from_invoke_tags(content: str) -> list[TextualCall]:
    calls = []
    for match in _INVOKE_RE.finditer(content):
        pairs = _NAMED_PARAM_RE.findall(match.group(2))
        if pairs:  # an empty <invoke> only wraps a <function=...> block parsed above
            calls.append(TextualCall(match.group(1), {key: _coerce(val) for key, val in pairs}))
    return calls


def _call_from_json(data: Any) -> TextualCall | None:
    if not isinstance(data, dict):
        return None
    name = next((data[k] for k in _JSON_NAME_KEYS if isinstance(data.get(k), str)), None)
    if not name:
        return None
    args = next((data[k] for k in _JSON_ARG_KEYS if k in data), None)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if args is None:  # flat form: {"tool": "search", "query": "..."}
        args = {k: v for k, v in data.items() if k not in _JSON_NAME_KEYS}
    return TextualCall(name, args if isinstance(args, dict) else {})


def _from_json(pattern: re.Pattern[str], content: str) -> list[TextualCall]:
    calls = []
    for match in pattern.finditer(content):
        try:
            call = _call_from_json(json.loads(match.group(1)))
        except ValueError:
            continue
        if call:
            calls.append(call)
    return calls


def find_textual_calls(content: str | None, allowed: set[str] | frozenset[str]) -> list[TextualCall]:
    """Registered tool calls written as text in ``content`` (at most a few)."""
    if not content or ("<" not in content and "{" not in content):
        return []
    calls = _from_function_tags(content) + _from_invoke_tags(content) + _from_json(_TOOL_CALL_JSON_RE, content) + _from_json(_JSON_LINE_RE, content)
    for call in calls:
        call.name = _ALIASES.get(call.name, call.name)
    return [c for c in calls if c.name in allowed][:MAX_TEXTUAL_CALLS]


def _is_json_tool_line(match: re.Match[str]) -> str:
    """Blank a JSON line that calls a registered tool; keep any other JSON. [IV]

    A bare JSON object can be a legitimate answer ("give me that as JSON"), so
    only lines naming one of our tools count as leaked calls.
    """
    from .registry import ALLOWED_TOOL_NAMES

    try:
        call = _call_from_json(json.loads(match.group(1)))
    except ValueError:
        return match.group(0)
    if call and _ALIASES.get(call.name, call.name) in ALLOWED_TOOL_NAMES:
        return ""
    return match.group(0)


def strip_textual_calls(content: str | None) -> str:
    """``content`` with any written-out tool-call markup removed."""
    if not content:
        return ""
    text = _FUNCTION_RE.sub("", content)
    text = _INVOKE_RE.sub("", text)
    text = _TOOL_CALL_JSON_RE.sub("", text)
    text = _JSON_LINE_RE.sub(_is_json_tool_line, text)
    text = _STRAY_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def has_textual_call_markup(content: str | None) -> bool:
    if not content:
        return False
    if _FUNCTION_RE.search(content) or _INVOKE_RE.search(content) or _TOOL_CALL_JSON_RE.search(content):
        return True
    return any(_is_json_tool_line(m) == "" for m in _JSON_LINE_RE.finditer(content))
