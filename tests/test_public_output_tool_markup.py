"""Tool calls written as text must never reach Discord. [REH][IV]

The plain-text router route has a guard, but image/X-image routes call
``brain_infer`` directly. The outbound sanitizer is the only boundary shared
by every route, so the markup must be removed there.
"""

from bot.public_output import SAFE_FALLBACK_MESSAGE, sanitize_public_text
from bot.tools.textual_calls import has_textual_call_markup

LIVE_LEAK = "<function=search_web> <parameter=query> dario amodei wife cami epstein files eric schmidt </parameter> </function> </invoke>"


def test_live_nemotron_leak_is_stripped_at_send_boundary() -> None:
    cleaned = sanitize_public_text(LIVE_LEAK)
    assert cleaned == SAFE_FALLBACK_MESSAGE
    assert not has_textual_call_markup(cleaned)


def test_markup_with_surrounding_answer_keeps_the_answer() -> None:
    cleaned = sanitize_public_text(f"Here is what I found.\n\n{LIVE_LEAK}")
    assert cleaned == "Here is what I found."
    assert not has_textual_call_markup(cleaned)


def test_json_tool_call_markup_is_stripped() -> None:
    raw = 'Sure. <tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
    cleaned = sanitize_public_text(raw)
    assert cleaned == "Sure."
    assert not has_textual_call_markup(cleaned)


def test_plain_text_with_comparison_untouched() -> None:
    raw = "poland's crime rate is lower, 1 < 2"
    assert sanitize_public_text(raw) == raw
