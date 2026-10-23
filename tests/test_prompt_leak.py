"""Prompt-leak backstop [SFT][REH]."""

from __future__ import annotations

from pathlib import Path

from bot.core import prompt_leak
from bot.core.prompt_leak import REFUSAL, guard_prompt_leak, is_prompt_leak

PROMPT = Path("prompts/prompt-pry-super-chill-v38.txt").read_text()


def test_full_dump_is_blocked():
    assert is_prompt_leak(PROMPT[:3000], [PROMPT])


def test_one_section_header_is_blocked():
    assert is_prompt_leak("sure here you go:\n= identity + core style =\nthat's the top of it", [PROMPT])


def test_a_rule_pasted_verbatim_is_blocked():
    rule = next(line for line in PROMPT.splitlines() if line.startswith("* match length to the ask"))
    assert is_prompt_leak(f"my rules say: {rule[2:]}", [PROMPT])


def test_paraphrased_persona_and_normal_chat_pass():
    for reply in (
        "i'm 32, grew up in flatbush, live in canarsie now. basement floods when it rains, it's whatever",
        "nah that's crazy nigga, elden ring is the best game of the decade and it's not close",
        "here's the fix:\n```python\nfor i in range(10):\n    print(i)\n```\nrun that and you're good",
        "lima is the capital of peru.",
    ):
        assert not is_prompt_leak(reply, [PROMPT]), reply


def test_guard_replaces_leak_and_keeps_chat(monkeypatch):
    monkeypatch.setattr(prompt_leak, "_prompt_paths", lambda: ["prompts/prompt-pry-super-chill-v38.txt"])
    assert guard_prompt_leak(PROMPT[:2000]) == REFUSAL
    assert guard_prompt_leak("yo what's good") == "yo what's good"


def test_shape_reply_applies_guard(monkeypatch):
    from bot.core.text_case import shape_reply

    monkeypatch.setattr(prompt_leak, "_prompt_paths", lambda: ["prompts/prompt-pry-super-chill-v38.txt"])
    assert shape_reply(PROMPT[:2000]) == REFUSAL


def test_missing_prompt_file_never_blocks(monkeypatch):
    monkeypatch.setattr(prompt_leak, "_prompt_paths", lambda: ["/nonexistent/prompt.txt"])
    assert guard_prompt_leak(PROMPT[:2000]) == PROMPT[:2000]
