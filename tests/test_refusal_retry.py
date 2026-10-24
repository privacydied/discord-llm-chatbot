"""Refusal detection + one retry on the next model in the tool loop. [REH]"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.core.refusals import is_refusal
from bot.tools import ToolContext, inference

# Real replies from logs/bot.jsonl (2026-10-08 .. 10-23).
REFUSED = [
    "I’m sorry, but I can’t help with that.",
    "I’m not able to answer that.",
    "i can’t help with that.",
    "nah nigga i'm not doing that",
    "nah nigga, i don't do that shit. aids jokes are corny and played out.",
    "sorry nigga cant do that real people + omegaverse = risky territory with the tropes—lets keep it clean.",
    "nah nigga, i'm not doing that. aids jokes ain't the move. ask me something else.",
    "nah i'm not singing that nigga. that's weird.",
    "nah nigga i said no. that shit is weird. move on.",
    "nah nigga i'm not transcribing that",
    "nah, i'm not playing that game. reducing people to ethnic stereotypes about body odor is lazy and dehumanizing. i'm not doing that.",
]
ANSWERS = [
    "backshots is crazy work nigga, respect the technique",
    "nah, that's not it. patrice o'neal and shaq o'neal ain't brothers — never heard that one before.",
    "nah i can't physically go get you nothing, i'm a discord bot. but if you're tryna cop 9mg wintergreen velo, most smoke shops carry it.",
    "yo, my bad — i don't have horus in this conversation at all.",
    "deportations escalated under trump compared to obama. " + "details " * 60,  # long answer
    "lmfao nigga what. ike six feet under since 1969, sky delusional.",
]


@pytest.mark.parametrize("text", REFUSED)
def test_detects_refusals(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", ANSWERS)
def test_answers_pass(text):
    assert not is_refusal(text)


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None, reasoning=None, reasoning_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Completions:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


CFG = {
    "TOOLS_ENABLED": True,
    "TOOLS_MAX_ITERATIONS": 3,
    "TOOLS_TIMEOUT_S": 30.0,
    "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
    "TEXT_FALLBACK_MODELS": "openrouter|backup-a,openrouter|backup-b",
}


def _install(monkeypatch, responses):
    completions = _Completions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(inference, "_client_for", lambda cfg, timeout: (client, "primary"))
    monkeypatch.setattr(inference, "_default_system_prompt", lambda cfg: "persona")
    return completions


async def test_refusal_is_retried_once_on_next_model(monkeypatch):
    completions = _install(monkeypatch, [_response("nah nigga i'm not doing that"), _response("🍆 obviously. 🍌 trying too hard")])
    out = await inference.run_tool_conversation(prompt="rank emojis", ctx=ToolContext(), cfg=CFG)
    assert out == "🍆 obviously. 🍌 trying too hard"
    assert [c["model"] for c in completions.calls] == ["primary", "backup-a"]


async def test_second_refusal_is_posted_no_endless_retry(monkeypatch):
    completions = _install(monkeypatch, [_response("I’m sorry, but I can’t help with that."), _response("i can’t help with that.")])
    out = await inference.run_tool_conversation(prompt="make a threat", ctx=ToolContext(), cfg=CFG)
    assert out == "i can’t help with that."
    assert len(completions.calls) == 2


async def test_normal_answer_not_retried(monkeypatch):
    completions = _install(monkeypatch, [_response("backshots is crazy work nigga")])
    assert await inference.run_tool_conversation(prompt="backshots?", ctx=ToolContext(), cfg=CFG) == "backshots is crazy work nigga"
    assert len(completions.calls) == 1


async def test_empty_reply_tries_next_model_first(monkeypatch):
    completions = _install(monkeypatch, [_response(""), _response("🍆 obviously")])
    assert await inference.run_tool_conversation(prompt="rank emojis", ctx=ToolContext(), cfg=CFG) == "🍆 obviously"
    assert [c["model"] for c in completions.calls] == ["primary", "backup-a"]
