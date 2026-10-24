"""'@bot is this AI?' runs the detector in code, then the model phrases the scores. [CA]"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.router_components.ai_check_intent import is_ai_check_question
from bot.tools.types import ToolResult


@pytest.mark.parametrize(
    "text",
    [
        "<@1392310898311303259> is this AI?",  # live 2026-10-24
        "is that AI generated",
        "is this pic fake",
        "this real or nah?",
        "real or fake?",
        "is it photoshopped",
        "yo is this ai fr",
        "is this an ai image",
        "this looks ai generated 💀",
        "is this a deepfake",
        "is this AI? https://x.com/a/status/2113707791609184428",
    ],
)
def test_ai_check_questions(text):
    assert is_ai_check_question(text)


@pytest.mark.parametrize(
    "text",
    ["is this real?", "is this true", "what is ai", "is ai gonna take our jobs", "ai is mid", "fake", "is chatgpt better than gemini", "who made this"],
)
def test_not_ai_check_questions(text):
    assert not is_ai_check_question(text)


def _router():
    from bot.router import Router

    router = Router.__new__(Router)
    router.bot = SimpleNamespace(user=SimpleNamespace(id=9))
    router.config = {}
    router.logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)

    @asynccontextmanager
    async def _typing(_channel):
        yield

    router._optional_typing = _typing
    return router


MSG = SimpleNamespace(id=1, content="<@9> is this AI?", channel=None, reference=None)


async def test_routes_to_detector_and_persona_reply(monkeypatch):
    check = AsyncMock(return_value=ToolResult.success("AI-image detector results (Sightengine):\n- photo 1: 97% AI probability -> very likely AI-generated\n\nguidance"))
    answer = AsyncMock(return_value="Detector Says 97% AI, That's Fake Nigga")
    monkeypatch.setattr("bot.tools.builtins.ai_check.check_ai_generated", check)
    monkeypatch.setattr("bot.tools.inference.answer_from_context", answer)
    action = await _router()._maybe_ai_check(MSG, MSG.content)
    assert action.content == "detector says 97% ai, that's fake nigga"  # shape_reply lowercases
    assert "97% AI probability" in answer.await_args.kwargs["context"]


async def test_nothing_to_check_falls_through(monkeypatch):
    monkeypatch.setattr("bot.tools.builtins.ai_check.check_ai_generated", AsyncMock(return_value=ToolResult.failure("nothing to check: no link")))
    assert await _router()._maybe_ai_check(MSG, MSG.content) is None


async def test_model_failure_falls_back_to_plain_scores(monkeypatch):
    monkeypatch.setattr("bot.tools.builtins.ai_check.check_ai_generated", AsyncMock(return_value=ToolResult.success("AI-image detector results (Sightengine):\n- photo 1: 4% AI probability\n\nguidance")))
    monkeypatch.setattr("bot.tools.inference.answer_from_context", AsyncMock(return_value=None))
    action = await _router()._maybe_ai_check(MSG, MSG.content)
    assert "4% ai probability" in action.content and "guidance" not in action.content


async def test_non_question_is_ignored():
    assert await _router()._maybe_ai_check(MSG, "what do you think of this tweet") is None
