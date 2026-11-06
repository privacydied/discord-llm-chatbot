"""Regression: the user's caption must reach the VL call, not only the text stage."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bot.router import Router


def _make_router() -> Router:
    router = object.__new__(Router)
    router.logger = MagicMock()
    router._get_system_prompt = lambda key, default=None: "GENERIC VL PROMPT" if key == "vl_prompt" else (default or "")
    return router


@pytest.mark.asyncio
async def test_user_caption_is_forwarded_to_vl_prompt(monkeypatch) -> None:
    import bot.brain as brain_mod
    import bot.see as see_mod

    vl_prompts: list[str] = []
    text_inputs: list[str] = []

    async def fake_see_infer(image_path, prompt=None, model_override=None):
        vl_prompts.append(prompt)
        return SimpleNamespace(content="a pint glass shattering; headline reads 'ANDY BURNHAM TO BAN ...'", error=None)

    async def fake_brain_infer(text_input, context=None, **_kwargs):
        text_inputs.append(text_input)
        return "ok"

    monkeypatch.setattr(see_mod, "see_infer", fake_see_infer)
    monkeypatch.setattr(brain_mod, "brain_infer", fake_brain_infer)

    router = _make_router()
    await router._unified_vl_to_text_pipeline(["/tmp/x.jpg"], "what does the headline say?", "what does the headline say?")

    assert vl_prompts, "VL was never called"
    assert "what does the headline say?" in vl_prompts[0]
    assert "GENERIC VL PROMPT" in vl_prompts[0]
    assert "User message: what does the headline say?" in text_inputs[0]


@pytest.mark.asyncio
async def test_no_caption_keeps_generic_vl_prompt(monkeypatch) -> None:
    import bot.brain as brain_mod
    import bot.see as see_mod

    vl_prompts: list[str] = []

    async def fake_see_infer(image_path, prompt=None, model_override=None):
        vl_prompts.append(prompt)
        return SimpleNamespace(content="a cat", error=None)

    async def fake_brain_infer(text_input, context=None, **_kwargs):
        return "ok"

    monkeypatch.setattr(see_mod, "see_infer", fake_see_infer)
    monkeypatch.setattr(brain_mod, "brain_infer", fake_brain_infer)

    router = _make_router()
    await router._unified_vl_to_text_pipeline(["/tmp/x.jpg"], "", "Thoughts?")

    assert vl_prompts == ["GENERIC VL PROMPT"]
