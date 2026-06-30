"""Tests for the skip_history mechanism in bot.contextual_brain.

Regression coverage for: ambient replies inheriting unrelated channel
context. `contextual_brain_infer` already skips the rolling per-channel
buffer whenever `extra_context` is non-empty (`skip_history = bool(extra_context)`)
-- these tests lock in that contract from the ambient feature's perspective:
once an ambient-scoped local context is supplied as `extra_context`, the
rolling buffer (which may hold an unrelated prior conversation) must never
reach the final prompt sent to the model. They also pin down the inverse:
normal replies with no local context still get the rolling buffer, unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.contextual_brain import contextual_brain_infer_simple

TOPIC_A_HISTORY = "Alice: topic A -- let's talk about cars\nCarl: topic A -- I like sedans"
TOPIC_B_AMBIENT_CONTEXT = "[Message to respond to]\nBob: topic B -- what about pizza?"


def _mock_bot_with_rolling_buffer(history_text: str) -> MagicMock:
    bot = MagicMock()
    bot.user = MagicMock(id=999)
    ecm = MagicMock()
    ecm.get_context_for_user = MagicMock(return_value=[{"role": "user", "content": history_text}])
    ecm.format_context_string = MagicMock(return_value=history_text)
    ecm.get_contextual_response = AsyncMock(
        return_value=MagicMock(response_text="ok", used_history=[], fallback=False),
    )
    bot.enhanced_context_manager = ecm
    return bot


def _mock_message() -> MagicMock:
    message = MagicMock()
    message.id = 1
    message.guild = MagicMock(id=10)
    message.author = MagicMock(id=20)
    message.mentions = []
    message.reference = None
    message.channel = MagicMock()
    return message


@pytest.mark.asyncio
async def test_ambient_local_context_excludes_rolling_buffer() -> None:
    """extra_context populated (ambient path) => rolling buffer (topic A) must
    not reach the final prompt; the ambient context (topic B) must."""
    message = _mock_message()
    bot = _mock_bot_with_rolling_buffer(TOPIC_A_HISTORY)

    captured: dict[str, str] = {}

    async def fake_brain_infer(prompt, context="", system_prompt=None):
        captured["prompt"] = prompt
        return MagicMock(content="response")

    with patch("bot.contextual_brain.brain_infer", side_effect=fake_brain_infer):
        result = await contextual_brain_infer_simple(
            message,
            "topic B -- what about pizza?",
            bot,
            extra_context=TOPIC_B_AMBIENT_CONTEXT,
        )

    assert result == "ok"
    assert "topic A" not in captured["prompt"]
    assert "topic B" in captured["prompt"]
    assert "[Message to respond to]" in captured["prompt"]


@pytest.mark.asyncio
async def test_normal_reply_without_local_context_still_uses_rolling_buffer() -> None:
    """Regression: when there's no local/extra context (the normal non-ambient
    case this fix must not touch), the rolling buffer is still injected."""
    message = _mock_message()
    bot = _mock_bot_with_rolling_buffer(TOPIC_A_HISTORY)

    captured: dict[str, str] = {}

    async def fake_brain_infer(prompt, context="", system_prompt=None):
        captured["prompt"] = prompt
        return MagicMock(content="response")

    with patch("bot.contextual_brain.brain_infer", side_effect=fake_brain_infer):
        result = await contextual_brain_infer_simple(
            message,
            "what's up",
            bot,
            extra_context=None,
        )

    assert result == "ok"
    assert "topic A" in captured["prompt"]
    bot.enhanced_context_manager.format_context_string.assert_called_once()
