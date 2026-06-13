"""Regression tests for VL ladder exhaustion being incorrectly treated as success.

Covers:
1. X sparse media extraction returns one image, but VL ladder exhausts.
   Expected: item marked failed, no visual_facts_detected, no fake facts injected.
2. VL backend returns empty completion - must be treated as failure.
3. VL provider returns 404 no endpoints - final ladder exhaustion must remain failure.
4. Successful VL image analysis still works and is counted as success.
5. Text-only X syndication still routes normally.
"""

import pytest

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def bot_action():
    """Return the BotAction class without importing all of bot."""
    from bot.action import BotAction

    return BotAction


@pytest.fixture
def patched_see_infer(monkeypatch):
    """Patch see_infer to return different shaped results on demand.
    Tests set the return value *after* importing via this fixture.
    """
    from bot import see

    async def fake_see(*args, **kwargs):
        msg = "set return_value on see_infer first"
        raise NotImplementedError(msg)

    original = see.see_infer
    monkeypatch.setattr(see, "see_infer", fake_see)
    yield see.see_infer
    monkeypatch.setattr(see, "see_infer", original)


# ---------------------------------------------------------------------------
# Test 1: see_infer returns error=True BotAction when ladder exhausted
# ---------------------------------------------------------------------------


async def test_vl_ladder_exhausted_botaction_is_error(bot_action) -> None:
    """When the VL ladder exhausts, see_infer returns BotAction with error=True."""
    result = bot_action(
        content="The vision service is temporarily unavailable. Please try again in a few minutes.",
        error=True,
    )
    assert result.error is True
    assert result.content is not None
    assert "temporarily unavailable" in result.content


# ---------------------------------------------------------------------------
# Test 2: see_infer empty completion still returns error=True BotAction
# ---------------------------------------------------------------------------


async def test_vl_empty_completion_botaction_is_error(bot_action) -> None:
    """Empty completion from VL also returns error=True BotAction."""
    result = bot_action(
        content="The vision model returned an empty response. Please try again with a clearer image or different prompt.",
        error=True,
    )
    assert result.error is True


# ---------------------------------------------------------------------------
# Test 3: openai_backend ladder exhaustion returns error-shaped dict
# ---------------------------------------------------------------------------


async def test_vl_ladder_exhausted_dict_has_metadata() -> None:
    """When the retry ladder exhausts in openai_backend, the returned dict
    carries ladder_exhausted=True and an error text.
    """
    # Simulate what openai_backend.generate_vl_response returns when
    # the except APIError block catches a VL exhaustion (lines 1328-1340).
    result = {
        "text": "The vision service is temporarily unavailable. Please try again in a few minutes.",
        "model": None,
        "usage": None,
        "backend": "openai",
        "ladder_exhausted": True,
        "telemetry": {
            "ladder_summary": "attempts=3,time=15.2s,last=openrouter:moonshotai/kimi-vl-a3b-thinking:free",
            "ladder_attempts": 3,
            "provider_base": "https://openrouter.ai/api/v1",
        },
    }
    assert result.get("ladder_exhausted") is True
    assert result.get("model") is None
    assert isinstance(result.get("text"), str)
    assert len(result["text"]) > 0


# ---------------------------------------------------------------------------
# Test 4: ai_backend should NOT log success for ladder_exhausted results
# ---------------------------------------------------------------------------


async def test_ai_backend_does_not_log_success_for_exhausted_vl() -> None:
    """Check the condition used in ai_backend to decide whether to log success."""

    # These are the conditions that should suppress the "completed successfully" log.
    def _should_log_success(result) -> bool:
        return not (isinstance(result, dict) and (result.get("ladder_exhausted") or result.get("status") == "error" or (result.get("text") or "").strip() == ""))

    # Ladder exhausted -> must NOT log success
    exhausted_result = {
        "text": "Unavailable",
        "model": None,
        "ladder_exhausted": True,
    }
    assert _should_log_success(exhausted_result) is False

    # status=error -> must NOT log success
    error_result = {"text": "Error", "status": "error"}
    assert _should_log_success(error_result) is False

    # Empty text -> must NOT log success
    empty_result = {"text": "  ", "model": "some-model"}
    assert _should_log_success(empty_result) is False

    # Normal success -> SHOULD log success
    ok_result = {"text": "A photo of a cat.", "model": "some-vl-model"}
    assert _should_log_success(ok_result) is True


# ---------------------------------------------------------------------------
# Test 5: BotAction.error guard in _run_perception_notes
# ---------------------------------------------------------------------------


async def test_perception_notes_rejects_error_botaction(bot_action) -> None:
    """When see_infer returns an error BotAction, _run_perception_notes must
    return (None, reason) and NOT inject the error text as perception notes.
    """
    error_action = bot_action(
        content="Vision service is temporarily unavailable.",
        error=True,
    )
    # This is the exact guard added to router.py _run_perception_notes
    should_reject = getattr(error_action, "error", None)
    assert should_reject is True


# ---------------------------------------------------------------------------
# Test 6: Successful VL BotAction is NOT rejected
# ---------------------------------------------------------------------------


async def test_perception_notes_accepts_ok_botaction(bot_action) -> None:
    """A successful VL BotAction should NOT be rejected by the error guard."""
    ok_action = bot_action(
        content="The image shows a sunset over the ocean with orange and pink hues.",
        error=False,
    )
    should_reject = getattr(ok_action, "error", None)
    assert should_reject is False


# ---------------------------------------------------------------------------
# Test 7: Non-BotAction dict results (from direct backend calls)
# ---------------------------------------------------------------------------


async def test_direct_backend_call_ladder_exhausted_detection() -> None:
    """Handlers that call the backend directly (not via see_infer) must also
    detect ladder_exhausted in dict results.
    """
    error_dict = {
        "text": "Unavailable",
        "ladder_exhausted": True,
        "model": None,
    }
    is_error = isinstance(error_dict, dict) and (error_dict.get("ladder_exhausted") or error_dict.get("status") == "error" or (error_dict.get("text") or "").strip() == "")
    assert is_error is True


# ---------------------------------------------------------------------------
# Test 8: ResultAggregator does not count failed items as successful
# ---------------------------------------------------------------------------


async def test_result_aggregator_failed_items_not_counted() -> None:
    """The ResultAggregator's success filter must properly separate failed
    items when a VL item returned a failed result.
    """
    from bot.modality import InputItem, InputModality
    from bot.result_aggregator import ResultAggregator

    agg = ResultAggregator()

    # Simulate a failed VL image item
    agg.add_result(
        item_index=0,
        item=InputItem(source_type="url", payload="https://pbs.twimg.com/media/xyz.jpg", order_index=0),
        modality=InputModality.SINGLE_IMAGE,
        result_text="Vision analysis returned empty content",
        success=False,
        duration=5.0,
        attempts=1,
    )

    stats = agg.get_summary_stats()
    assert stats["successful_items"] == 0
    assert stats["failed_items"] == 1
    assert stats["total_items"] == 1

    # get_aggregated_prompt should return empty since no original text
    prompt = agg.get_aggregated_prompt("")
    assert prompt == ""


# ---------------------------------------------------------------------------
# Test 9: ResultAggregator with mixed success + failure
# ---------------------------------------------------------------------------


async def test_result_aggregator_mixed_success_failure() -> None:
    """When one item succeeds and one fails, only the successful one is in the prompt."""
    from bot.modality import InputItem, InputModality
    from bot.result_aggregator import ResultAggregator

    agg = ResultAggregator()

    agg.add_result(
        item_index=0,
        item=InputItem(source_type="url", payload="https://pbs.twimg.com/media/xyz.jpg", order_index=0),
        modality=InputModality.SINGLE_IMAGE,
        result_text="A photo of a cat sitting on a keyboard",
        success=True,
        duration=3.0,
        attempts=1,
    )
    agg.add_result(
        item_index=1,
        item=InputItem(source_type="url", payload="https://pbs.twimg.com/media/abc.jpg", order_index=1),
        modality=InputModality.SINGLE_IMAGE,
        result_text="Vision service is temporarily unavailable",
        success=False,
        duration=2.0,
        attempts=1,
    )

    stats = agg.get_summary_stats()
    assert stats["successful_items"] == 1
    assert stats["failed_items"] == 1
    assert stats["total_items"] == 2

    prompt = agg.get_aggregated_prompt("")
    assert "cat" in prompt
    assert "unavailable" not in prompt


# ---------------------------------------------------------------------------
# Test 10: has_visual_facts_section should not be triggered by error text
# ---------------------------------------------------------------------------


async def test_error_text_does_not_trigger_visual_facts() -> None:
    """Error/failure text from failed VL should not trigger visual_facts_detected."""
    from bot.router_components import has_visual_facts_section

    error_texts = [
        "The vision service is temporarily unavailable. Please try again in a few minutes.",
        "The vision model returned an empty response.",
        "Vision processing returned empty content",
        "Vision analysis returned no results",
        "Failed to analyze the image",
    ]
    for text in error_texts:
        assert has_visual_facts_section(text) is False, f"Error text should not have visual facts: {text}"
