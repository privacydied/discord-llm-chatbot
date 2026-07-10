"""Regression test: SafetyResult escalation used to compare `SafetyLevel.value`
strings ("blocked" < "safe" < "warning" alphabetically), so a BLOCKED prompt
never actually escalated `validate_request`'s overall result past SAFE -
content flagged with a blocked keyword was silently APPROVED. [SFT][REH]
"""

import pytest

from bot.vision.safety_filter import SafetyLevel, VisionSafetyFilter, _more_severe
from bot.vision.types import VisionRequest, VisionTask


def test_severity_ordering_is_not_alphabetical() -> None:
    # The literal bug: string comparison disagrees with intended severity order.
    assert SafetyLevel.BLOCKED.value < SafetyLevel.SAFE.value  # "blocked" < "safe"
    # The fix must therefore NOT rely on comparing `.value` strings.
    assert _more_severe(SafetyLevel.BLOCKED, SafetyLevel.SAFE) is True
    assert _more_severe(SafetyLevel.BLOCKED, SafetyLevel.WARNING) is True
    assert _more_severe(SafetyLevel.WARNING, SafetyLevel.SAFE) is True
    assert _more_severe(SafetyLevel.SAFE, SafetyLevel.BLOCKED) is False
    assert _more_severe(SafetyLevel.SAFE, SafetyLevel.SAFE) is False


@pytest.mark.asyncio
async def test_blocked_prompt_keyword_is_rejected_end_to_end() -> None:
    """Exercise the real validate_request aggregation path (not a synthetic
    SafetyLevel) with a deterministic blocked-keyword policy, independent of
    whatever configs/vision_policy.json currently contains. [REH]
    """
    safety_filter = VisionSafetyFilter({})
    safety_filter.blocked_keywords = {"forbiddenword"}
    safety_filter.warning_keywords = set()
    safety_filter.blocked_patterns = []
    safety_filter.warning_patterns = []

    request = VisionRequest(
        task=VisionTask.TEXT_TO_IMAGE,
        prompt="a photo of a forbiddenword",
        user_id="111",
    )

    result = await safety_filter.validate_request(request)

    assert result.approved is False
    assert result.level == SafetyLevel.BLOCKED
    assert any("forbiddenword" in issue for issue in result.detected_issues)
