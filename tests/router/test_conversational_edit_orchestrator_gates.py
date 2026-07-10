"""Verify the conversational image-edit route runs through the REAL vision
orchestrator's safety filter + budget gates - same gates as /imgedit, no
bypass. [SFT][REH]
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot.router import Router
from bot.vision.orchestrator import VisionOrchestrator
from bot.vision.safety_filter import SafetyLevel, SafetyResult
from bot.vision.types import VisionTask


class DummyBot:
    def __init__(self) -> None:
        self.config = {
            "VISION_ENABLED": True,
            "VISION_CONVERSATIONAL_EDIT_ENABLED": True,
        }
        self.tts_manager = None
        self.loop = None
        self.system_prompts = {"vl_prompt": None}
        self.enhanced_context_manager = None


@pytest.fixture
def orchestrator():
    config = {
        "VISION_MAX_CONCURRENT_JOBS": 10,
        "VISION_MAX_USER_CONCURRENT_JOBS": 2,
        "VISION_JOB_TIMEOUT_SECONDS": 300,
        "VISION_ARTIFACTS_DIR": "/tmp/test_artifacts",
        "VISION_ARTIFACT_TTL_DAYS": 7,
        "VISION_JOBS_DIR": "/tmp/test_jobs",
        "VISION_LEDGER_PATH": "/tmp/test_ledger.json",
    }
    with patch("bot.vision.orchestrator.VisionJobStore") as mock_job_store:
        mock_job_store.return_value = AsyncMock()
        orch = VisionOrchestrator(config)
        orch.gateway = AsyncMock()
        orch.safety_filter = AsyncMock()
        orch.budget_manager = AsyncMock()
        orch.pricing_table = Mock()
        orch.safety_filter.validate_request.return_value = SafetyResult(
            approved=True,
            level=SafetyLevel.SAFE,
            reason="",
            user_message="",
            detected_issues=[],
        )
        orch.job_store.save_job = AsyncMock()
        orch.job_store.load_job = AsyncMock()
        yield orch


@pytest.mark.asyncio
async def test_edit_route_invokes_safety_filter_and_budget_check(orchestrator, monkeypatch) -> None:
    from bot.vision.budget_manager_v2 import BudgetResult
    from bot.vision.money import Money

    orchestrator.budget_manager.check_budget.return_value = BudgetResult(
        approved=True,
        reason="",
        user_message="",
        remaining_budget=Money("5.00"),
    )
    orchestrator.budget_manager.reserve_budget = AsyncMock()

    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = orchestrator

    # Don't wait on the real gateway/provider call - only the gate matters here.
    monkeypatch.setattr(router, "_await_conversational_edit_job", AsyncMock(return_value=None))

    message = Mock()
    message.author.id = 42
    message.guild = None
    message.channel.id = 7

    await router._run_conversational_edit_job(message, "give this guy a beard", type("R", (), {"data": b"fake-png-bytes", "content_type": "image/png", "source": "reply"})())

    assert orchestrator.safety_filter.validate_request.await_count == 1
    called_request = orchestrator.safety_filter.validate_request.await_args.args[0]
    assert called_request.task == VisionTask.IMAGE_TO_IMAGE
    assert called_request.input_image_data == b"fake-png-bytes"
    assert called_request.prompt == "give this guy a beard"

    assert orchestrator.budget_manager.check_budget.await_count == 1
    budget_request = orchestrator.budget_manager.check_budget.await_args.args[0]
    assert budget_request is called_request


@pytest.mark.asyncio
async def test_edit_route_blocked_by_safety_filter_returns_friendly_error(orchestrator, monkeypatch) -> None:
    orchestrator.safety_filter.validate_request.return_value = SafetyResult(
        approved=False,
        level=SafetyLevel.BLOCKED,
        reason="blocked_keyword:x",
        user_message="That request violates our content policy.",
        detected_issues=["blocked_keyword:x"],
    )

    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = orchestrator
    router._metric_inc = Mock()

    message = Mock()
    message.author.id = 42
    message.guild = None
    message.channel.id = 7

    resolved = type("R", (), {"data": b"fake-png-bytes", "content_type": "image/png", "source": "reply"})()
    action = await router._run_conversational_edit_job(message, "nsfw stuff", resolved)

    assert action.error is True
    assert action.content == "That request violates our content policy."
    router._metric_inc.assert_any_call("vision.route.conversational_edit", {"outcome": "safety_blocked"})
