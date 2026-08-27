"""Unit tests for Router's conversational image-edit job lifecycle helpers:
submit -> poll (bounded) -> reply-with-attachment or friendly error. Mirrors
/imgedit's error surface (safety/budget/provider) without a live provider.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bot.router import Router
from bot.router_components import ParsedEditFlags, ResolvedEditImage
from bot.vision.types import VisionError, VisionErrorType, VisionJobState, VisionProvider


class DummyBot:
    def __init__(self) -> None:
        self.config = {
            "VISION_ENABLED": True,
            "VISION_CONVERSATIONAL_EDIT_ENABLED": True,
            "VISION_CONVERSATIONAL_EDIT_TIMEOUT_S": 0.02,
            "VISION_CONVERSATIONAL_EDIT_POLL_INTERVAL_S": 0.01,
        }
        self.tts_manager = None
        self.loop = None
        self.system_prompts = {"vl_prompt": None}
        self.enhanced_context_manager = None


@pytest.fixture
def router():
    r = Router(DummyBot())
    r._vision_orchestrator = Mock()
    r._vision_orchestrator.submit_job = AsyncMock()
    r._vision_orchestrator.get_job_status = AsyncMock()
    r._vision_orchestrator.cancel_job = AsyncMock()
    r._metric_inc = Mock()
    return r


def _message():
    m = Mock()
    m.author.id = 42
    m.guild = None
    m.channel.id = 7
    return m


def _resolved():
    return ResolvedEditImage(data=b"bytes", content_type="image/png", source="current")


def _job(state: VisionJobState, *, response=None, error=None, job_id="abcdef1234567890"):
    return SimpleNamespace(
        job_id=job_id,
        state=SimpleNamespace(value=state.value),
        response=response,
        error=error,
        is_terminal_state=lambda: state in (VisionJobState.COMPLETED, VisionJobState.FAILED, VisionJobState.CANCELLED, VisionJobState.EXPIRED),
    )


@pytest.mark.asyncio
class TestRunConversationalEditJob:
    async def test_submit_raises_content_filtered_maps_to_safety_blocked(self, router) -> None:
        router._vision_orchestrator.submit_job.side_effect = VisionError(
            error_type=VisionErrorType.CONTENT_FILTERED,
            message="blocked",
            user_message="Your request violates content policy.",
        )
        action = await router._run_conversational_edit_job(_message(), "nsfw", _resolved())
        assert action.error is True
        assert action.content == "Your request violates content policy."
        router._metric_inc.assert_any_call("vision.route.conversational_edit", {"outcome": "safety_blocked"})

    async def test_submit_raises_quota_exceeded_maps_to_budget_blocked(self, router) -> None:
        router._vision_orchestrator.submit_job.side_effect = VisionError(
            error_type=VisionErrorType.QUOTA_EXCEEDED,
            message="over budget",
            user_message="You've hit your budget limit.",
        )
        action = await router._run_conversational_edit_job(_message(), "edit it", _resolved())
        assert action.error is True
        assert action.content == "You've hit your budget limit."
        router._metric_inc.assert_any_call("vision.route.conversational_edit", {"outcome": "budget_blocked"})

    async def test_submit_raises_generic_exception_maps_to_provider_error(self, router) -> None:
        router._vision_orchestrator.submit_job.side_effect = RuntimeError("boom")
        action = await router._run_conversational_edit_job(_message(), "edit it", _resolved())
        assert action.error is True
        router._metric_inc.assert_any_call("vision.route.conversational_edit", {"outcome": "provider_error"})

    async def test_successful_job_returns_action_with_files(self, router, tmp_path) -> None:
        artifact = tmp_path / "out.png"
        artifact.write_bytes(b"fake-png")
        router._vision_orchestrator.submit_job.return_value = SimpleNamespace(job_id="job123")
        router._vision_orchestrator.get_job_status.return_value = _job(
            VisionJobState.COMPLETED,
            response=SimpleNamespace(artifacts=[artifact]),
        )

        action = await router._run_conversational_edit_job(_message(), "edit it", _resolved())

        assert action.error is False
        assert action.content == ""
        assert len(action.files) == 1
        router._metric_inc.assert_any_call("vision.route.conversational_edit", {"outcome": "success"})

    async def test_failed_job_maps_error_type_to_outcome(self, router) -> None:
        router._vision_orchestrator.submit_job.return_value = SimpleNamespace(job_id="job123")
        router._vision_orchestrator.get_job_status.return_value = _job(
            VisionJobState.FAILED,
            error=VisionError(
                error_type=VisionErrorType.CONTENT_FILTERED,
                message="blocked mid-flight",
                user_message="That image could not be processed due to content policy.",
            ),
        )

        action = await router._run_conversational_edit_job(_message(), "edit it", _resolved())

        assert action.error is True
        assert action.content == "That image could not be processed due to content policy."
        router._metric_inc.assert_any_call("vision.route.conversational_edit", {"outcome": "safety_blocked"})

    async def test_completed_job_with_missing_artifact_file_is_provider_error(self, router) -> None:
        router._vision_orchestrator.submit_job.return_value = SimpleNamespace(job_id="job123")
        router._vision_orchestrator.get_job_status.return_value = _job(
            VisionJobState.COMPLETED,
            response=SimpleNamespace(artifacts=[Path("/nonexistent/path.png")]),
        )

        action = await router._run_conversational_edit_job(_message(), "edit it", _resolved())

        assert action.error is True
        assert "no image" in action.content.lower()
        router._metric_inc.assert_any_call("vision.route.conversational_edit", {"outcome": "provider_error"})

    async def test_timeout_polls_then_cancels_and_reports_timeout(self, router) -> None:
        router._vision_orchestrator.submit_job.return_value = SimpleNamespace(job_id="job123")
        router._vision_orchestrator.get_job_status.side_effect = [
            _job(VisionJobState.RUNNING),
            _job(VisionJobState.RUNNING),
            None,  # final status lookup after cancel returns nothing
        ]

        action = await router._run_conversational_edit_job(_message(), "edit it", _resolved())

        assert action.error is True
        assert "timed out" in action.content.lower()
        router._vision_orchestrator.cancel_job.assert_awaited_once()
        router._metric_inc.assert_any_call("vision.route.conversational_edit", {"outcome": "provider_error"})


@pytest.mark.asyncio
class TestConversationalEditFlags:
    """Flags parsed from the explicit trigger must map onto VisionRequest
    fields, and a placeholder message must not affect the returned BotAction.
    """

    async def test_flags_map_onto_vision_request_fields(self, router) -> None:
        artifact_path = Path("/nonexistent/path.png")  # job fails fast; we only care about the submitted request
        router._vision_orchestrator.submit_job.return_value = SimpleNamespace(job_id="job123")
        router._vision_orchestrator.get_job_status.return_value = _job(
            VisionJobState.COMPLETED,
            response=SimpleNamespace(artifacts=[artifact_path]),
        )
        flags = ParsedEditFlags(
            prompt="make him a superhero",
            seed=7,
            steps=20,
            strength=0.6,
            guidance=9.5,
            negative="blurry",
            provider="together",
            model="some/model",
        )

        await router._run_conversational_edit_job(_message(), flags.prompt, _resolved(), flags)

        submitted = router._vision_orchestrator.submit_job.await_args.args[0]
        assert submitted.seed == 7
        assert submitted.steps == 20
        assert submitted.strength == 0.6
        assert submitted.guidance_scale == 9.5
        assert submitted.negative_prompt == "blurry"
        assert submitted.preferred_model == "some/model"
        assert submitted.preferred_provider == VisionProvider.TOGETHER

    async def test_none_flags_apply_no_overrides(self, router) -> None:
        """Heuristic-trigger path (flags=None) must behave exactly as before flags existed."""
        router._vision_orchestrator.submit_job.return_value = SimpleNamespace(job_id="job123")
        router._vision_orchestrator.get_job_status.return_value = _job(
            VisionJobState.COMPLETED,
            response=SimpleNamespace(artifacts=[Path("/nonexistent/path.png")]),
        )

        await router._run_conversational_edit_job(_message(), "edit it", _resolved())

        submitted = router._vision_orchestrator.submit_job.await_args.args[0]
        assert submitted.seed is None
        assert submitted.preferred_model is None
        assert submitted.preferred_provider is None

    async def test_placeholder_send_failure_does_not_affect_result(self, router, monkeypatch) -> None:
        """A placeholder is best-effort UX -- if it can't be sent, the actual
        edit result must be unaffected. [REH]
        """
        import bot.router as router_mod

        monkeypatch.setattr(router_mod, "safe_send", AsyncMock(side_effect=RuntimeError("no perms")))
        artifact = Path("/nonexistent/path.png")
        router._vision_orchestrator.submit_job.return_value = SimpleNamespace(job_id="job123")
        router._vision_orchestrator.get_job_status.return_value = _job(
            VisionJobState.COMPLETED,
            response=SimpleNamespace(artifacts=[artifact]),
        )

        action = await router._run_conversational_edit_job(_message(), "edit it", _resolved())

        assert action.error is True  # missing artifact -> provider_error, unrelated to placeholder
        assert "no image" in action.content.lower()

    async def test_placeholder_sent_and_cleared_around_the_job(self, router, monkeypatch) -> None:
        import bot.router as router_mod

        placeholder = AsyncMock()
        monkeypatch.setattr(router_mod, "safe_send", AsyncMock(return_value=placeholder))
        artifact = Path("/nonexistent/path.png")
        router._vision_orchestrator.submit_job.return_value = SimpleNamespace(job_id="job123")
        router._vision_orchestrator.get_job_status.return_value = _job(
            VisionJobState.COMPLETED,
            response=SimpleNamespace(artifacts=[artifact]),
        )

        action = await router._run_conversational_edit_job(_message(), "edit it", _resolved())

        router_mod.safe_send.assert_awaited_once()
        placeholder.delete.assert_awaited_once()
        # The returned action is the real result, unaffected by the placeholder lifecycle.
        assert action.error is True
        assert "no image" in action.content.lower()
