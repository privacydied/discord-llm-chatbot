"""Test Vision System Integration.

Basic integration tests to verify all vision components can be imported
and initialized correctly with the bot configuration.
"""

import asyncio
import contextlib
import sys
from pathlib import Path

from bot.config import load_config
from bot.vision import (
    VisionArtifactCache,
    VisionBudgetManager,
    VisionGateway,
    VisionIntentRouter,
    VisionJobStore,
    VisionOrchestrator,
    VisionProvider,
    VisionRequest,
    VisionSafetyFilter,
    VisionTask,
)


class TestVisionIntegration:
    """Integration tests for vision system components."""

    def setup_method(self) -> None:
        """Setup test configuration."""
        self.config = load_config()
        # Override for testing
        self.config["VISION_ENABLED"] = True
        self.config["VISION_API_KEY"] = "test_key"
        self.config["VISION_DATA_DIR"] = Path("/tmp/vision_test_data")
        self.config["VISION_ARTIFACTS_DIR"] = Path("/tmp/vision_test_data/artifacts")
        self.config["VISION_JOBS_DIR"] = Path("/tmp/vision_test_data/jobs")
        self.config["VISION_POLICY_PATH"] = "configs/vision_policy.json"

    def test_import_all_components(self) -> None:
        """Test that all vision components can be imported."""
        # This test passes if imports work without exception
        assert VisionGateway is not None
        assert VisionOrchestrator is not None
        assert VisionJobStore is not None
        assert VisionIntentRouter is not None
        assert VisionSafetyFilter is not None
        assert VisionBudgetManager is not None
        assert VisionArtifactCache is not None

    def test_vision_gateway_init(self) -> None:
        """Test VisionGateway initialization."""
        try:
            gateway = VisionGateway(self.config)
            assert gateway is not None
        except Exception as e:
            msg = f"VisionGateway initialization failed: {e}"
            raise AssertionError(msg)

    def test_job_store_init(self) -> None:
        """Test VisionJobStore initialization."""
        try:
            job_store = VisionJobStore(self.config)
            assert job_store is not None
        except Exception as e:
            msg = f"VisionJobStore initialization failed: {e}"
            raise AssertionError(msg)

    def test_intent_router_init(self) -> None:
        """Test VisionIntentRouter initialization."""
        try:
            intent_router = VisionIntentRouter(self.config)
            assert intent_router is not None
        except Exception as e:
            msg = f"VisionIntentRouter initialization failed: {e}"
            raise AssertionError(msg)

    def test_safety_filter_init(self) -> None:
        """Test VisionSafetyFilter initialization."""
        try:
            safety_filter = VisionSafetyFilter(self.config)
            assert safety_filter is not None
        except Exception as e:
            msg = f"VisionSafetyFilter initialization failed: {e}"
            raise AssertionError(msg)

    def test_budget_manager_init(self) -> None:
        """Test VisionBudgetManager initialization."""
        try:
            budget_manager = VisionBudgetManager(self.config)
            assert budget_manager is not None
        except Exception as e:
            msg = f"VisionBudgetManager initialization failed: {e}"
            raise AssertionError(msg)

    def test_artifact_cache_init(self) -> None:
        """Test VisionArtifactCache initialization."""
        try:
            artifact_cache = VisionArtifactCache(self.config)
            assert artifact_cache is not None
        except Exception as e:
            msg = f"VisionArtifactCache initialization failed: {e}"
            raise AssertionError(msg)

    def test_vision_request_creation(self) -> None:
        """Test VisionRequest creation."""
        try:
            request = VisionRequest(
                task=VisionTask.TEXT_TO_IMAGE,
                prompt="A beautiful sunset",
                user_id="test_user_123",
                preferred_provider=VisionProvider.TOGETHER,
            )
            assert request is not None
            assert request.task == VisionTask.TEXT_TO_IMAGE
            assert request.prompt == "A beautiful sunset"
            assert request.user_id == "test_user_123"
        except Exception as e:
            msg = f"VisionRequest creation failed: {e}"
            raise AssertionError(msg)

    async def test_orchestrator_init(self) -> None:
        """Test VisionOrchestrator initialization (async)."""
        orchestrator = None
        try:
            orchestrator = VisionOrchestrator(self.config)
            assert orchestrator is not None
        except Exception as e:
            msg = f"VisionOrchestrator initialization failed: {e}"
            raise AssertionError(msg)
        finally:
            if orchestrator:
                with contextlib.suppress(Exception):
                    await orchestrator.close()


def main() -> int | None:
    """Run basic integration test."""
    try:
        # Test imports
        test = TestVisionIntegration()
        test.setup_method()

        test.test_import_all_components()

        test.test_vision_gateway_init()
        test.test_job_store_init()
        test.test_intent_router_init()
        test.test_safety_filter_init()
        test.test_budget_manager_init()
        test.test_artifact_cache_init()
        test.test_vision_request_creation()

        asyncio.run(test.test_orchestrator_init())

        return 0

    except Exception as e:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
