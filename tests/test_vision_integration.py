"""
Test Vision System Integration

Basic integration tests to verify all vision components can be imported
and initialized correctly with the bot configuration.
"""

import asyncio
import sys
from pathlib import Path

from bot.config import load_config
from bot.vision import (
    VisionGateway,
    VisionOrchestrator,
    VisionJobStore,
    VisionIntentRouter,
    VisionSafetyFilter,
    VisionBudgetManager,
    VisionArtifactCache,
    VisionRequest,
    VisionTask,
    VisionProvider,
)


class TestVisionIntegration:
    """Integration tests for vision system components"""

    def setup_method(self):
        """Setup test configuration"""
        self.config = load_config()
        # Override for testing
        self.config["VISION_ENABLED"] = True
        self.config["VISION_API_KEY"] = "test_key"
        self.config["VISION_DATA_DIR"] = Path("/tmp/vision_test_data")
        self.config["VISION_ARTIFACTS_DIR"] = Path("/tmp/vision_test_data/artifacts")
        self.config["VISION_JOBS_DIR"] = Path("/tmp/vision_test_data/jobs")
        self.config["VISION_POLICY_PATH"] = "configs/vision_policy.json"

    def test_import_all_components(self):
        """Test that all vision components can be imported"""
        # This test passes if imports work without exception
        assert VisionGateway is not None
        assert VisionOrchestrator is not None
        assert VisionJobStore is not None
        assert VisionIntentRouter is not None
        assert VisionSafetyFilter is not None
        assert VisionBudgetManager is not None
        assert VisionArtifactCache is not None

    def test_vision_gateway_init(self):
        """Test VisionGateway initialization"""
        try:
            gateway = VisionGateway(self.config)
            assert gateway is not None
            print("✅ VisionGateway initialized successfully")
        except Exception as e:
            print(f"❌ VisionGateway initialization failed: {e}")
            raise AssertionError(f"VisionGateway initialization failed: {e}")

    def test_job_store_init(self):
        """Test VisionJobStore initialization"""
        try:
            job_store = VisionJobStore(self.config)
            assert job_store is not None
            print("✅ VisionJobStore initialized successfully")
        except Exception as e:
            print(f"❌ VisionJobStore initialization failed: {e}")
            raise AssertionError(f"VisionJobStore initialization failed: {e}")

    def test_intent_router_init(self):
        """Test VisionIntentRouter initialization"""
        try:
            intent_router = VisionIntentRouter(self.config)
            assert intent_router is not None
            print("✅ VisionIntentRouter initialized successfully")
        except Exception as e:
            print(f"❌ VisionIntentRouter initialization failed: {e}")
            raise AssertionError(f"VisionIntentRouter initialization failed: {e}")

    def test_safety_filter_init(self):
        """Test VisionSafetyFilter initialization"""
        try:
            safety_filter = VisionSafetyFilter(self.config)
            assert safety_filter is not None
            print("✅ VisionSafetyFilter initialized successfully")
        except Exception as e:
            print(f"❌ VisionSafetyFilter initialization failed: {e}")
            raise AssertionError(f"VisionSafetyFilter initialization failed: {e}")

    def test_budget_manager_init(self):
        """Test VisionBudgetManager initialization"""
        try:
            budget_manager = VisionBudgetManager(self.config)
            assert budget_manager is not None
            print("✅ VisionBudgetManager initialized successfully")
        except Exception as e:
            print(f"❌ VisionBudgetManager initialization failed: {e}")
            raise AssertionError(f"VisionBudgetManager initialization failed: {e}")

    def test_artifact_cache_init(self):
        """Test VisionArtifactCache initialization"""
        try:
            artifact_cache = VisionArtifactCache(self.config)
            assert artifact_cache is not None
            print("✅ VisionArtifactCache initialized successfully")
        except Exception as e:
            print(f"❌ VisionArtifactCache initialization failed: {e}")
            raise AssertionError(f"VisionArtifactCache initialization failed: {e}")

    def test_vision_request_creation(self):
        """Test VisionRequest creation"""
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
            print("✅ VisionRequest created successfully")
        except Exception as e:
            print(f"❌ VisionRequest creation failed: {e}")
            raise AssertionError(f"VisionRequest creation failed: {e}")

    async def test_orchestrator_init(self):
        """Test VisionOrchestrator initialization (async)"""
        orchestrator = None
        try:
            orchestrator = VisionOrchestrator(self.config)
            assert orchestrator is not None
            print("✅ VisionOrchestrator initialized successfully")
        except Exception as e:
            print(f"❌ VisionOrchestrator initialization failed: {e}")
            raise AssertionError(f"VisionOrchestrator initialization failed: {e}")
        finally:
            if orchestrator:
                try:
                    await orchestrator.close()
                except Exception as e:
                    print(f"Warning: Orchestrator cleanup failed: {e}")


def main():
    """Run basic integration test"""
    print("🧪 Running Vision System Integration Tests\n")

    try:
        # Test imports
        test = TestVisionIntegration()
        test.setup_method()

        print("📦 Testing component imports...")
        test.test_import_all_components()

        print("\n🏗️ Testing component initialization...")
        test.test_vision_gateway_init()
        test.test_job_store_init()
        test.test_intent_router_init()
        test.test_safety_filter_init()
        test.test_budget_manager_init()
        test.test_artifact_cache_init()
        test.test_vision_request_creation()

        print("\n🔄 Testing async orchestrator...")
        asyncio.run(test.test_orchestrator_init())

        print("\n✅ All integration tests passed!")
        print("🎉 Vision system is ready for deployment")
        return 0

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
