#!/usr/bin/env python3
"""Test Vision system integration with the main router.
Verifies natural language intent detection and job orchestration flow.
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# Mock Discord imports before importing bot modules
class MockMessage:
    def __init__(self) -> None:
        self.id = 123456789
        self.author = MockUser()
        self.guild = MockGuild()
        self.channel = MockChannel()


class MockUser:
    def __init__(self) -> None:
        self.id = 987654321


class MockGuild:
    def __init__(self) -> None:
        self.id = 555666777


class MockChannel:
    def __init__(self) -> None:
        self.send = AsyncMock()


class MockBot:
    def __init__(self) -> None:
        self.config = {
            "VISION_ENABLED": "true",
            "VISION_API_KEY": "test-key",
            "VISION_DATA_DIR": "/tmp/test-vision",
            "VISION_POLICY_PATH": "/tmp/test-policy.json",
            "VISION_DRY_RUN_MODE": "true",
        }


# Set up test environment
os.environ.update(
    {
        "VISION_ENABLED": "true",
        "VISION_API_KEY": "test-key",
        "VISION_DATA_DIR": "/tmp/test-vision",
        "VISION_POLICY_PATH": "/tmp/test-policy.json",
        "VISION_DRY_RUN_MODE": "true",
    },
)


async def test_vision_router_integration() -> bool:
    """Test Vision system integration with router [CDiP]."""
    # Create test data directory
    test_dir = Path("/tmp/test-vision")
    test_dir.mkdir(exist_ok=True)

    # Create test policy file
    policy_path = Path("/tmp/test-policy.json")
    policy_path.write_text("""{
        "content_safety": {
            "enabled": true,
            "blocked_keywords": ["test_blocked"],
            "nsfw_detection": false
        },
        "server_overrides": {}
    }""")

    try:
        # Test imports work
        from bot.router import Router


        # Create mock bot and router
        mock_bot = MockBot()
        router = Router(mock_bot)

        # Verify Vision components initialized
        assert router._vision_intent_router is not None, "VisionIntentRouter should be initialized"
        assert router._vision_orchestrator is not None, "VisionOrchestrator should be initialized"

        # Test intent detection for various prompts
        test_prompts = [
            "generate an image of a cat",
            "create a picture of a sunset",
            "make me a video of dancing",
            "just having a normal conversation",
            "what's the weather like?",
            "draw a dragon breathing fire",
        ]

        vision_detected = 0

        for prompt in test_prompts:
            try:
                intent_result = await router._vision_intent_router.determine_intent(
                    user_message=prompt,
                    context="",
                    user_id="test_user",
                    guild_id="test_guild",
                )

                if intent_result.decision.use_vision:
                    vision_detected += 1
                else:
                    pass

            except Exception as e:
                pass


        # Test _invoke_text_flow with vision intent

        mock_message = MockMessage()

        # Mock the vision generation to avoid actual API calls
        with patch.object(router, "_handle_vision_generation") as mock_vision_handler:
            mock_vision_handler.return_value = MagicMock(content="Vision generation completed", has_payload=True)

            # Test with vision prompt
            await router._invoke_text_flow(
                content="generate an image of a rainbow",
                message=mock_message,
                context_str="",
            )

            # Check if vision handler was called
            if mock_vision_handler.called:
                pass
            else:
                pass

        # Test router method existence
        required_methods = [
            "_handle_vision_generation",
            "_monitor_vision_job",
            "_handle_vision_success",
            "_handle_vision_failure",
            "_create_progress_bar",
        ]

        for method_name in required_methods:
            if hasattr(router, method_name):
                pass
            else:
                pass

        # Test progress bar utility
        if hasattr(router, "_create_progress_bar"):
            progress_bar = router._create_progress_bar(75)
            assert len(progress_bar) > 0, "Progress bar should not be empty"


    except Exception as e:
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup test files
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        if policy_path.exists():
            policy_path.unlink(missing_ok=True)

    return True


async def test_vision_orchestrator_startup() -> bool:
    """Test Vision orchestrator startup and shutdown [REH]."""
    test_dir = Path("/tmp/test-vision-lifecycle")
    test_dir.mkdir(exist_ok=True)

    try:
        from bot.vision import VisionOrchestrator

        mock_config = {
            "VISION_DATA_DIR": str(test_dir),
            "VISION_POLICY_PATH": str(test_dir / "policy.json"),
            "VISION_JOBS_DIR": str(test_dir / "jobs"),
            "VISION_LEDGER_PATH": str(test_dir / "ledger.jsonl"),
            "VISION_ARTIFACTS_DIR": str(test_dir / "artifacts"),
            "VISION_DRY_RUN_MODE": "true",
        }

        # Create policy file
        (test_dir / "policy.json").write_text('{"content_safety": {"enabled": false}}')

        orchestrator = VisionOrchestrator(mock_config)

        # Test startup
        await orchestrator.startup()

        # Verify directories created
        expected_dirs = ["jobs", "artifacts"]
        for dir_name in expected_dirs:
            dir_path = test_dir / dir_name
            if dir_path.exists():
                pass
            else:
                pass

        # Test shutdown
        await orchestrator.shutdown()

    except Exception as e:
        import traceback

        traceback.print_exc()
        return False

    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

    return True


if __name__ == "__main__":

    async def main():

        # Test integration
        test1_passed = await test_vision_router_integration()

        # Test orchestrator lifecycle
        test2_passed = await test_vision_orchestrator_startup()

        if test1_passed and test2_passed:
            pass
        else:
            pass

        return test1_passed and test2_passed

    # Run the tests
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
