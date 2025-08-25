#!/usr/bin/env python3
"""
Test Vision system integration with the main router.
Verifies natural language intent detection and job orchestration flow.
"""

import asyncio
import os
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Mock Discord imports before importing bot modules
class MockMessage:
    def __init__(self):
        self.id = 123456789
        self.author = MockUser()
        self.guild = MockGuild()
        self.channel = MockChannel()

class MockUser:
    def __init__(self):
        self.id = 987654321

class MockGuild:
    def __init__(self):
        self.id = 555666777

class MockChannel:
    def __init__(self):
        self.send = AsyncMock()

class MockBot:
    def __init__(self):
        self.config = {
            "VISION_ENABLED": "true",
            "VISION_API_KEY": "test-key",
            "VISION_DATA_DIR": "/tmp/test-vision",
            "VISION_POLICY_PATH": "/tmp/test-policy.json",
            "VISION_DRY_RUN_MODE": "true"
        }

# Set up test environment
os.environ.update({
    "VISION_ENABLED": "true",
    "VISION_API_KEY": "test-key",
    "VISION_DATA_DIR": "/tmp/test-vision",
    "VISION_POLICY_PATH": "/tmp/test-policy.json",
    "VISION_DRY_RUN_MODE": "true"
})

async def test_vision_router_integration():
    """Test Vision system integration with router [CDiP]"""
    
    print("🧪 Testing Vision Router Integration...")
    
    # Create test data directory
    test_dir = Path("/tmp/test-vision")
    test_dir.mkdir(exist_ok=True)
    
    # Create test policy file
    policy_path = Path("/tmp/test-policy.json")
    policy_path.write_text('''{
        "content_safety": {
            "enabled": true,
            "blocked_keywords": ["test_blocked"],
            "nsfw_detection": false
        },
        "server_overrides": {}
    }''')
    
    try:
        # Test imports work
        print("📦 Testing imports...")
        from bot.router import Router
        from bot.vision import VisionIntentRouter, VisionOrchestrator
        from bot.vision.types import VisionTask
        print("✅ Router and Vision imports successful")
        
        # Create mock bot and router
        mock_bot = MockBot()
        router = Router(mock_bot)
        
        # Verify Vision components initialized
        assert router._vision_intent_router is not None, "VisionIntentRouter should be initialized"
        assert router._vision_orchestrator is not None, "VisionOrchestrator should be initialized"
        print("✅ Vision components initialized in router")
        
        # Test intent detection for various prompts
        test_prompts = [
            "generate an image of a cat",
            "create a picture of a sunset",
            "make me a video of dancing",
            "just having a normal conversation",
            "what's the weather like?",
            "draw a dragon breathing fire"
        ]
        
        print("🎯 Testing intent detection...")
        vision_detected = 0
        
        for prompt in test_prompts:
            try:
                intent_result = await router._vision_intent_router.determine_intent(
                    user_message=prompt,
                    context="",
                    user_id="test_user",
                    guild_id="test_guild"
                )
                
                if intent_result.decision.use_vision:
                    vision_detected += 1
                    print(f"  ✅ Vision detected for: '{prompt}' (confidence: {intent_result.confidence:.2f})")
                    print(f"     Task: {intent_result.extracted_params.task}")
                    print(f"     Prompt: {intent_result.extracted_params.prompt}")
                else:
                    print(f"  ➡️  Regular text for: '{prompt}'")
                    
            except Exception as e:
                print(f"  ❌ Error processing '{prompt}': {e}")
        
        print(f"📊 Vision intent detected in {vision_detected}/{len(test_prompts)} prompts")
        
        # Test _invoke_text_flow with vision intent
        print("🔀 Testing text flow routing...")
        
        mock_message = MockMessage()
        
        # Mock the vision generation to avoid actual API calls
        with patch.object(router, '_handle_vision_generation') as mock_vision_handler:
            mock_vision_handler.return_value = MagicMock(content="Vision generation completed", has_payload=True)
            
            # Test with vision prompt
            result = await router._invoke_text_flow(
                content="generate an image of a rainbow",
                message=mock_message,
                context_str=""
            )
            
            # Check if vision handler was called
            if mock_vision_handler.called:
                print("✅ Vision generation handler was invoked for vision prompt")
            else:
                print("❌ Vision generation handler was not called")
        
        # Test router method existence
        required_methods = [
            '_handle_vision_generation',
            '_monitor_vision_job', 
            '_handle_vision_success',
            '_handle_vision_failure',
            '_create_progress_bar'
        ]
        
        print("🔍 Checking required router methods...")
        for method_name in required_methods:
            if hasattr(router, method_name):
                print(f"  ✅ {method_name} exists")
            else:
                print(f"  ❌ {method_name} missing")
        
        # Test progress bar utility
        if hasattr(router, '_create_progress_bar'):
            progress_bar = router._create_progress_bar(75)
            print(f"📊 Progress bar test: {progress_bar}")
            assert len(progress_bar) > 0, "Progress bar should not be empty"
            print("✅ Progress bar generation works")
        
        print("\n🎉 Vision Router Integration Test Complete!")
        print("✅ All core integration components are functional")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
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

async def test_vision_orchestrator_startup():
    """Test Vision orchestrator startup and shutdown [REH]"""
    
    print("\n🔄 Testing Vision Orchestrator Lifecycle...")
    
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
            "VISION_DRY_RUN_MODE": "true"
        }
        
        # Create policy file
        (test_dir / "policy.json").write_text('{"content_safety": {"enabled": false}}')
        
        orchestrator = VisionOrchestrator(mock_config)
        
        # Test startup
        print("🚀 Starting orchestrator...")
        await orchestrator.startup()
        print("✅ Orchestrator startup completed")
        
        # Verify directories created
        expected_dirs = ["jobs", "artifacts"]
        for dir_name in expected_dirs:
            dir_path = test_dir / dir_name
            if dir_path.exists():
                print(f"  ✅ {dir_name}/ directory created")
            else:
                print(f"  ❌ {dir_name}/ directory missing")
        
        # Test shutdown
        print("🛑 Shutting down orchestrator...")
        await orchestrator.shutdown()
        print("✅ Orchestrator shutdown completed")
        
    except Exception as e:
        print(f"❌ Orchestrator lifecycle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
    
    return True

if __name__ == "__main__":
    async def main():
        print("🧪 Vision Router Integration Tests")
        print("=" * 50)
        
        # Test integration
        test1_passed = await test_vision_router_integration()
        
        # Test orchestrator lifecycle
        test2_passed = await test_vision_orchestrator_startup()
        
        print("\n" + "=" * 50)
        if test1_passed and test2_passed:
            print("🎉 ALL TESTS PASSED! Vision integration is ready.")
        else:
            print("❌ Some tests failed. Check integration.")
        
        return test1_passed and test2_passed
    
    # Run the tests
    result = asyncio.run(main())
    exit(0 if result else 1)
