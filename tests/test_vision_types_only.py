#!/usr/bin/env python3
"""
Vision Types Test - Test core vision types without Discord dependencies

Tests only the vision types and enums to verify basic functionality
without requiring full bot initialization or Discord libraries.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_vision_types_import():
    """Test importing vision types"""
    try:
        from bot.vision.types import (
            VisionTask, VisionProvider, VisionJobState,
            VisionError, VisionErrorType, 
            VisionRequest, VisionResponse, VisionJob,
            IntentDecision, IntentResult
        )
        print("✅ Vision types imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import vision types: {e}")
        return False

def test_vision_enums():
    """Test vision enums"""
    try:
        from bot.vision.types import VisionTask, VisionProvider, VisionJobState
        
        # Test VisionTask enum
        assert VisionTask.TEXT_TO_IMAGE.value == "text_to_image"
        assert VisionTask.IMAGE_TO_IMAGE.value == "image_to_image"  
        assert VisionTask.TEXT_TO_VIDEO.value == "text_to_video"
        assert VisionTask.IMAGE_TO_VIDEO.value == "image_to_video"
        
        # Test VisionProvider enum
        assert VisionProvider.TOGETHER.value == "together"
        assert VisionProvider.NOVITA.value == "novita"
        
        # Test VisionJobState enum
        assert VisionJobState.CREATED.value == "created"
        assert VisionJobState.QUEUED.value == "queued"
        assert VisionJobState.RUNNING.value == "running"
        assert VisionJobState.COMPLETED.value == "completed"
        assert VisionJobState.FAILED.value == "failed"
        
        print("✅ Vision enums work correctly")
        return True
    except Exception as e:
        print(f"❌ Vision enums test failed: {e}")
        return False

def test_vision_request():
    """Test VisionRequest creation"""
    try:
        from bot.vision.types import VisionRequest, VisionTask, VisionProvider
        
        request = VisionRequest(
            task=VisionTask.TEXT_TO_IMAGE,
            prompt="A beautiful sunset",
            user_id="test_user_123",
            preferred_provider=VisionProvider.TOGETHER
        )
        
        assert request.task == VisionTask.TEXT_TO_IMAGE
        assert request.prompt == "A beautiful sunset"
        assert request.user_id == "test_user_123"
        assert request.preferred_provider == VisionProvider.TOGETHER
        assert request.width == 1024  # Default value
        assert request.height == 1024  # Default value
        
        print("✅ VisionRequest creation works correctly")
        return True
    except Exception as e:
        print(f"❌ VisionRequest test failed: {e}")
        return False

def test_vision_response():
    """Test VisionResponse creation"""
    try:
        from bot.vision.types import VisionResponse, VisionProvider
        
        response = VisionResponse(
            provider=VisionProvider.TOGETHER,
            success=True,
            result_urls=["https://example.com/image.png"],
            processing_time_seconds=5.2
        )
        
        assert response.provider == VisionProvider.TOGETHER
        assert response.success
        assert len(response.result_urls) == 1
        assert response.result_urls[0] == "https://example.com/image.png"
        assert response.processing_time_seconds == 5.2
        
        print("✅ VisionResponse creation works correctly")
        return True
    except Exception as e:
        print(f"❌ VisionResponse test failed: {e}")
        return False

def test_vision_job():
    """Test VisionJob creation and state transitions"""
    try:
        from bot.vision.types import (
            VisionJob, VisionRequest, VisionTask, 
            VisionJobState
        )
        
        # Create request
        request = VisionRequest(
            task=VisionTask.TEXT_TO_IMAGE,
            prompt="Test prompt",
            user_id="test_user"
        )
        
        # Create job
        job = VisionJob(
            job_id="test_job_123",
            request=request,
            state=VisionJobState.CREATED
        )
        
        assert job.job_id == "test_job_123"
        assert job.state == VisionJobState.CREATED
        assert job.request.prompt == "Test prompt"
        
        # Test state transition
        job.transition_to(VisionJobState.QUEUED, "Job queued for processing")
        assert job.state == VisionJobState.QUEUED
        assert len(job.state_history) == 2  # Created + Queued
        
        # Test progress update
        job.update_progress(50, "Processing...")
        assert job.progress_percentage == 50
        assert job.progress_message == "Processing..."
        
        print("✅ VisionJob creation and state management works correctly")
        return True
    except Exception as e:
        print(f"❌ VisionJob test failed: {e}")
        return False

def test_vision_error():
    """Test VisionError handling"""
    try:
        from bot.vision.types import VisionError, VisionErrorType
        
        error = VisionError(
            error_type=VisionErrorType.PROVIDER_ERROR,
            message="Provider is unavailable",
            user_message="The image generation service is temporarily unavailable."
        )
        
        assert error.error_type == VisionErrorType.PROVIDER_ERROR
        assert error.message == "Provider is unavailable"
        assert error.user_message == "The image generation service is temporarily unavailable."
        
        print("✅ VisionError handling works correctly")
        return True
    except Exception as e:
        print(f"❌ VisionError test failed: {e}")
        return False

def main():
    """Run all vision types tests"""
    print("🧪 Testing Vision System Types\n")
    
    tests = [
        test_vision_types_import,
        test_vision_enums,
        test_vision_request,
        test_vision_response,
        test_vision_job,
        test_vision_error
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All vision types tests passed!")
        return 0
    else:
        print("💥 Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
