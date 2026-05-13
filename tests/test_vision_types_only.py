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
    from bot.vision.types import (
        VisionTask,
        VisionProvider,
        VisionJobState,
        VisionError,
        VisionErrorType,
        VisionRequest,
        VisionResponse,
        VisionJob,
        IntentDecision,
        IntentResult,
    )

    # Touch imported symbols to avoid unused-import warnings in this smoke test
    _ = (
        VisionTask,
        VisionProvider,
        VisionJobState,
        VisionError,
        VisionErrorType,
        VisionRequest,
        VisionResponse,
        VisionJob,
        IntentDecision,
        IntentResult,
    )


def test_vision_enums():
    """Test vision enums"""
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


def test_vision_request():
    """Test VisionRequest creation"""
    from bot.vision.types import VisionRequest, VisionTask, VisionProvider

    request = VisionRequest(
        task=VisionTask.TEXT_TO_IMAGE,
        prompt="A beautiful sunset",
        user_id="test_user_123",
        preferred_provider=VisionProvider.TOGETHER,
    )

    assert request.task == VisionTask.TEXT_TO_IMAGE
    assert request.prompt == "A beautiful sunset"
    assert request.user_id == "test_user_123"
    assert request.preferred_provider == VisionProvider.TOGETHER
    assert request.width == 1024  # Default value
    assert request.height == 1024  # Default value


def test_vision_response():
    """Test VisionResponse creation"""
    from bot.vision.types import VisionResponse, VisionProvider
    from pathlib import Path

    response = VisionResponse(
        provider=VisionProvider.TOGETHER,
        success=True,
        job_id="job_123",
        model_used="test-model",
        artifacts=[Path("/tmp/image.png")],
        processing_time_seconds=5.2,
    )

    assert response.provider == VisionProvider.TOGETHER
    assert response.success
    assert len(response.artifacts) == 1
    assert response.artifacts[0] == Path("/tmp/image.png")
    assert response.processing_time_seconds == 5.2


def test_vision_job():
    """Test VisionJob creation and state transitions"""
    from bot.vision.types import (
        VisionJob,
        VisionRequest,
        VisionTask,
        VisionJobState,
    )

    # Create request
    request = VisionRequest(task=VisionTask.TEXT_TO_IMAGE, prompt="Test prompt", user_id="test_user")

    # Create job
    job = VisionJob(job_id="test_job_123", request=request, state=VisionJobState.CREATED)

    assert job.job_id == "test_job_123"
    assert job.state == VisionJobState.CREATED
    assert job.request.prompt == "Test prompt"

    # Test state transition
    job.transition_to(VisionJobState.QUEUED, "Job queued for processing")
    assert job.state == VisionJobState.QUEUED
    assert len(job.log_entries) >= 1  # Transition logged

    # Test progress update
    job.update_progress(50, "Processing...")
    assert job.progress_percentage == 50
    # Progress message is logged, not stored as a field


def test_vision_error():
    """Test VisionError handling"""
    from bot.vision.types import VisionError, VisionErrorType

    error = VisionError(
        error_type=VisionErrorType.PROVIDER_ERROR,
        message="Provider is unavailable",
        user_message="The image generation service is temporarily unavailable.",
    )

    assert error.error_type == VisionErrorType.PROVIDER_ERROR
    assert error.message == "Provider is unavailable"
    assert error.user_message == "The image generation service is temporarily unavailable."
