#!/usr/bin/env python3
"""Vision Types Standalone Test - Test without bot package imports.

Direct testing of vision types without going through bot package
to avoid Discord dependency issues.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_direct_types_import() -> None:
    """Test importing vision types directly."""
    sys.path.insert(0, str(project_root / "bot" / "vision"))

    # Import the enums and classes directly from vision/types.py
    from bot.vision.types import VisionError, VisionErrorType, VisionJob, VisionJobState, VisionProvider, VisionRequest, VisionResponse, VisionTask

    _ = (
        VisionTask,
        VisionProvider,
        VisionJobState,
        VisionError,
        VisionErrorType,
        VisionRequest,
        VisionResponse,
        VisionJob,
    )


def test_enum_values() -> None:
    """Test enum values work correctly."""
    from bot.vision.types import VisionJobState, VisionProvider, VisionTask

    # Test VisionTask
    assert VisionTask.TEXT_TO_IMAGE.value == "text_to_image"
    assert VisionTask.IMAGE_TO_IMAGE.value == "image_to_image"
    assert VisionTask.TEXT_TO_VIDEO.value == "text_to_video"
    assert VisionTask.IMAGE_TO_VIDEO.value == "image_to_video"

    # Test VisionProvider
    assert VisionProvider.TOGETHER.value == "together"
    assert VisionProvider.NOVITA.value == "novita"

    # Test VisionJobState
    assert VisionJobState.CREATED.value == "created"
    assert VisionJobState.QUEUED.value == "queued"
    assert VisionJobState.RUNNING.value == "running"
    assert VisionJobState.COMPLETED.value == "completed"
    assert VisionJobState.FAILED.value == "failed"
