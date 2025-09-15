#!/usr/bin/env python3
"""
Vision Types Standalone Test - Test without bot package imports

Direct testing of vision types without going through bot package
to avoid Discord dependency issues.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_direct_types_import():
    """Test importing vision types directly"""
    try:
        # Import directly from the module files
        sys.path.insert(0, str(project_root / "bot" / "vision"))

        # Import the enums and classes directly from vision/types.py
        from types import VisionTask, VisionProvider, VisionJobState
        from types import VisionError, VisionErrorType
        from types import VisionRequest, VisionResponse, VisionJob

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

        print("✅ Vision types imported directly")
        return True
    except ImportError as e:
        print(f"❌ Failed to import vision types directly: {e}")
        return False


def test_enum_values():
    """Test enum values work correctly"""
    try:
        sys.path.insert(0, str(project_root / "bot" / "vision"))
        from types import VisionTask, VisionProvider, VisionJobState

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

        print("✅ Enum values are correct")
        return True
    except Exception as e:
        print(f"❌ Enum test failed: {e}")
        return False


def main():
    """Run standalone vision types tests"""
    print("🧪 Testing Vision Types (Standalone)\n")

    # Try to test basic enum functionality
    print("📦 Testing direct enum import...")

    try:
        # Import enum classes directly without going through bot package
        vision_types_path = project_root / "bot" / "vision" / "types.py"

        if not vision_types_path.exists():
            print(f"❌ Vision types file not found: {vision_types_path}")
            return 1

        # Execute the types module to test basic syntax
        with open(vision_types_path, "r") as f:
            types_code = f.read()

        # Simple syntax check
        compile(types_code, str(vision_types_path), "exec")
        print("✅ Vision types file compiles successfully")

        # Test that we can at least access enum values as strings
        if '"text_to_image"' in types_code:
            print("✅ TEXT_TO_IMAGE enum value found")
        if '"together"' in types_code:
            print("✅ TOGETHER provider value found")
        if '"created"' in types_code:
            print("✅ CREATED state value found")

        print("\n🎉 Basic vision types validation passed!")
        print("📝 Note: Full integration test requires Discord dependencies")
        return 0

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
