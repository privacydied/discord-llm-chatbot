"""
Simple smoke test for environment variables and multimodal functionality.
CHANGE: Simplified version without pytest for basic validation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import bot modules
from bot.config import (
    load_config,
    validate_required_env,
    validate_prompt_files,
    check_venv_activation,
)
from bot.events import has_image_attachments, get_image_urls


def test_env_and_model_integration():
    """
    Integration smoke test to verify .env→code mapping and multimodal branch coverage.
    """
    print("\n🧪 Running comprehensive environment and multimodal smoke tests...\n")

    # Test 1: Environment variables
    print("1. Testing environment variable loading...")
    config = load_config()
    assert config is not None
    print("   ✅ Configuration loaded successfully")

    # Test 2: Required environment variables
    print("2. Testing required environment variables...")
    try:
        validate_required_env()
        print("   ✅ All required environment variables present")
    except Exception as e:
        print(f"   ❌ Missing required environment variables: {e}")
        return False

    # Test 3: Prompt files
    print("3. Testing prompt file accessibility...")
    try:
        validate_prompt_files()
        prompt_file = config.get("PROMPT_FILE")
        vl_prompt_file = config.get("VL_PROMPT_FILE")
        print(f"   ✅ Text prompt file: {prompt_file}")
        print(f"   ✅ VL prompt file: {vl_prompt_file}")
    except Exception as e:
        print(f"   ❌ Prompt file validation failed: {e}")
        return False

    # Test 4: Image detection logic
    print("4. Testing image detection logic...")
    mock_msg_with_image = MagicMock()
    mock_attachment = MagicMock()
    mock_attachment.filename = "test.png"
    mock_attachment.url = "https://example.com/test.png"
    mock_msg_with_image.attachments = [mock_attachment]

    mock_msg_no_image = MagicMock()
    mock_msg_no_image.attachments = []

    has_image = has_image_attachments(mock_msg_with_image)
    no_image = has_image_attachments(mock_msg_no_image)
    image_urls = get_image_urls(mock_msg_with_image)

    assert has_image is True, "Should detect image attachments"
    assert no_image is False, "Should not detect images when none present"
    assert len(image_urls) == 1, "Should extract image URLs"
    assert image_urls[0] == "https://example.com/test.png", "Should return correct URL"
    print("   ✅ Image detection logic works correctly")

    # Test 5: Model configuration
    print("5. Testing model configuration...")
    text_model = config.get("OPENAI_TEXT_MODEL")
    vl_model = config.get("VL_MODEL")

    assert text_model is not None, "Text model not configured"
    assert vl_model is not None, "VL model not configured"
    print(f"   ✅ Text model: {text_model}")
    print(f"   ✅ VL model: {vl_model}")

    # Test 6: .venv check
    print("6. Testing .venv enforcement...")
    try:
        check_venv_activation()
        print("   ✅ .venv enforcement check completed")
    except Exception as e:
        print(f"   ⚠️  .venv warning (may be expected): {e}")

    print("\n🎉 All smoke tests passed! The hybrid multimodal system is ready.\n")
    return True


if __name__ == "__main__":
    success = test_env_and_model_integration()
    if success:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
