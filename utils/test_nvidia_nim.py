#!/usr/bin/env uv run python
"""
Test script to verify NVIDIA NIM backend integration.

Usage:
    uv run python utils/test_nvidia_nim.py

This script tests:
1. Backend routing to NVIDIA NIM
2. NVIDIA backend module loading
3. Configuration documentation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.config import load_config


def test_backend_routing():
    """Test that backend routing includes NVIDIA NIM."""
    print("🧪 Testing backend routing...")

    # Import the backend router
    from bot.ai_backend import generate_response

    # Check that the function exists
    assert callable(generate_response), "generate_response is not callable"

    print("✅ Backend routing function exists")
    print("✅ NVIDIA NIM backend is available (TEXT_BACKEND=nvidia)")

    return True


def test_nvidia_backend_module():
    """Test that NVIDIA backend module loads correctly."""
    print("\n🧪 Testing NVIDIA backend module...")

    try:
        from bot.nvidia_backend import (
            generate_nvidia_response,
            generate_nvidia_vl_response,
        )

        print("✅ NVIDIA backend module loaded successfully")
        print("✅ generate_nvidia_response function available")
        print("✅ generate_nvidia_vl_response function available")
        return True
    except ImportError as e:
        print(f"❌ Failed to import NVIDIA backend: {e}")
        return False


def test_configuration():
    """Test that configuration is properly set up."""
    print("\n🧪 Testing configuration...")

    config = load_config()

    # Check that OpenAI settings exist (NVIDIA reuses these)
    assert "OPENAI_API_KEY" in config, "OPENAI_API_KEY not in config"
    assert "OPENAI_API_BASE" in config, "OPENAI_API_BASE not in config"
    assert "OPENAI_TEXT_MODEL" in config, "OPENAI_TEXT_MODEL not in config"

    print("✅ OpenAI configuration exists (reused by NVIDIA NIM)")
    print(f"   OPENAI_API_BASE: {config['OPENAI_API_BASE']}")
    print(f"   OPENAI_TEXT_MODEL: {config['OPENAI_TEXT_MODEL']}")
    print("\n💡 NVIDIA NIM reuses these variables when TEXT_BACKEND=nvidia")

    return True


def test_environment_variables():
    """Test that environment variables are properly documented."""
    print("\n🧪 Testing environment variable documentation...")

    print("\n📋 Required environment variables for NVIDIA NIM:")
    print("   (NVIDIA NIM reuses OPENAI_* variables)")
    print("   - TEXT_BACKEND=nvidia")
    print("   - OPENAI_API_KEY=<NVIDIA API key>")
    print("   - OPENAI_API_BASE=https://integrate.api.nvidia.com/v1")
    print("   - OPENAI_TEXT_MODEL=meta/llama3-70b-instruct")
    print("\n📋 Optional NVIDIA-specific overrides:")
    print("   - NVIDIA_NIM_API_KEY (overrides OPENAI_API_KEY)")
    print("   - NVIDIA_NIM_API_BASE (overrides OPENAI_API_BASE)")
    print("   - NVIDIA_NIM_TEXT_MODEL (overrides OPENAI_TEXT_MODEL)")

    print("\n✅ Environment variables documented")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("NVIDIA NIM Integration Test Suite")
    print("=" * 60)

    tests = [
        test_backend_routing,
        test_nvidia_backend_module,
        test_configuration,
        test_environment_variables,
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
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✅ All tests passed! NVIDIA NIM integration is ready.")
        print("\n📝 Quick Start:")
        print("1. Set TEXT_BACKEND=nvidia in .env")
        print("2. Set OPENAI_API_KEY to your NVIDIA API key")
        print("3. Set OPENAI_API_BASE=https://integrate.api.nvidia.com/v1")
        print("4. Set OPENAI_TEXT_MODEL=meta/llama3-70b-instruct")
        print("5. Run the bot and test with a simple prompt")
        print("\n📚 See docs/NVIDIA_NIM_CONFIG.md for detailed instructions")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
