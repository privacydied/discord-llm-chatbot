#!/usr/bin/env python3
"""
Test Vision system integration without Discord dependencies.
Validates the router integration pattern and method signatures.
"""

from pathlib import Path


def test_vision_integration():
    """Test Vision integration points without full Discord stack [CDiP]"""
    router_path = Path("/volume1/py/discord-llm-chatbot/bot/router.py")
    router_content = router_path.read_text()

    required_methods = [
        "_handle_vision_generation",
        "_monitor_vision_job",
    ]

    for method in required_methods:
        assert f"def {method}" in router_content, f"Required method {method} missing from router.py"


def test_vision_types_import():
    """Test that vision types module loads without error"""
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

    # Touch imported symbols to avoid unused-import warnings
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
