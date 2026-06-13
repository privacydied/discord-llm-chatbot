#!/usr/bin/env python3
"""Test Vision system integration without Discord dependencies.
Validates the router integration pattern and method signatures.
"""

from pathlib import Path


def test_vision_integration() -> None:
    """Test Vision integration points without full Discord stack [CDiP]."""
    project_root = Path(__file__).resolve().parents[1]
    router_path = project_root / "bot" / "router.py"
    router_content = router_path.read_text()

    required_methods = [
        "_handle_vision_generation",
        "_monitor_vision_job",
    ]

    for method in required_methods:
        assert f"def {method}" in router_content, f"Required method {method} missing from router.py"


def test_vision_types_import() -> None:
    """Test that vision types module loads without error."""
    from bot.vision.types import (
        IntentDecision,
        IntentResult,
        VisionError,
        VisionErrorType,
        VisionJob,
        VisionJobState,
        VisionProvider,
        VisionRequest,
        VisionResponse,
        VisionTask,
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
