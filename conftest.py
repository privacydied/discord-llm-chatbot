"""Top-level pytest configuration.

Hooks
-----
pytest_collection_modifyitems: skip tests requiring local Kokoro ONNX / voices
assets when they are absent (e.g. CI runners that don't store the 340 MB model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence  # noqa: F401

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent

_KOKORO_FILES = [
    _PROJECT_ROOT / "tts" / "kokoro-v1.0.onnx",
    _PROJECT_ROOT / "tts" / "voices-v1.0.bin",
]


def _has_kokoro_assets() -> bool:
    return all(p.exists() for p in _KOKORO_FILES)


_HAS_KOKORO = _has_kokoro_assets()


def pytest_collection_modifyitems(config, items):
    """Skip tests marked 'needs_kokoro_assets' when model files are missing."""
    if _HAS_KOKORO:
        return

    skipper = pytest.mark.skip(reason="requires local TTS model/voices files")
    for item in items:
        if "needs_kokoro_assets" in item.keywords:
            item.add_marker(skipper)


def pytest_configure(config: "pytest.Config") -> None:  # noqa: ARG001
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "needs_kokoro_assets: tests that require local TTS model/voices files",
    )
