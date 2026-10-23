"""Expected vision failures log as one-line warnings, not ERROR + traceback [REH]."""

from __future__ import annotations

from bot.vision.types import EXPECTED_VISION_ERRORS, VisionErrorType


def test_clean_provider_outcomes_are_expected():
    for kind in (VisionErrorType.QUOTA_EXCEEDED, VisionErrorType.CONTENT_FILTERED, VisionErrorType.RATE_LIMITED):
        assert kind in EXPECTED_VISION_ERRORS


def test_real_failures_keep_tracebacks():
    for kind in (
        VisionErrorType.SYSTEM_ERROR,
        VisionErrorType.PROVIDER_ERROR,
        VisionErrorType.SERVER_ERROR,
        VisionErrorType.AUTHENTICATION_ERROR,
        VisionErrorType.UNKNOWN_ERROR,
    ):
        assert kind not in EXPECTED_VISION_ERRORS
