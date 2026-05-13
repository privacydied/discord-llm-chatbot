"""Tests for bot/logging_enforcer.py — warning suppression and SuppressingLogger.

Phase 18: Logging reductions — prevent repeated warning spam.
"""

import logging
import time


from bot.logging_enforcer import SuppressingLogger, _is_warning_suppressed, _warning_last_seen


class TestIsWarningSuppressed:
    """Test the module-level suppression gate used by SuppressingLogger."""

    def setup_method(self):
        _warning_last_seen.clear()

    def test_first_call_not_suppressed(self):
        assert _is_warning_suppressed("first msg") is False

    def test_same_msg_immediately_suppressed(self):
        assert _is_warning_suppressed("dup") is False
        assert _is_warning_suppressed("dup") is True

    def test_different_msgs_not_suppressed(self):
        assert _is_warning_suppressed("a") is False
        assert _is_warning_suppressed("b") is False
        assert _is_warning_suppressed("c") is False

    def test_suppression_expires_after_window(self):

        assert _is_warning_suppressed("expires") is False
        # Simulate time passage by injecting a past timestamp
        import bot.logging_enforcer as mod

        mod._warning_last_seen["expires"] = time.monotonic() - mod._SUPPRESS_WINDOW - 1
        assert _is_warning_suppressed("expires") is False


class TestSuppressingLogger:
    """Test the SuppressingLogger subclass exists and is a logger."""

    def test_is_logger_subclass(self):
        assert issubclass(SuppressingLogger, logging.Logger)

    def test_can_be_instantiated(self):
        logger = SuppressingLogger("test_sl_instantiated")
        assert logger.name == "test_sl_instantiated"
