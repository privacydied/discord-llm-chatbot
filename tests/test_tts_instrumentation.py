"""Tests for TTS instrumentation utilities."""

import unittest
from pathlib import Path
from typing import Never
from unittest.mock import patch

import pytest

from bot.tts.instrumentation import (
    get_tts_metrics,
    log_cache_event,
    log_gibberish_detection,
    log_phonemiser_selection,
    log_tts_config,
    log_tts_error,
    log_tts_generation,
    log_voice_loading,
    reset_tts_metrics,
    timed_function,
)


class TestTTSInstrumentation(unittest.TestCase):
    """Test cases for TTS instrumentation utilities."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Reset metrics before each test
        reset_tts_metrics()

    def test_log_tts_config(self) -> None:
        """Test logging TTS configuration."""
        with patch("logging.Logger.info") as mock_info:
            # Test with regular config
            config = {
                "TTS_LANGUAGE": "en",
                "TTS_VOICE": "test_voice",
                "TTS_BACKEND": "kokoro",
                "TTS_MODEL_PATH": "/path/to/model",
            }
            log_tts_config(config)

            # Check that info was logged
            assert mock_info.called
            assert mock_info.call_count == 4  # Main config + 3 specific items

            # Test with sensitive info
            config_with_sensitive = {
                "TTS_LANGUAGE": "en",
                "API_KEY": "secret_key",
                "PASSWORD": "secret_password",
            }
            mock_info.reset_mock()
            log_tts_config(config_with_sensitive)

            # Check that sensitive info was filtered
            assert mock_info.called
            for call_args in mock_info.call_args_list:
                _args, kwargs = call_args
                if "extra" in kwargs and "config" in kwargs["extra"]:
                    assert "API_KEY" not in kwargs["extra"]["config"]
                    assert "PASSWORD" not in kwargs["extra"]["config"]

    def test_log_phonemiser_selection(self) -> None:
        """Test logging phonemiser selection."""
        with patch("logging.Logger.info") as mock_info:
            available = {
                "espeak": True,
                "phonemizer": False,
                "g2p_en": True,
            }
            log_phonemiser_selection("en", "espeak", available)

            # Check that info was logged
            assert mock_info.called
            args, kwargs = mock_info.call_args
            assert "espeak" in args[0]
            assert "en" in args[0]
            assert kwargs["extra"]["phonemiser"] == "espeak"
            assert kwargs["extra"]["language"] == "en"
            assert kwargs["extra"]["available"] == available

    def test_log_voice_loading(self) -> None:
        """Test logging voice loading."""
        with patch("logging.Logger.debug") as mock_debug:
            log_voice_loading("test_voice", (256,), 1.234)

            # Check that debug was logged
            assert mock_debug.called
            args, kwargs = mock_debug.call_args
            assert "test_voice" in args[0]
            assert "(256,)" in args[0]
            assert kwargs["extra"]["voice_id"] == "test_voice"
            assert kwargs["extra"]["vector_shape"] == (256,)
            self.assertAlmostEqual(kwargs["extra"]["vector_norm"], 1.234)

    def test_log_tts_generation(self) -> None:
        """Test logging TTS generation."""
        with patch("logging.Logger.info") as mock_info:
            # Get initial metrics
            initial_metrics = get_tts_metrics()

            # Log a TTS generation
            log_tts_generation("This is a test text", "test_voice", Path("/tmp/test_output.wav"), 150.5)

            # Check that info was logged
            assert mock_info.called
            args, kwargs = mock_info.call_args
            assert "test_voice" in args[0]
            assert "150.5" in args[0]
            assert kwargs["extra"]["voice_id"] == "test_voice"
            assert kwargs["extra"]["text_length"] == 19
            assert kwargs["extra"]["output_path"] == "/tmp/test_output.wav"
            self.assertAlmostEqual(kwargs["extra"]["duration_ms"], 150.5)

            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            assert updated_metrics["tts_generation_count"] == initial_metrics["tts_generation_count"] + 1
            self.assertAlmostEqual(
                updated_metrics["tts_generation_time_total"],
                initial_metrics["tts_generation_time_total"] + 0.1505,
            )

    def test_log_tts_error(self) -> None:
        """Test logging TTS error."""
        with patch("logging.Logger.error") as mock_error:
            # Get initial metrics
            initial_metrics = get_tts_metrics()

            # Log a TTS error
            log_tts_error("TestError", "This is a test error", {"detail": "test_detail"})

            # Check that error was logged
            assert mock_error.called
            args, kwargs = mock_error.call_args
            assert "TestError" in args[0]
            assert "This is a test error" in args[0]
            assert kwargs["extra"]["error_type"] == "TestError"
            assert kwargs["extra"]["error_message"] == "This is a test error"
            assert kwargs["extra"]["detail"] == "test_detail"

            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            assert updated_metrics["tts_generation_errors"] == initial_metrics["tts_generation_errors"] + 1

    def test_log_gibberish_detection(self) -> None:
        """Test logging gibberish detection."""
        with patch("logging.Logger.warning") as mock_warning:
            # Get initial metrics
            initial_metrics = get_tts_metrics()

            # Log a gibberish detection
            metrics = {
                "zero_crossing_rate": 0.5,
                "mean_amplitude": 0.001,
                "clipping_ratio": 0.0,
            }
            log_gibberish_detection(metrics)

            # Check that warning was logged
            assert mock_warning.called
            args, kwargs = mock_warning.call_args
            assert "Gibberish audio detected" in args[0]
            assert kwargs["extra"]["zero_crossing_rate"] == 0.5
            assert kwargs["extra"]["mean_amplitude"] == 0.001
            assert kwargs["extra"]["clipping_ratio"] == 0.0

            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            assert updated_metrics["tts_gibberish_detected"] == initial_metrics["tts_gibberish_detected"] + 1

    def test_log_cache_event(self) -> None:
        """Test logging cache events."""
        with patch("logging.Logger.debug") as mock_debug:
            # Get initial metrics
            initial_metrics = get_tts_metrics()

            # Log a cache hit
            log_cache_event("test_hash_1", True)

            # Check that debug was logged
            assert mock_debug.called
            args, kwargs = mock_debug.call_args
            assert "cache hit" in args[0]
            assert "test_hash_1" in args[0]
            assert kwargs["extra"]["text_hash"] == "test_hash_1"
            assert kwargs["extra"]["event"] == "cache.hit"

            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            assert updated_metrics["tts_cache_hits"] == initial_metrics["tts_cache_hits"] + 1

            # Reset mock
            mock_debug.reset_mock()

            # Log a cache miss
            log_cache_event("test_hash_2", False)

            # Check that debug was logged
            assert mock_debug.called
            args, kwargs = mock_debug.call_args
            assert "cache miss" in args[0]
            assert "test_hash_2" in args[0]
            assert kwargs["extra"]["text_hash"] == "test_hash_2"
            assert kwargs["extra"]["event"] == "cache.miss"

            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            assert updated_metrics["tts_cache_misses"] == initial_metrics["tts_cache_misses"] + 1

    def test_timed_function(self) -> None:
        """Test timed function decorator."""
        with patch("logging.Logger.debug") as mock_debug:
            # Define a test function
            @timed_function
            def test_func(x, y):
                return x + y

            # Call the function
            result = test_func(1, 2)

            # Check that the function worked correctly
            assert result == 3

            # Check that debug was logged
            assert mock_debug.called
            args, kwargs = mock_debug.call_args
            assert "test_func" in args[0]
            assert "executed" in args[0]
            assert kwargs["extra"]["function"] == "test_func"
            assert "duration_ms" in kwargs["extra"]

    def test_timed_function_error(self) -> None:
        """Test timed function decorator with error."""
        with patch("logging.Logger.error") as mock_error:
            # Define a test function that raises an exception
            @timed_function
            def test_func_error() -> Never:
                msg = "Test error"
                raise ValueError(msg)

            # Call the function and expect an exception
            with pytest.raises(ValueError):
                test_func_error()

            # Check that error was logged
            assert mock_error.called
            args, kwargs = mock_error.call_args
            assert "test_func_error" in args[0]
            assert "failed" in args[0]
            assert "Test error" in args[0]
            assert kwargs["extra"]["function"] == "test_func_error"
            assert "duration_ms" in kwargs["extra"]
            assert kwargs["extra"]["error"] == "Test error"

    def test_get_tts_metrics(self) -> None:
        """Test getting TTS metrics."""
        # Initial metrics should all be zero
        metrics = get_tts_metrics()
        assert metrics["tts_generation_count"] == 0
        assert metrics["tts_generation_time_total"] == 0.0
        assert metrics["tts_generation_errors"] == 0
        assert metrics["tts_gibberish_detected"] == 0
        assert metrics["tts_cache_hits"] == 0
        assert metrics["tts_cache_misses"] == 0
        assert metrics["tts_generation_time_avg"] == 0.0
        assert metrics["tts_cache_hit_rate"] == 0.0
        assert metrics["tts_error_rate"] == 0.0

        # Simulate some activity
        log_tts_generation("Test 1", "voice1", Path("/tmp/test1.wav"), 100.0)
        log_tts_generation("Test 2", "voice1", Path("/tmp/test2.wav"), 200.0)
        log_tts_error("TestError", "Test error")
        log_cache_event("hash1", True)
        log_cache_event("hash2", True)
        log_cache_event("hash3", False)

        # Check updated metrics
        metrics = get_tts_metrics()
        assert metrics["tts_generation_count"] == 2
        self.assertAlmostEqual(metrics["tts_generation_time_total"], 0.3)
        assert metrics["tts_generation_errors"] == 1
        assert metrics["tts_cache_hits"] == 2
        assert metrics["tts_cache_misses"] == 1
        self.assertAlmostEqual(metrics["tts_generation_time_avg"], 0.15)
        self.assertAlmostEqual(metrics["tts_cache_hit_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["tts_error_rate"], 0.5)

    def test_reset_tts_metrics(self) -> None:
        """Test resetting TTS metrics."""
        # Simulate some activity
        log_tts_generation("Test", "voice1", Path("/tmp/test.wav"), 100.0)
        log_tts_error("TestError", "Test error")
        log_cache_event("hash", True)

        # Check that metrics were updated
        metrics_before = get_tts_metrics()
        assert metrics_before["tts_generation_count"] > 0

        # Reset metrics
        reset_tts_metrics()

        # Check that metrics were reset
        metrics_after = get_tts_metrics()
        assert metrics_after["tts_generation_count"] == 0
        assert metrics_after["tts_generation_time_total"] == 0.0
        assert metrics_after["tts_generation_errors"] == 0
        assert metrics_after["tts_gibberish_detected"] == 0
        assert metrics_after["tts_cache_hits"] == 0
        assert metrics_after["tts_cache_misses"] == 0


if __name__ == "__main__":
    unittest.main()
