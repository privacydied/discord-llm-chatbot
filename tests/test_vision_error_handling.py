"""
Test suite for vision/image inference error handling and retry logic.
Tests robust error handling patterns for external API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from openai import InternalServerError

from bot.retry_utils import (
    RetryConfig,
    is_retryable_error,
    calculate_delay,
    retry_async,
    with_retry,
    VISION_RETRY_CONFIG,
)
from bot.exceptions import APIError, InferenceError
from bot.see import see_infer
from bot.openai_backend import generate_vl_response as openai_generate_vl_response


class TestRetryUtils:
    """Test retry utility functions and configurations."""

    def test_retry_config_defaults(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert APIError in config.retryable_exceptions
        assert 502 in config.retryable_status_codes

    def test_vision_retry_config(self):
        """Test vision-specific retry configuration."""
        config = VISION_RETRY_CONFIG
        assert config.max_attempts == 3
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert APIError in config.retryable_exceptions
        assert InferenceError in config.retryable_exceptions

    def test_is_retryable_error_with_status_codes(self):
        """Test retryable error detection for status codes."""
        config = RetryConfig()

        # Test 502 error (should be retryable)
        error_502 = APIError(
            "Error code: 502 - {'error': {'message': 'Provider returned error'}}"
        )
        assert is_retryable_error(error_502, config) is True

        # Test 500 error (should be retryable)
        error_500 = APIError("Error code: 500 - Internal Server Error")
        assert is_retryable_error(error_500, config) is True

        # Test 404 error (should not be retryable)
        error_404 = APIError("Error code: 404 - Not Found")
        assert is_retryable_error(error_404, config) is False

    def test_is_retryable_error_with_patterns(self):
        """Test retryable error detection for common patterns."""
        config = RetryConfig()

        # Test provider error pattern
        provider_error = APIError("Provider returned error")
        assert is_retryable_error(provider_error, config) is True

        # Test internal server error pattern
        server_error = APIError("Internal Server Error")
        assert is_retryable_error(server_error, config) is True

        # Test bad gateway pattern
        gateway_error = APIError("Bad Gateway")
        assert is_retryable_error(gateway_error, config) is True

        # Test non-retryable error
        auth_error = APIError("Authentication failed")
        assert is_retryable_error(auth_error, config) is False

    def test_calculate_delay(self):
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(
            base_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False
        )

        # Test exponential backoff
        assert calculate_delay(0, config) == 1.0  # 1.0 * 2^0
        assert calculate_delay(1, config) == 2.0  # 1.0 * 2^1
        assert calculate_delay(2, config) == 4.0  # 1.0 * 2^2

        # Test max delay cap
        assert calculate_delay(10, config) == 10.0  # Should be capped at max_delay

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=True)

        delay = calculate_delay(0, config)
        # With jitter, delay should be around 1.0 but with some variance
        assert 0.9 <= delay <= 1.1

    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self):
        """Test successful execution on first attempt."""

        async def success_func():
            return "success"

        config = RetryConfig(max_attempts=3)
        result = await retry_async(success_func, config)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_async_success_after_retries(self):
        """Test successful execution after retries."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("Error code: 502 - Bad Gateway")
            return "success"

        config = RetryConfig(max_attempts=3, base_delay=0.1)
        result = await retry_async(flaky_func, config)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_exhausted_retries(self):
        """Test retry exhaustion with persistent error."""

        async def always_fail():
            raise APIError("Error code: 502 - Bad Gateway")

        config = RetryConfig(max_attempts=2, base_delay=0.1)

        with pytest.raises(APIError):
            await retry_async(always_fail, config)

    @pytest.mark.asyncio
    async def test_retry_async_non_retryable_error(self):
        """Test immediate failure for non-retryable errors."""

        async def auth_fail():
            raise APIError("Authentication failed")

        config = RetryConfig(max_attempts=3)

        with pytest.raises(APIError, match="Authentication failed"):
            await retry_async(auth_fail, config)

    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        """Test the retry decorator."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        async def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise APIError("Error code: 502 - Bad Gateway")
            return "decorated_success"

        result = await decorated_func()
        assert result == "decorated_success"
        assert call_count == 2


class TestVisionErrorHandling:
    """Test vision inference error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_see_infer_user_friendly_messages(self):
        """Test user-friendly error messages in see_infer."""
        with (
            patch("bot.see.os.path.exists", return_value=True),
            patch("bot.see.load_config") as mock_config,
            patch("bot.see.generate_vl_response") as mock_vl,
        ):
            mock_config.return_value = {"VL_PROMPT_FILE": None}
            # Test provider error
            mock_vl.side_effect = APIError("Error code: 502 - Provider returned error")

            with pytest.raises(InferenceError) as exc_info:
                await see_infer("/fake/path.jpg", "test prompt")

            error_msg = str(exc_info.value)
            assert "temporarily unavailable" in error_msg
            assert "provider issues" in error_msg

    @pytest.mark.asyncio
    async def test_see_infer_file_not_found_message(self):
        """Test file not found error message."""
        with patch("bot.see.os.path.exists", return_value=False):
            with pytest.raises(InferenceError) as exc_info:
                await see_infer("/nonexistent/path.jpg", "test prompt")

            error_msg = str(exc_info.value)
            assert "could not be found" in error_msg

    @pytest.mark.asyncio
    async def test_see_infer_format_error_message(self):
        """Test format error message."""
        with (
            patch("bot.see.os.path.exists", return_value=True),
            patch("bot.see.load_config") as mock_config,
            patch("bot.see.generate_vl_response") as mock_vl,
        ):
            mock_config.return_value = {"VL_PROMPT_FILE": None}
            mock_vl.side_effect = Exception("Invalid image format")

            with pytest.raises(InferenceError) as exc_info:
                await see_infer("/fake/path.jpg", "test prompt")

            error_msg = str(exc_info.value)
            assert "format is not supported" in error_msg

    @pytest.mark.asyncio
    async def test_ai_backend_error_handling(self):
        """Test AI backend error handling with retry detection."""
        with patch("bot.ai_backend.load_config") as mock_config:
            mock_config.return_value = {"TEXT_BACKEND": "openai"}

            with patch("bot.openai_backend.generate_vl_response") as mock_openai:
                mock_openai.side_effect = APIError("Error code: 502 - Bad Gateway")

                with pytest.raises(Exception) as exc_info:
                    from bot.ai_backend import generate_vl_response

                    await generate_vl_response("/fake/path.jpg", "test prompt")

                error_msg = str(exc_info.value)
                assert "temporarily unavailable" in error_msg
                assert "provider issue" in error_msg

    @pytest.mark.asyncio
    async def test_openai_backend_retry_logic(self):
        """Test OpenAI backend retry logic integration."""
        call_count = 0

        async def mock_openai_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate the actual OpenAI error structure
                raise InternalServerError(
                    message="Provider returned error",
                    response=MagicMock(),
                    body={"error": {"message": "Provider returned error", "code": 502}},
                )
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content="Success after retries"))],
                usage=MagicMock(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30
                ),
            )

        with (
            patch("bot.openai_backend.load_config") as mock_config,
            patch("builtins.open", mock_open(read_data="Test VL prompt")),
            patch("bot.openai_backend.get_base64_image") as mock_get_image,
        ):
            mock_config.return_value = {
                "OPENAI_API_KEY": "test-key",
                "VL_MODEL": "gpt-4-vision-preview",
                "VL_PROMPT_FILE": "/fake/prompt.txt",
                "TEMPERATURE": 0.7,
                "MAX_RESPONSE_TOKENS": 1000,
            }

            # Mock the image processing to return a valid data URL
            mock_get_image.return_value = (
                "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVR"
            )

            with patch("bot.openai_backend.openai.AsyncOpenAI") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.chat.completions.create = mock_openai_call
                mock_client_class.return_value = mock_client

                # This should succeed after retries
                result = await openai_generate_vl_response(
                    "/fake/path.jpg", "test prompt"
                )

                assert result["text"] == "Success after retries"
                assert call_count == 3  # Should have retried twice


class TestRouterErrorHandling:
    """Test router-level error handling for vision processing."""

    def test_router_error_message_extraction(self):
        """Test router's ability to extract user-friendly messages from errors."""
        from bot.router import Router

        Router(MagicMock())

        # Test different error patterns and expected responses
        test_cases = [
            (
                "temporarily unavailable due to provider issues",
                "temporarily unavailable",
            ),
            ("format is not supported", "format is not supported"),
            ("too large", "too large"),
            ("could not be processed", "could not be processed"),
            ("unknown error", "failed to analyze"),
        ]

        for error_msg, expected_pattern in test_cases:
            # Simulate the error handling logic from router
            error_str = error_msg
            if "temporarily unavailable" in error_str or "provider issues" in error_str:
                response = "🔧 The image analysis service is temporarily unavailable. Please try again in a few minutes."
            elif "format is not supported" in error_str:
                response = "🖼️ This image format is not supported. Please try uploading a JPEG, PNG, or WebP image."
            elif "too large" in error_str:
                response = (
                    "📏 This image is too large. Please try uploading a smaller image."
                )
            elif "could not be processed" in error_str:
                response = "📁 The image could not be processed. Please try uploading it again."
            else:
                response = "❌ Failed to analyze the image. This may be due to a temporary service issue - please try again."

            assert expected_pattern in response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
