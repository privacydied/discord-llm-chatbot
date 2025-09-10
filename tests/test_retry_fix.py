#!/usr/bin/env python3
"""
Test script to verify that the retry logic is working for 502 provider errors.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bot"))

from bot.retry_utils import is_retryable_error, VISION_RETRY_CONFIG
from bot.exceptions import APIError


def test_retry_logic():
    """Test that 502 errors are properly detected as retryable."""

    # Test 1: 502 error should be retryable
    error_502 = APIError(
        "Error code: 502 - {'error': {'message': 'Provider returned error', 'code': 502}}"
    )
    assert is_retryable_error(error_502, VISION_RETRY_CONFIG), (
        "502 error should be retryable"
    )
    print("✅ Test 1 passed: 502 errors are detected as retryable")

    # Test 2: Simulate the actual error format from the logs
    actual_error_msg = "Error code: 502 - {'error': {'message': 'Provider returned error', 'code': 502, 'metadata': {'raw': '<html>\r\n<head><title>500 Internal Server Error</title></head>\r\n<body>\r\n<center><h1>500 Internal Server Error</h1></center>\r\n<hr><center>nginx</center>\r\n</body>\r\n</html>\r\n', 'provider_name': 'Chutes'}}, 'user_id': 'user_2sEKmio7Kouj8hyL0Z9Setxz3y9'}"
    api_error_502 = APIError(f"Failed to generate VL response: {actual_error_msg}")
    assert is_retryable_error(api_error_502, VISION_RETRY_CONFIG), (
        "Actual 502 error should be retryable"
    )
    print("✅ Test 2 passed: Actual 502 errors are detected as retryable")

    # Test 3: "Provider returned error" pattern should be retryable
    provider_error = APIError("Provider returned error from Chutes")
    assert is_retryable_error(provider_error, VISION_RETRY_CONFIG), (
        "Provider error should be retryable"
    )
    print("✅ Test 3 passed: Provider errors are detected as retryable")

    # Test 4: Non-retryable error should not be retryable
    auth_error = APIError("Invalid API key")
    assert not is_retryable_error(auth_error, VISION_RETRY_CONFIG), (
        "Auth error should not be retryable"
    )
    print("✅ Test 4 passed: Auth errors are not retryable")

    print("\n🎉 All retry logic tests passed!")


if __name__ == "__main__":
    test_retry_logic()
