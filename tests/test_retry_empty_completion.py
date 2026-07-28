"""Empty VL completions must stay retryable so the ladder keeps going. [REH]

Regression for: OpenRouter free VL models return HTTP 200 with no assistant
text. `openai_backend` raises `APIError("VL model <m> returned empty
completion")`, but `EnhancedRetryManager._is_retryable_error` only knew the
text path's phrasing ("empty text response"), so the VL wording was classified
non-retryable. That logged a scary ERROR traceback, skipped the model's
remaining attempts, and — because the non-retryable branch returns before
`_record_failure` — never tripped the circuit breaker on a model that is
persistently empty.
"""

from __future__ import annotations

from bot.enhanced_retry import EnhancedRetryManager, ProviderConfig, ProviderStatus
from bot.exceptions import APIError

EMPTY_VL_MESSAGE = "VL model nvidia/nemotron-nano-12b-v2-vl:free returned empty completion"


class TestEmptyCompletionRetryable:
    def test_vl_empty_completion_is_retryable(self) -> None:
        mgr = EnhancedRetryManager()
        assert mgr._is_retryable_error(APIError(EMPTY_VL_MESSAGE)) is True

    def test_text_empty_response_phrasings_are_retryable(self) -> None:
        mgr = EnhancedRetryManager()
        for msg in (
            "Model returned empty text response",
            "Provider returned empty response body",
        ):
            assert mgr._is_retryable_error(APIError(msg)) is True, msg

    def test_auth_failure_still_non_retryable(self) -> None:
        """The widened allowlist must not soften hard config failures."""
        mgr = EnhancedRetryManager()
        assert mgr._is_retryable_error(APIError("Error code: 401 - invalid api key")) is False

    async def test_empty_completion_falls_back_to_next_model(self) -> None:
        mgr = EnhancedRetryManager()
        mgr.provider_configs["vision"] = [
            ProviderConfig("openrouter", "empty-model", timeout=5.0, max_attempts=2, base_delay=0.0, jitter=False),
            ProviderConfig("openrouter", "good-model", timeout=5.0, max_attempts=1, base_delay=0.0, jitter=False),
        ]
        calls: list[str] = []

        def factory(provider_config):
            async def run():
                calls.append(provider_config.model)
                if provider_config.model == "empty-model":
                    raise APIError(EMPTY_VL_MESSAGE)
                return {"text": "a cat"}

            return run

        result = await mgr.run_with_fallback("vision", factory, per_item_budget=30.0)

        assert result.success is True
        assert result.result == {"text": "a cat"}
        # Both attempts on the empty model were spent (not skipped as non-retryable)
        assert calls == ["empty-model", "empty-model", "good-model"]

    async def test_repeated_empty_completions_open_the_circuit(self) -> None:
        """Failures now get recorded, so a permanently empty model gets benched."""
        mgr = EnhancedRetryManager()
        mgr.provider_configs["vision"] = [
            ProviderConfig("openrouter", "empty-model", timeout=5.0, max_attempts=3, base_delay=0.0, jitter=False),
        ]

        def factory(provider_config):
            async def run():
                raise APIError(EMPTY_VL_MESSAGE)

            return run

        result = await mgr.run_with_fallback("vision", factory, per_item_budget=30.0)

        assert result.success is False
        breaker = mgr._get_circuit_breaker("openrouter:empty-model")
        assert breaker.failure_count >= breaker.failure_threshold
        assert breaker.status is ProviderStatus.CIRCUIT_OPEN
