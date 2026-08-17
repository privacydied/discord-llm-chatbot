"""Retired models are benched, not re-walked on every request. [REH][PA]

A model that has been retired upstream answers in milliseconds with 410 Gone /
404 "unavailable for free" / "model not found". That is far too fast to trip the
timeout-based circuit breaker, so before this the ladder re-discovered the same
corpse on every single message — and each dead rung it walked ate wall-clock
budget that the live rungs below it never got.
"""

from __future__ import annotations

import logging

import pytest

from bot.ai_backend import is_handled_provider_failure
from bot.enhanced_retry import (
    EnhancedRetryManager,
    ProviderConfig,
    ProviderStatus,
    is_dead_model_error,
)
from bot.exceptions import APIError

# Real upstream payloads observed in production logs.
GONE_410 = (
    "APIStatusError: Error code: 410 - {'type': 'about:blank', 'title': 'Gone', 'status': 410, "
    "'detail': \"The model 'deepseek-ai/deepseek-v4-pro' has reached its end of life on "
    '2026-08-07T09:00:00Z and is no longer available."}'
)
FREE_SLUG_RETIRED = "NotFoundError: Error code: 404 - {'error': {'message': 'This model is unavailable for free. The paid version is available now - use this slug instead: inclusionai/ling-3.0-flash', 'code': 404}}"
NO_ENDPOINTS = "NotFoundError: Error code: 404 - {'error': {'message': 'No endpoints found for foo/bar:free.'}}"


class TestDeadModelDetection:
    @pytest.mark.parametrize(
        "message",
        [GONE_410, FREE_SLUG_RETIRED, NO_ENDPOINTS, "Error code: 404 - model not found", "unknown model: foo/bar"],
    )
    def test_permanent_failures_detected(self, message: str) -> None:
        assert is_dead_model_error(APIError(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Error code: 429 - rate limit exceeded",
            "Error code: 503 - service unavailable",
            "LLM attempt timed out after 45.0s for openrouter:foo",
            "Error code: 500 - internal server error",
        ],
    )
    def test_transient_failures_not_treated_as_dead(self, message: str) -> None:
        assert is_dead_model_error(APIError(message)) is False

    @pytest.mark.parametrize("message", [GONE_410, FREE_SLUG_RETIRED])
    def test_dead_model_is_not_retryable(self, message: str) -> None:
        assert EnhancedRetryManager()._is_retryable_error(APIError(message)) is False


def _two_rung_ladder(dead_error: Exception) -> tuple[EnhancedRetryManager, callable, dict[str, int]]:
    mgr = EnhancedRetryManager()
    mgr.provider_configs["text"] = [
        ProviderConfig("nvidia", "dead-model", timeout=1.0, max_attempts=2, base_delay=0.0, jitter=False),
        ProviderConfig("openrouter", "live-model", timeout=1.0, max_attempts=2, base_delay=0.0, jitter=False),
    ]
    calls = {"dead": 0, "live": 0}

    def factory(provider_config: ProviderConfig):
        async def run() -> str:
            if provider_config.model == "dead-model":
                calls["dead"] += 1
                raise dead_error
            calls["live"] += 1
            return "OK"

        return run

    return mgr, factory, calls


class TestDeadModelBenching:
    @pytest.mark.parametrize("message", [GONE_410, FREE_SLUG_RETIRED])
    async def test_dead_rung_is_benched_and_skipped_next_request(self, message: str, monkeypatch) -> None:
        monkeypatch.setenv("DEAD_MODEL_COOLDOWN_S", "1800")
        mgr, factory, calls = _two_rung_ladder(APIError(message))

        first = await mgr.run_with_fallback("text", factory, per_item_budget=5.0)

        assert first.success is True
        # One attempt only: the remaining attempt for a retired model is wasted budget.
        assert calls["dead"] == 1

        breaker = mgr._get_circuit_breaker("nvidia:dead-model")
        assert breaker.status == ProviderStatus.CIRCUIT_OPEN
        assert breaker.cooldown_duration >= 1800.0

        second = await mgr.run_with_fallback("text", factory, per_item_budget=5.0)

        assert second.success is True
        assert calls["dead"] == 1, "benched model must not be retried on the next request"
        assert calls["live"] == 2

    async def test_legacy_cooldown_env_still_honoured(self, monkeypatch) -> None:
        monkeypatch.delenv("DEAD_MODEL_COOLDOWN_S", raising=False)
        monkeypatch.setenv("OPENROUTER_DEAD_MODEL_COOLDOWN_S", "900")
        mgr, factory, _ = _two_rung_ladder(APIError(GONE_410))

        await mgr.run_with_fallback("text", factory, per_item_budget=5.0)

        assert mgr._get_circuit_breaker("nvidia:dead-model").cooldown_duration >= 900.0

    async def test_bench_logs_warning_without_traceback(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="bot.enhanced_retry")
        mgr, factory, _ = _two_rung_ladder(APIError(GONE_410))

        await mgr.run_with_fallback("text", factory, per_item_budget=5.0)

        benched = [r for r in caplog.records if "retired/unavailable" in r.message]
        assert benched, "expected the bench decision to be logged"
        assert all(r.levelno == logging.WARNING and r.exc_info is None for r in benched)
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestHandledFailureClassification:
    @pytest.mark.parametrize(
        "message",
        [
            f"Text fallback ladder failed after 4 attempt(s) in 90.10s (last_provider=nvidia:x): {GONE_410}",
            "Text model retired/unavailable (last_provider=nvidia:x) — update TEXT_FALLBACK_MODELS in .env.",
            "All text providers exhausted",
            "Per-item budget of 45.0s exceeded",
        ],
    )
    def test_ladder_exhaustion_is_handled_not_a_crash(self, message: str) -> None:
        assert is_handled_provider_failure(APIError(message)) is True

    def test_genuine_bug_still_reported_loudly(self) -> None:
        assert is_handled_provider_failure(TypeError("'NoneType' object is not subscriptable")) is False
