"""Tests for vision ladder fallback via EnhancedRetryManager.
Verifies that VISION_FALLBACK_MODELS ladder is used for VL calls.
"""

from typing import Never

import httpx
import pytest

from bot.enhanced_retry import EnhancedRetryManager, ProviderConfig


def make_httpx_429(retry_after: float = 1.0) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(429, headers={"Retry-After": str(retry_after)}, request=request)
    return httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)


@pytest.mark.asyncio
async def test_enhanced_retry_manager_vision_ladder_fallback() -> None:
    """Test that vision ladder falls back from first to second provider."""
    mgr = EnhancedRetryManager()
    mgr.circuit_breakers.clear()

    # Configure vision ladder: first fails, second succeeds
    mgr.provider_configs["vision"] = [
        ProviderConfig("openrouter", "vision-fail-a", timeout=2.0, max_attempts=1),
        ProviderConfig("openrouter", "vision-ok-b", timeout=2.0, max_attempts=1),
    ]

    def factory(pc: ProviderConfig):
        async def run():
            if pc.model == "vision-fail-a":
                msg = "429 Too Many Requests"
                raise Exception(msg)
            return {
                "text": "Vision OK",
                "model": pc.model,
                "usage": {"total_tokens": 100},
                "backend": "openai",
            }

        return run

    res = await mgr.run_with_fallback("vision", factory, per_item_budget=10.0)

    assert res.success is True
    assert res.provider_used.endswith(":vision-ok-b")
    assert res.attempts == 2
    assert res.fallback_occurred is True
    assert res.result["text"] == "Vision OK"
    assert res.result["model"] == "vision-ok-b"


@pytest.mark.asyncio
async def test_enhanced_retry_manager_vision_ladder_all_fail() -> None:
    """Test that vision ladder exhaustion is properly reported."""
    mgr = EnhancedRetryManager()
    mgr.circuit_breakers.clear()

    # Configure vision ladder: both providers fail
    mgr.provider_configs["vision"] = [
        ProviderConfig("openrouter", "vision-fail-a", timeout=1.0, max_attempts=1),
        ProviderConfig("openrouter", "vision-fail-b", timeout=1.0, max_attempts=1),
    ]

    def factory(pc: ProviderConfig):
        async def run() -> Never:
            msg = "503 Service Unavailable"
            raise Exception(msg)

        return run

    res = await mgr.run_with_fallback("vision", factory, per_item_budget=5.0)

    assert res.success is False
    assert res.error is not None
    assert res.fallback_occurred is True
    assert res.attempts >= 2


@pytest.mark.asyncio
async def test_enhanced_retry_manager_vision_single_provider_success() -> None:
    """Test that vision ladder works with a single provider that succeeds."""
    mgr = EnhancedRetryManager()
    mgr.circuit_breakers.clear()

    mgr.provider_configs["vision"] = [
        ProviderConfig("openrouter", "vision-single", timeout=5.0, max_attempts=2),
    ]

    def factory(pc: ProviderConfig):
        async def run():
            return {
                "text": "Single provider OK",
                "model": pc.model,
                "usage": {"total_tokens": 50},
                "backend": "openai",
            }

        return run

    res = await mgr.run_with_fallback("vision", factory, per_item_budget=10.0)

    assert res.success is True
    assert res.provider_used.endswith(":vision-single")
    assert res.attempts == 1
    assert res.fallback_occurred is False


@pytest.mark.asyncio
async def test_vision_ladder_respects_per_provider_timeouts() -> None:
    """Test that per-provider timeouts from ladder config are respected."""
    mgr = EnhancedRetryManager()
    mgr.circuit_breakers.clear()

    # Configure with different timeouts
    mgr.provider_configs["vision"] = [
        ProviderConfig("openrouter", "fast-model", timeout=2.0, max_attempts=1),
        ProviderConfig("openrouter", "slow-model", timeout=10.0, max_attempts=1),
    ]

    timeouts_observed = []

    def factory(pc: ProviderConfig):
        timeouts_observed.append(pc.timeout)

        async def run():
            if pc.model == "fast-model":
                msg = "429 Too Many Requests"
                raise Exception(msg)
            return {"text": "OK", "model": pc.model}

        return run

    res = await mgr.run_with_fallback("vision", factory, per_item_budget=20.0)

    assert res.success is True
    # Both providers should have been attempted
    assert 2.0 in timeouts_observed
    assert 10.0 in timeouts_observed


@pytest.mark.asyncio
async def test_vision_ladder_circuit_breaker_skips_failed_provider() -> None:
    """Test that circuit breaker skips providers that have recently failed."""
    mgr = EnhancedRetryManager()
    mgr.circuit_breakers.clear()

    mgr.provider_configs["vision"] = [
        ProviderConfig("openrouter", "flaky-model", timeout=2.0, max_attempts=1),
        ProviderConfig("openrouter", "stable-model", timeout=2.0, max_attempts=1),
    ]

    call_count = {"flaky": 0, "stable": 0}

    def factory(pc: ProviderConfig):
        async def run():
            if pc.model == "flaky-model":
                call_count["flaky"] += 1
                msg = "500 Internal Server Error"
                raise Exception(msg)
            call_count["stable"] += 1
            return {"text": "OK", "model": pc.model}

        return run

    # First call: flaky fails, stable succeeds
    res1 = await mgr.run_with_fallback("vision", factory, per_item_budget=10.0)
    assert res1.success is True
    assert call_count["flaky"] == 1
    assert call_count["stable"] == 1

    # Force circuit breaker to open by recording more failures
    flaky_key = "openrouter:flaky-model"
    mgr._record_failure(flaky_key)
    mgr._record_failure(flaky_key)  # Should trigger circuit open

    # Second call: should skip flaky (circuit open) and go directly to stable
    res2 = await mgr.run_with_fallback("vision", factory, per_item_budget=10.0)
    assert res2.success is True
    # flaky should not have been called again (circuit open)
    assert call_count["flaky"] == 1
    assert call_count["stable"] == 2


def test_env_vision_ladder_is_authoritative_and_not_clobbered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = EnhancedRetryManager()

    # Provide 3 env models; regression used to reset ladder to default_vision[:2]
    monkeypatch.setenv(
        "VISION_FALLBACK_MODELS",
        "openrouter|env-vl-1,openrouter|env-vl-2,openrouter|env-vl-3",
    )
    monkeypatch.setenv("VL_MODEL", "")
    # Auto-discovery outranks the env ladder by design; disable it for this test.
    monkeypatch.setenv("VISION_AUTO_DISCOVERY", "0")

    # Ensure config cache doesn't hide monkeypatched env
    try:
        from bot.config import invalidate_config_cache

        invalidate_config_cache()
    except Exception:
        pass

    summary = mgr.refresh_from_env()
    assert summary["vision"] == ["env-vl-1", "env-vl-2", "env-vl-3"]
