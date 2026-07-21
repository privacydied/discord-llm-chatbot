"""All-circuits-open must yield a half-open probe, not instant muteness. [REH]

Regression for: 60s breaker cooldown + a burst of provider failures opened
every circuit → every request for the next minute failed instantly with
"Text fallback ladder failed after 0 attempt(s) in 0.00s".
"""

from __future__ import annotations

import time

from bot.enhanced_retry import (
    CircuitBreakerState,
    EnhancedRetryManager,
    ProviderConfig,
    ProviderStatus,
)


def _manager_with_open_circuits() -> tuple[EnhancedRetryManager, list[str]]:
    mgr = EnhancedRetryManager()
    providers = [
        ProviderConfig("openrouter", "model-a", timeout=5.0, max_attempts=1),
        ProviderConfig("openrouter", "model-b", timeout=5.0, max_attempts=1),
    ]
    mgr.provider_configs["text"] = providers
    keys = [f"{p.name}:{p.model}" for p in providers]
    for key in keys:
        mgr.circuit_breakers[key] = CircuitBreakerState(
            status=ProviderStatus.CIRCUIT_OPEN,
            failure_count=2,
            last_failure_time=time.time(),  # cooldown just started
            cooldown_duration=60.0,
            failure_threshold=2,
        )
    return mgr, keys


class TestAllCircuitsOpenProbe:
    async def test_probe_allows_one_attempt_and_can_succeed(self) -> None:
        mgr, _ = _manager_with_open_circuits()

        calls = []

        def factory(provider_config):
            async def run():
                calls.append(f"{provider_config.name}:{provider_config.model}")
                return "hello"

            return run

        result = await mgr.run_with_fallback("text", factory, per_item_budget=30.0)

        assert result.success is True
        assert result.result == "hello"
        assert len(calls) == 1  # exactly one probe, not a full hammer

    async def test_failed_probe_reopens_circuit(self) -> None:
        mgr, keys = _manager_with_open_circuits()

        def factory(provider_config):
            async def run():
                msg = "still down"
                raise ConnectionError(msg)

            return run

        result = await mgr.run_with_fallback("text", factory, per_item_budget=30.0)

        assert result.success is False
        # The probed provider's breaker must be open again (failure_count was
        # left one below threshold, so a single failure re-trips it).
        probed = [k for k in keys if mgr.circuit_breakers[k].status == ProviderStatus.CIRCUIT_OPEN]
        assert len(probed) == len(keys)

    async def test_no_probe_when_a_circuit_is_healthy(self) -> None:
        mgr, keys = _manager_with_open_circuits()
        mgr.circuit_breakers[keys[1]] = CircuitBreakerState()  # healthy

        mgr._ensure_probe_when_all_circuits_open(mgr.provider_configs["text"])

        # The open breaker must stay untouched — no probe needed.
        assert mgr.circuit_breakers[keys[0]].status == ProviderStatus.CIRCUIT_OPEN
