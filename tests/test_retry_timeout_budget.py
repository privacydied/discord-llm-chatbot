"""A hung provider must not eat the budget its fallbacks need. [PA][REH]

Replays the 2026-07-28 14:39:02 text ladder from logs/bot.jsonl:

    nemotron-3-ultra   attempt 1  timeout 45.0s
    nemotron-3-ultra   attempt 2  timeout 45.0s   <-- 90s of a 120s budget
    laguna-s-2.1       429 x2                     (~0.2s)
    ling-3.0-flash     429 x2                     (~1.3s)
    deepseek-v4-pro    attempt 1, truncated to 25.8s of its 45s -> timed out
    => "Text fallback ladder failed after 7 attempt(s) in 120.00s
        (last_provider=unknown): TimeoutError: Per-item budget of 120.0s exceeded"

Two defects: the hung model was retried in place (a timeout has already proven
the window is too short), and the synthesized budget error dropped both the
provider attribution and the real last error.
"""

from __future__ import annotations

import asyncio

from bot.enhanced_retry import EnhancedRetryManager, ProviderConfig
from bot.exceptions import APIError


def _text_ladder(mgr: EnhancedRetryManager) -> None:
    mgr.provider_configs["text"] = [
        ProviderConfig("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", timeout=45.0, max_attempts=2, base_delay=0.0, jitter=False),
        ProviderConfig("openrouter", "poolside/laguna-s-2.1:free", timeout=45.0, max_attempts=2, base_delay=0.0, jitter=False),
        ProviderConfig("openrouter", "nvidia/deepseek-v4-pro", timeout=45.0, max_attempts=2, base_delay=0.0, jitter=False),
    ]


class TestTimeoutDoesNotStarveFallbacks:
    async def test_timeout_falls_through_instead_of_retrying_in_place(self) -> None:
        mgr = EnhancedRetryManager()
        _text_ladder(mgr)
        attempts: list[str] = []

        def factory(provider_config):
            async def run():
                attempts.append(provider_config.model)
                if provider_config.model.startswith("nvidia/nemotron"):
                    await asyncio.sleep(30)  # hangs past its (shortened) window
                if "laguna" in provider_config.model:
                    raise APIError("Error code: 429 - Provider returned error")
                return "answer"

            return run

        # Shrink the ladder's clock so the test runs fast: 0.2s windows.
        for pc in mgr.provider_configs["text"]:
            pc.timeout = 0.2

        result = await mgr.run_with_fallback("text", factory, per_item_budget=5.0)

        assert result.success is True
        assert result.result == "answer"
        # The hung model gets exactly ONE window, not two.
        assert attempts.count("nvidia/nemotron-3-ultra-550b-a55b:free") == 1
        # The fast 429 still retries in place — it costs nothing.
        assert attempts.count("poolside/laguna-s-2.1:free") == 2
        assert attempts[-1] == "nvidia/deepseek-v4-pro"

    async def test_last_rung_keeps_its_retries(self) -> None:
        """With nothing left to fall through to, retrying is still the best move."""
        mgr = EnhancedRetryManager()
        mgr.provider_configs["text"] = [
            ProviderConfig("openrouter", "only-model", timeout=0.2, max_attempts=2, base_delay=0.0, jitter=False),
        ]
        attempts: list[str] = []

        def factory(provider_config):
            async def run():
                attempts.append(provider_config.model)
                if len(attempts) == 1:
                    await asyncio.sleep(30)
                return "recovered"

            return run

        result = await mgr.run_with_fallback("text", factory, per_item_budget=5.0)

        assert result.success is True
        assert attempts == ["only-model", "only-model"]

    async def test_opt_out_restores_in_place_retry(self, monkeypatch) -> None:
        monkeypatch.setenv("LADDER_TIMEOUT_SKIPS_RETRY", "0")
        mgr = EnhancedRetryManager()
        mgr.provider_configs["text"] = [
            ProviderConfig("openrouter", "hung", timeout=0.2, max_attempts=2, base_delay=0.0, jitter=False),
            ProviderConfig("openrouter", "good", timeout=0.2, max_attempts=1, base_delay=0.0, jitter=False),
        ]
        attempts: list[str] = []

        def factory(provider_config):
            async def run():
                attempts.append(provider_config.model)
                if provider_config.model == "hung":
                    await asyncio.sleep(30)
                return "ok"

            return run

        await mgr.run_with_fallback("text", factory, per_item_budget=5.0)

        assert attempts.count("hung") == 2


class TestBudgetErrorAttribution:
    async def test_budget_error_names_provider_and_real_cause(self) -> None:
        mgr = EnhancedRetryManager()
        mgr.provider_configs["text"] = [
            ProviderConfig("openrouter", "slow-a", timeout=0.3, max_attempts=2, base_delay=5.0, jitter=False),
        ]

        def factory(provider_config):
            async def run():
                await asyncio.sleep(30)

            return run

        result = await mgr.run_with_fallback("text", factory, per_item_budget=0.5)

        assert result.success is False
        # Was "unknown" — openai_backend renders this as last_provider=<...>
        assert result.provider_used == "openrouter:slow-a"
        assert getattr(result.error, "provider_key", None) == "openrouter:slow-a"
        msg = str(result.error)
        assert "Per-item budget" in msg
        assert "last error" in msg  # the real cause survives, not just "budget exceeded"
        assert "timed out" in msg.lower()
        assert result.error.__cause__ is not None
