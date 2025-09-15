#!/usr/bin/env python3
"""
Simulate EnhancedRetryManager provider fallback and retry behavior in isolation.
Scenarios:
- Retryable failure on primary provider then success on fallback.
- Non-retryable error short-circuits retries.
- Budget exhaustion skips long operations.
- Circuit breaker opens and skips degraded provider on subsequent runs.

Run with:
    uv run python scripts/test_vl_fallback.py
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Callable, Dict, Tuple

from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel

# Project imports
from bot.enhanced_retry import get_retry_manager, ProviderConfig

# -----------------------------
# Logging Setup (Dual Sinks) [RAT][REH]
# -----------------------------
ICON_BY_LEVEL = {
    "DEBUG": "🔎",
    "INFO": "ℹ",
    "WARNING": "⚠",
    "ERROR": "✖",
    "CRITICAL": "✖",
}


class JSONLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level": record.levelname,
            "name": record.name,
            # Preserve keys per user rules; fill with None if absent
            "subsys": getattr(record, "subsys", None),
            "guild_id": getattr(record, "guild_id", None),
            "user_id": getattr(record, "user_id", None),
            "msg_id": getattr(record, "msg_id", None),
            "event": getattr(record, "event", record.getMessage()[:24]),
            "detail": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("vl_fallback_test")
    logger.setLevel(logging.DEBUG)

    # Pretty Rich console handler
    console = Console(force_terminal=True)
    pretty_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        log_time_format="%H:%M:%S.%f",
    )
    pretty_handler.setLevel(logging.INFO)

    # Structured JSON Lines handler to stdout
    json_handler = logging.StreamHandler(stream=sys.stdout)
    json_handler.setLevel(logging.DEBUG)
    json_handler.setFormatter(JSONLineFormatter())

    # Clear existing and attach both
    logger.handlers.clear()
    logger.addHandler(pretty_handler)
    logger.addHandler(json_handler)

    # Enforcer: ensure both handlers present [RAT]
    assert any(isinstance(h, RichHandler) for h in logger.handlers), (
        "pretty_handler missing"
    )
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers), (
        "jsonl_handler missing"
    )

    logger.info("Logging configured with dual sinks (Rich + JSONL)")
    return logger


# -----------------------------
# Scenario Helpers
# -----------------------------
def configure_test_ladder():
    mgr = get_retry_manager()
    # Reset circuit breakers to isolate scenarios [REH][PA]
    mgr.circuit_breakers.clear()
    # Short timeouts/attempts for fast tests [PA]
    mgr.provider_configs["vision"] = [
        ProviderConfig(
            name="testprov1",
            model="model-a",
            timeout=0.8,
            max_attempts=2,
            base_delay=0.2,
            max_delay=0.6,
            exponential_base=1.5,
        ),
        ProviderConfig(
            name="testprov2",
            model="model-b",
            timeout=0.8,
            max_attempts=2,
            base_delay=0.2,
            max_delay=0.6,
            exponential_base=1.5,
        ),
    ]
    logging.getLogger("vl_fallback_test").info(
        "Configured test ladder: %s",
        ", ".join(f"{pc.name}:{pc.model}" for pc in mgr.provider_configs["vision"]),
    )
    return mgr


def make_handler_fn(
    logger: logging.Logger, behavior: str
) -> Callable[[ProviderConfig], Callable[[], str]]:
    """
    Return a coro factory matching EnhancedRetryManager signature that simulates different behaviors.
    behavior: one of ["retryable_fail_primary", "non_retryable", "slow_op", "always_fail", "succeed_on_second_attempt"]
    """
    # shared state to simulate per-provider attempts across retries [REH]
    state: Dict[Tuple[str, str], int] = {}

    def factory(provider_cfg: ProviderConfig):
        async def run():
            prov = provider_cfg.name
            if behavior == "retryable_fail_primary":
                if prov == "testprov1":
                    # 502 retryable error
                    raise Exception(
                        "502 Provider returned error: temporary upstream issue"
                    )
                return f"OK from {prov}"
            elif behavior == "non_retryable":
                # Non-retryable (auth) error
                raise Exception("authentication failed: invalid api key")
            elif behavior == "slow_op":
                # Sleep beyond timeout to invoke timeout handling
                await asyncio.sleep(provider_cfg.timeout + 0.5)
                return f"finished after sleep on {prov}"
            elif behavior == "always_fail":
                raise Exception("502 Provider returned error")
            elif behavior == "always_fail_primary_only":
                if prov == "testprov1":
                    raise Exception("502 Provider returned error")
                return f"OK from {prov}"
            elif behavior == "succeed_on_second_attempt":
                key = (behavior, prov)
                cnt = state.get(key, 0) + 1
                state[key] = cnt
                if cnt == 1:
                    raise Exception("503 Service unavailable: try again")
                return f"OK after retry on {prov}"
            else:
                return f"OK default {prov}"

        return run

    return factory


async def scenario_retryable_then_fallback(logger: logging.Logger):
    mgr = configure_test_ladder()
    res = await mgr.run_with_fallback(
        "vision", make_handler_fn(logger, "retryable_fail_primary"), per_item_budget=3.0
    )
    logger.info(
        f"Scenario retryable_then_fallback -> success={res.success}, provider={res.provider_used}, attempts={res.attempts}, time={res.total_time:.2f}s"
    )


async def scenario_non_retryable(logger: logging.Logger):
    mgr = configure_test_ladder()
    res = await mgr.run_with_fallback(
        "vision", make_handler_fn(logger, "non_retryable"), per_item_budget=3.0
    )
    logger.info(
        f"Scenario non_retryable -> success={res.success}, error={res.error}, attempts={res.attempts}, time={res.total_time:.2f}s"
    )


async def scenario_budget_exhaustion(logger: logging.Logger):
    mgr = configure_test_ladder()
    # Very small budget to force skip [PA]
    res = await mgr.run_with_fallback(
        "vision", make_handler_fn(logger, "slow_op"), per_item_budget=0.7
    )
    logger.info(
        f"Scenario budget_exhaustion -> success={res.success}, error={res.error}, attempts={res.attempts}, time={res.total_time:.2f}s"
    )


async def scenario_circuit_breaker_skip(logger: logging.Logger):
    mgr = configure_test_ladder()
    # First run: open circuit only for primary provider
    _ = await mgr.run_with_fallback(
        "vision",
        make_handler_fn(logger, "always_fail_primary_only"),
        per_item_budget=2.0,
    )
    # Second run should skip primary and succeed on fallback provider
    res = await mgr.run_with_fallback(
        "vision", make_handler_fn(logger, "retryable_fail_primary"), per_item_budget=3.0
    )
    logger.info(
        f"Scenario circuit_breaker_skip -> success={res.success}, provider={res.provider_used}, attempts={res.attempts}, time={res.total_time:.2f}s"
    )


async def scenario_retry_within_provider(logger: logging.Logger):
    mgr = configure_test_ladder()
    res = await mgr.run_with_fallback(
        "vision",
        make_handler_fn(logger, "succeed_on_second_attempt"),
        per_item_budget=3.0,
    )
    logger.info(
        f"Scenario retry_within_provider -> success={res.success}, provider={res.provider_used}, attempts={res.attempts}, time={res.total_time:.2f}s"
    )


async def main():
    logger = setup_logging()

    console = Console()
    console.print(
        Panel.fit(
            "EnhancedRetryManager Fallback/Retry Simulator",
            subtitle="VL pipeline validation",
            style="bold green",
        )
    )

    tasks = [
        scenario_retryable_then_fallback(logger),
        scenario_non_retryable(logger),
        scenario_budget_exhaustion(logger),
        scenario_circuit_breaker_skip(logger),
        scenario_retry_within_provider(logger),
    ]

    # Run sequentially to keep logs grouped [CA]
    for t in tasks:
        start = time.time()
        try:
            await t
        except Exception as e:
            logger.error(f"Scenario raised exception: {type(e).__name__}: {e}")
        finally:
            elapsed = time.time() - start
            logger.info(f"Scenario completed in {elapsed:.2f}s")
            print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
