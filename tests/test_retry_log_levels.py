"""Handled ladder fallbacks log as warnings, not ERROR tracebacks. [REH]

A non-retryable provider error and a 404/no-endpoints model are both *handled*
routing decisions — the ladder benches that provider and moves to the next one.
Logging them via `logger.exception` printed a full traceback at ERROR, which
reads as a crash in the console and buries the genuinely fatal cases (auth
failure, which aborts the whole ladder and stays at ERROR).
"""

from __future__ import annotations

import logging

from bot.enhanced_retry import EnhancedRetryManager, ProviderConfig
from bot.exceptions import APIError


def _ladder(*errors: Exception) -> tuple[EnhancedRetryManager, callable]:
    mgr = EnhancedRetryManager()
    mgr.provider_configs["vision"] = [ProviderConfig("openrouter", f"model-{i}", timeout=5.0, max_attempts=1, base_delay=0.0, jitter=False) for i in range(len(errors))]
    by_model = {f"model-{i}": err for i, err in enumerate(errors)}

    def factory(provider_config):
        async def run():
            raise by_model[provider_config.model]

        return run

    return mgr, factory


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == level]


class TestLadderLogLevels:
    async def test_non_retryable_error_logs_warning_without_traceback(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="bot.enhanced_retry")
        mgr, factory = _ladder(APIError("Error code: 400 - malformed image payload"))

        result = await mgr.run_with_fallback("vision", factory, per_item_budget=30.0)

        assert result.success is False
        skipped = [r for r in caplog.records if "skipping remaining attempts" in r.message]
        assert skipped, "expected the non-retryable skip to be logged"
        assert all(r.levelno == logging.WARNING for r in skipped)
        assert all(r.exc_info is None for r in skipped)
        assert not _records(caplog, logging.ERROR)

    async def test_404_no_endpoints_logs_warning_without_traceback(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="bot.enhanced_retry")
        mgr, factory = _ladder(APIError("Error code: 404 - No endpoints found for qwen/qwen-vl:free"))

        result = await mgr.run_with_fallback("vision", factory, per_item_budget=30.0)

        assert result.success is False
        benched = [r for r in caplog.records if "benching" in r.message]
        assert benched, "expected the dead-model bench to be logged"
        assert all(r.levelno == logging.WARNING for r in benched)
        assert all(r.exc_info is None for r in benched)
        assert not _records(caplog, logging.ERROR)

    async def test_auth_failure_still_logs_error_with_traceback(self, caplog) -> None:
        """Aborts the whole ladder and needs operator action — stays loud."""
        caplog.set_level(logging.WARNING, logger="bot.enhanced_retry")
        mgr, factory = _ladder(APIError("Error code: 401 - invalid api key"))

        result = await mgr.run_with_fallback("vision", factory, per_item_budget=30.0)

        assert result.success is False
        errors = _records(caplog, logging.ERROR)
        assert any("Authentication failure" in r.message for r in errors)
        assert any(r.exc_info is not None for r in errors)
