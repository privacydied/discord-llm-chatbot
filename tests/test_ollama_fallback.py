"""Tests for Ollama fallback integration."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from bot.enhanced_retry import (
    EnhancedRetryManager,
    ProviderConfig,
)


def _full_config(overrides=None):
    d = {
        "TEXT_FALLBACK_MODELS": None,
        "TEXT_FALLBACK_TIMEOUTS": None,
        "TEXT_FALLBACK_MAX_ATTEMPTS": None,
        "VISION_FALLBACK_MODELS": None,
        "VISION_FALLBACK_TIMEOUTS": None,
        "MEDIA_PROVIDER_TIMEOUT": None,
        "OLLAMA_HOST": None,
        "OLLAMA_MODEL": None,
        "OLLAMA_TIMEOUT": None,
    }
    if overrides:
        d.update(overrides)
    return d


def test_ollama_provider_config_defaults():
    """Ollama provider config has correct default values."""
    pc = ProviderConfig("ollama", "llama3", timeout=45.0, max_attempts=1)
    assert pc.name == "ollama"
    assert pc.model == "llama3"
    assert pc.timeout == 45.0


def test_circuit_state_isolated():
    """Circuit breaker state for ollama must not affect openrouter."""
    config = _full_config()
    with patch("bot.enhanced_retry.load_config", return_value=config):
        with patch.dict(os.environ, {}, clear=False):
            # Clear the TEXT_FALLBACK env vars to use defaults
            mgr = EnhancedRetryManager()
            mgr._record_failure("ollama:llama3")
            mgr._record_failure("ollama:llama3")
            assert not mgr._is_provider_available("ollama:llama3")
            assert mgr._is_provider_available("openrouter:whatever")


@pytest.mark.asyncio
async def test_run_with_fallback_ollama_success():
    """run_with_fallback succeeds when ollama is the only provider."""
    config = _full_config()
    with patch("bot.enhanced_retry.load_config", return_value=config):
        mgr = EnhancedRetryManager()
        mgr.provider_configs["text"] = [
            ProviderConfig("ollama", "llama3", timeout=5.0, max_attempts=1),
        ]

        async def _fake_ok():
            return "Hello from Ollama"

        rr = await mgr.run_with_fallback(
            "text", lambda pc: _fake_ok, per_item_budget=10.0
        )
        assert rr.success
        assert rr.result == "Hello from Ollama"
        assert rr.provider_used == "ollama:llama3"


@pytest.mark.asyncio
async def test_run_with_fallback_ollama_timeout():
    """run_with_fallback fails when ollama times out and no fallback exists."""
    config = _full_config()
    with patch("bot.enhanced_retry.load_config", return_value=config):
        mgr = EnhancedRetryManager()
        mgr.provider_configs["text"] = [
            ProviderConfig("ollama", "llama3", timeout=5.0, max_attempts=1),
        ]

        async def _fail_timeout():
            raise asyncio.TimeoutError()

        rr = await mgr.run_with_fallback(
            "text", lambda pc: _fail_timeout, per_item_budget=10.0
        )
        assert not rr.success
        assert rr.provider_used == "ollama:llama3"


@pytest.mark.asyncio
async def test_run_with_fallback_ollama_connection_error():
    """run_with_fallback fails when ollama connection is refused."""
    config = _full_config()
    with patch("bot.enhanced_retry.load_config", return_value=config):
        mgr = EnhancedRetryManager()
        mgr.provider_configs["text"] = [
            ProviderConfig("ollama", "llama3", timeout=5.0, max_attempts=1),
        ]

        async def _fail_conn():
            raise ConnectionError("refused")

        rr = await mgr.run_with_fallback(
            "text", lambda pc: _fail_conn, per_item_budget=10.0
        )
        assert not rr.success


@pytest.mark.asyncio
async def test_run_with_fallback_ollama_no_fallback_available():
    """When ollama fails and no fallback exists, error should identify ollama as last provider."""
    config = _full_config()
    with patch("bot.enhanced_retry.load_config", return_value=config):
        mgr = EnhancedRetryManager()
        mgr.provider_configs["text"] = [
            ProviderConfig("ollama", "llama3", timeout=5.0, max_attempts=1),
        ]

        async def _fail():
            raise ConnectionError("nope")

        rr = await mgr.run_with_fallback("text", lambda pc: _fail, per_item_budget=10.0)
        assert not rr.success
        assert rr.provider_used == "ollama:llama3"
        assert rr.attempts == 1
