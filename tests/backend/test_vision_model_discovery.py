"""Tests for OpenRouter free vision-model auto-discovery. [REH]"""

import asyncio
import json

import pytest

from bot.vision import free_model_discovery as disc

CATALOGUE = {
    "data": [
        {
            "id": "vendor/good-vl:free",
            "created": 100,
            "context_length": 128000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "vendor/paid-vl",
            "created": 100,
            "context_length": 128000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.000002", "completion": "0"},
        },
        {
            "id": "vendor/content-safety:free",
            "created": 100,
            "context_length": 128000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "vendor/lyria-audio:free",
            "created": 100,
            "context_length": 128000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text", "audio"]},
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "vendor/text-only:free",
            "created": 100,
            "context_length": 128000,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "vendor/tiny-context:free",
            "created": 100,
            "context_length": 512,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ],
}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # Overrides the suite-wide autouse fixture that disables discovery.
    monkeypatch.setenv("VISION_DISCOVERY_CACHE_PATH", str(tmp_path / "models.json"))
    monkeypatch.setenv("VISION_DISCOVERY_QUARANTINE_PATH", str(tmp_path / "quarantine.json"))
    monkeypatch.setenv("VISION_AUTO_DISCOVERY", "1")
    monkeypatch.setenv("VISION_DISCOVERY_PROBE", "0")  # probe tests enable it explicitly
    # The real .env must not leak its ladder into these assertions.
    monkeypatch.delenv("VL_MODEL", raising=False)
    monkeypatch.delenv("VISION_FALLBACK_MODELS", raising=False)
    disc._reset_for_tests()
    yield
    disc._reset_for_tests()


def test_selection_filters_paid_safety_audio_and_text_only():
    models = disc._select_models(CATALOGUE, limit=10)
    assert models == ["vendor/good-vl:free"]


def test_selection_respects_limit():
    payload = {
        "data": [
            {
                "id": f"vendor/m{i}-vl:free",
                "created": i,
                "context_length": 128000,
                "architecture": {"input_modalities": ["image"], "output_modalities": ["text"]},
                "pricing": {"prompt": "0", "completion": "0"},
            }
            for i in range(10)
        ],
    }
    assert len(disc._select_models(payload, limit=3)) == 3


async def test_discover_writes_cache_and_memoizes(monkeypatch, tmp_path):
    async def fake_fetch(_timeout):
        return CATALOGUE

    monkeypatch.setattr(disc, "_fetch_catalogue", fake_fetch)
    models = await disc.discover_free_vision_models(force=True)
    assert models == ["vendor/good-vl:free"]

    cached = json.loads((tmp_path / "models.json").read_text())
    assert cached["models"] == ["vendor/good-vl:free"]
    assert disc.get_cached_free_vision_models() == ["vendor/good-vl:free"]


async def test_fetch_failure_keeps_previous_ladder(monkeypatch):
    async def ok_fetch(_timeout):
        return CATALOGUE

    monkeypatch.setattr(disc, "_fetch_catalogue", ok_fetch)
    await disc.discover_free_vision_models(force=True)

    async def boom(_timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr(disc, "_fetch_catalogue", boom)
    assert await disc.discover_free_vision_models(force=True) == ["vendor/good-vl:free"]


async def test_empty_result_keeps_previous_ladder(monkeypatch):
    async def ok_fetch(_timeout):
        return CATALOGUE

    monkeypatch.setattr(disc, "_fetch_catalogue", ok_fetch)
    await disc.discover_free_vision_models(force=True)

    async def empty(_timeout):
        return {"data": []}

    monkeypatch.setattr(disc, "_fetch_catalogue", empty)
    assert await disc.discover_free_vision_models(force=True) == ["vendor/good-vl:free"]


def test_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("VISION_AUTO_DISCOVERY", "0")
    assert disc.get_cached_free_vision_models() == []


def test_ladder_uses_discovered_models_first(monkeypatch):
    from bot.enhanced_retry import EnhancedRetryManager

    monkeypatch.setattr(disc, "_memory_cache", (["vendor/good-vl:free"], 9e18))
    monkeypatch.setenv("VISION_FALLBACK_MODELS", "legacy/dead-vl:free")
    monkeypatch.setenv("VL_MODEL", "legacy/dead-vl:free")

    manager = EnhancedRetryManager()
    models = [pc.model for pc in manager.provider_configs["vision"]]
    # Discovered model leads; stale env model survives only as a tail rung.
    assert models[0] == "vendor/good-vl:free"
    assert "legacy/dead-vl:free" in models


# --- Free-only guarantees [SFT] -------------------------------------------


ZERO_PRICED_NON_FREE = {
    "data": [
        {
            # Zero-priced preview that is NOT a :free variant — starts billing later.
            "id": "vendor/preview-vl",
            "created": 100,
            "context_length": 128000,
            "architecture": {"input_modalities": ["image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ],
}


def test_zero_priced_non_free_variant_is_rejected():
    assert disc._select_models(ZERO_PRICED_NON_FREE, limit=5) == []


def test_free_slug_predicate():
    assert disc.is_free_slug("vendor/model:free")
    assert disc.is_free_slug("Vendor/Model:FREE")
    assert not disc.is_free_slug("vendor/model")
    assert not disc.is_free_slug("vendor/model:free-tier")
    assert not disc.is_free_slug("")


@pytest.mark.parametrize(
    "pricing",
    [
        {"prompt": "0", "completion": "0.000001"},
        {"prompt": "0.0000005", "completion": "0"},
        {"prompt": "0", "completion": "0", "image": "0.0001"},
        {"prompt": "0", "completion": "0", "request": "-1"},  # variable/unknown
        {"prompt": "0", "completion": "0", "web_search": "0.004"},  # future dimension
        {"prompt": "0"},  # completion missing entirely
        {},
    ],
)
def test_non_zero_or_unknown_pricing_is_rejected(pricing):
    assert not disc._is_free(pricing, "vendor/model:free")


def test_all_zero_pricing_on_free_variant_is_accepted():
    assert disc._is_free({"prompt": "0", "completion": "0", "image": "0"}, "vendor/model:free")


async def test_cache_file_cannot_inject_paid_model(tmp_path, monkeypatch):
    """A hand-edited/stale cache with a paid slug must not reach the ladder."""
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps({"models": ["vendor/paid-vl", "vendor/good-vl:free"], "fetched_at": 9e18}))
    disc._reset_for_tests()
    assert disc.get_cached_free_vision_models() == ["vendor/good-vl:free"]


def test_ladder_drops_paid_tail_models(monkeypatch):
    from bot.enhanced_retry import EnhancedRetryManager

    monkeypatch.setattr(disc, "_memory_cache", (["vendor/good-vl:free"], 9e18))
    monkeypatch.setenv("VISION_FALLBACK_MODELS", "openrouter|openai/gpt-4o-mini,openrouter|legacy/dead-vl:free")
    monkeypatch.setenv("VL_MODEL", "")
    monkeypatch.delenv("VISION_ALLOW_PAID_FALLBACK", raising=False)

    models = [pc.model for pc in EnhancedRetryManager().provider_configs["vision"]]
    assert models == ["vendor/good-vl:free", "legacy/dead-vl:free"]
    assert all(m.endswith(":free") for m in models)


def test_paid_tail_kept_when_user_opts_in(monkeypatch):
    from bot.enhanced_retry import EnhancedRetryManager

    monkeypatch.setattr(disc, "_memory_cache", (["vendor/good-vl:free"], 9e18))
    monkeypatch.setenv("VISION_FALLBACK_MODELS", "openrouter|openai/gpt-4o-mini")
    monkeypatch.setenv("VL_MODEL", "")
    monkeypatch.setenv("VISION_ALLOW_PAID_FALLBACK", "1")

    models = [pc.model for pc in EnhancedRetryManager().provider_configs["vision"]]
    assert "openai/gpt-4o-mini" in models


def test_local_ollama_rung_is_never_dropped(monkeypatch):
    from bot.enhanced_retry import ProviderConfig, _enforce_free_only

    monkeypatch.delenv("VISION_ALLOW_PAID_FALLBACK", raising=False)
    ladder = [
        ProviderConfig("openrouter", "vendor/good-vl:free"),
        ProviderConfig("ollama", "llava"),
        ProviderConfig("openrouter", "openai/gpt-4o"),
    ]
    assert [pc.model for pc in _enforce_free_only(ladder)] == ["vendor/good-vl:free", "llava"]


# --- Self-heal on dead-model errors ---------------------------------------


async def test_dead_vision_model_kicks_rediscovery(monkeypatch):
    """A retired VL model triggers an immediate catalogue refresh, debounced."""
    import bot.enhanced_retry as er

    monkeypatch.setattr(er, "_last_rediscovery_kick", 0.0)
    calls: list[bool] = []

    async def fake_refresh(*, force=False):
        calls.append(force)
        return ["vendor/good-vl:free"]

    monkeypatch.setattr(disc, "refresh_and_apply", fake_refresh)

    er._schedule_vision_rediscovery("vision")
    er._schedule_vision_rediscovery("vision")  # debounced
    er._schedule_vision_rediscovery("text")  # wrong modality
    await asyncio.sleep(0)
    await asyncio.gather(*list(er._rediscovery_tasks), return_exceptions=True)

    assert calls == [True]


async def test_rediscovery_kick_noop_when_disabled(monkeypatch):
    import bot.enhanced_retry as er

    monkeypatch.setenv("VISION_AUTO_DISCOVERY", "0")
    monkeypatch.setattr(er, "_last_rediscovery_kick", 0.0)
    called: list[bool] = []

    async def fake_refresh(*, force=False):
        called.append(force)
        return []

    monkeypatch.setattr(disc, "refresh_and_apply", fake_refresh)
    er._schedule_vision_rediscovery("vision")
    await asyncio.sleep(0)
    assert called == []
