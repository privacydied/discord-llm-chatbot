"""Tests for OpenRouter free vision-model auto-discovery. [REH]"""

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
    monkeypatch.setenv("VISION_DISCOVERY_CACHE_PATH", str(tmp_path / "models.json"))
    monkeypatch.setenv("VISION_AUTO_DISCOVERY", "1")
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
