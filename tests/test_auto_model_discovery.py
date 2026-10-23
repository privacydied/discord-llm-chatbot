"""Largest-free-model auto selection: ranking, tool capability, TOOLS_MODEL=auto. [REH]"""

from __future__ import annotations

import json

import pytest

import bot.vision.free_text_discovery as disc
from bot.tools import inference
from bot.vision.model_ranking import size_sort_key

CATALOGUE = [  # live OpenRouter free text models, 2026-10-24
    "recursive/flint-v1.0:free",
    "xingchen-agi/xing4.0-29b-a4b:free",
    "thinkingmachines/inkling-small:free",
    "nvidia/nemotron-3.5-lightning:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "liquid/lfm-2.5-2.6b:free",
]


def test_largest_first_then_unknown_then_small():
    ranked = sorted(CATALOGUE, key=size_sort_key)
    assert ranked[:4] == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
        "xingchen-agi/xing4.0-29b-a4b:free",
    ]
    assert set(ranked[4:6]) == {"recursive/flint-v1.0:free", "nvidia/nemotron-3.5-lightning:free"}
    assert set(ranked[6:]) == {"thinkingmachines/inkling-small:free", "cohere/north-mini-code:free", "liquid/lfm-2.5-2.6b:free"}


def _entry(model_id, tools=True):
    return {
        "id": model_id,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "pricing": {"prompt": "0", "completion": "0"},
        "context_length": 131072,
        "supported_parameters": ["tools", "max_tokens"] if tools else ["max_tokens"],
    }


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXT_DISCOVERY_CACHE_PATH", str(tmp_path / "free_text.json"))
    monkeypatch.setenv("TEXT_AUTO_DISCOVERY", "1")
    monkeypatch.delenv("TEXT_DISCOVERY_EXCLUDE", raising=False)
    monkeypatch.setattr(disc, "_memory_cache", None)
    monkeypatch.setattr(disc, "_memory_tools", None)
    return tmp_path / "free_text.json"


async def test_select_rank_and_tool_capability_persisted(cache, monkeypatch):
    payload = {"data": [_entry("google/gemma-4-31b-it:free"), _entry("nvidia/nemotron-3-ultra-550b-a55b:free", tools=False), _entry("nvidia/nemotron-3-super-120b-a12b:free")]}
    models = disc._select_models(payload, 10)
    assert models[0] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    await disc._commit(models, disc._tool_capable_ids(payload))
    assert json.loads(cache.read_text())["tools"] == ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free"]
    monkeypatch.setattr(disc, "_memory_tools", None)  # reload from disk
    assert disc.get_cached_free_tool_models()[0] == "nvidia/nemotron-3-super-120b-a12b:free"  # ultra lacks tools
    assert disc.resolve_auto_model("auto") == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert disc.resolve_auto_model("auto", tools=True) == "nvidia/nemotron-3-super-120b-a12b:free"
    assert disc.resolve_auto_model("some/pinned:free") == "some/pinned:free"


def test_excluded_and_old_cache_without_tools_field(cache, monkeypatch):
    cache.write_text(json.dumps({"models": ["nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "google/gemma-4-31b-it:free"], "fetched_at": 1}))
    monkeypatch.setenv("TEXT_DISCOVERY_EXCLUDE", "nano-omni")
    assert disc.get_cached_free_tool_models() == ["google/gemma-4-31b-it:free"]


def test_tool_loop_auto_mode_follows_discovery(monkeypatch):
    monkeypatch.setattr(inference, "_discovered_tool_models", lambda: ["big:free", "mid:free"])
    cfg = {"TOOLS_MODEL": "auto", "OPENAI_API_BASE": "https://openrouter.ai/api/v1", "TOOLS_FALLBACK_MODELS": "pinned:free"}
    assert inference._model_candidates(cfg, "big:free") == ["big:free", "mid:free", "pinned:free"]
    pinned = {**cfg, "TOOLS_MODEL": "pinned:free"}
    assert inference._model_candidates(pinned, "pinned:free") == ["pinned:free"]  # explicit model: no discovery
