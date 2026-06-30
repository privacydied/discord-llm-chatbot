"""Tests for STTManager's whisper model cache eviction (memory audit fix).

Construction bypasses __init__ (`STTManager.__new__`) so these stay fast,
deterministic unit tests -- the real __init__ starts a background thread
that imports faster_whisper and loads actual model weights, which is both
slow and exactly what these tests must avoid touching.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from unittest.mock import MagicMock

import bot.stt as stt_module
from bot.stt import ModelSpec, STTManager


def _bare_manager(default_spec: ModelSpec) -> STTManager:
    """Build an STTManager with no background warm-load thread started."""
    manager = STTManager.__new__(STTManager)
    manager.engine = "faster-whisper"
    manager._model_cache = OrderedDict()
    manager._model_locks = {}
    manager._ready_event = threading.Event()
    manager._default_spec = default_spec
    manager._available = False
    manager._init_thread = None
    return manager


class TestEvictLruLocked:
    def test_cap_evicts_oldest_non_default_entry(self, monkeypatch) -> None:
        monkeypatch.setattr(stt_module, "_MODEL_CACHE_MAX", 2)
        default = ModelSpec("base", "int8")
        manager = _bare_manager(default)
        manager._model_cache[default] = MagicMock(name="base_model")
        manager._model_cache[ModelSpec("small", "int8")] = MagicMock(name="small_model")
        manager._model_cache[ModelSpec("tiny", "int8")] = MagicMock(name="tiny_model")

        manager._evict_lru_locked()

        assert len(manager._model_cache) == 2
        assert default in manager._model_cache
        # "small" was inserted before "tiny" -> it's the LRU victim.
        assert ModelSpec("small", "int8") not in manager._model_cache
        assert ModelSpec("tiny", "int8") in manager._model_cache

    def test_default_spec_never_evicted_even_alone(self, monkeypatch) -> None:
        monkeypatch.setattr(stt_module, "_MODEL_CACHE_MAX", 0)
        default = ModelSpec("base", "int8")
        manager = _bare_manager(default)
        manager._model_cache[default] = MagicMock(name="base_model")

        manager._evict_lru_locked()  # would loop forever if default were a valid victim

        assert default in manager._model_cache

    def test_under_cap_evicts_nothing(self, monkeypatch) -> None:
        monkeypatch.setattr(stt_module, "_MODEL_CACHE_MAX", 5)
        default = ModelSpec("base", "int8")
        manager = _bare_manager(default)
        manager._model_cache[default] = MagicMock(name="base_model")
        manager._model_cache[ModelSpec("tiny", "int8")] = MagicMock(name="tiny_model")

        manager._evict_lru_locked()

        assert len(manager._model_cache) == 2


class TestLoadModelCacheBehavior:
    def test_cache_hit_moves_entry_to_most_recently_used(self) -> None:
        default = ModelSpec("base", "int8")
        manager = _bare_manager(default)
        other = ModelSpec("tiny", "int8")
        manager._model_cache[default] = MagicMock(name="base_model")
        manager._model_cache[other] = MagicMock(name="tiny_model")

        result = manager._load_model(default)

        assert result is manager._model_cache[default]
        # Accessing `default` (originally first/oldest) must move it to the end.
        assert next(reversed(manager._model_cache)) == default


class TestEvictIdleModels:
    def test_evicts_all_non_default_entries_and_returns_count(self) -> None:
        default = ModelSpec("base", "int8")
        manager = _bare_manager(default)
        manager._model_cache[default] = MagicMock(name="base_model")
        manager._model_cache[ModelSpec("small", "int8")] = MagicMock(name="small_model")
        manager._model_cache[ModelSpec("tiny", "int8")] = MagicMock(name="tiny_model")

        evicted = manager.evict_idle_models()

        assert evicted == 2
        assert list(manager._model_cache) == [default]

    def test_noop_when_only_default_cached(self) -> None:
        default = ModelSpec("base", "int8")
        manager = _bare_manager(default)
        manager._model_cache[default] = MagicMock(name="base_model")

        evicted = manager.evict_idle_models()

        assert evicted == 0
        assert default in manager._model_cache

    def test_thread_safe_acquires_per_spec_lock(self) -> None:
        """evict_idle_models must take each victim's own lock before popping
        it -- callers outside this module (e.g. the health check) rely on
        this, unlike the internal _evict_lru_locked which assumes the lock
        is already held."""
        default = ModelSpec("base", "int8")
        manager = _bare_manager(default)
        victim_spec = ModelSpec("tiny", "int8")
        manager._model_cache[default] = MagicMock(name="base_model")
        manager._model_cache[victim_spec] = MagicMock(name="tiny_model")
        lock = manager._get_lock_for(victim_spec)
        assert not lock.locked()

        manager.evict_idle_models()

        # Lock must be released after eviction (not left held).
        assert not lock.locked()


def test_get_stt_manager_if_initialized_does_not_create_one(monkeypatch) -> None:
    """The side-effect-free accessor must never instantiate STTManager --
    doing so would cold-start a background model-load thread just because
    something (e.g. a memory health check) asked whether STT is in use."""
    monkeypatch.setattr(stt_module, "_stt_manager", None)

    result = stt_module.get_stt_manager_if_initialized()

    assert result is None


def test_get_stt_manager_if_initialized_returns_existing_manager(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(stt_module, "_stt_manager", sentinel)

    result = stt_module.get_stt_manager_if_initialized()

    assert result is sentinel
