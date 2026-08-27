"""Regression test: discovered free text models must lead the retry ladder
even when TEXT_FALLBACK_MODELS is set in the environment -- mirroring the
vision ladder's unconditional merge. Previously text discovery only applied
when no env ladder existed at all, which silently disabled self-healing for
any operator who had TEXT_FALLBACK_MODELS configured (the common case). [REH]
"""

from bot.vision import free_text_discovery as tdisc


def _reset(monkeypatch):
    monkeypatch.delenv("VISION_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("VL_MODEL", raising=False)
    tdisc._reset_for_tests()


def test_discovered_text_models_lead_even_with_env_ladder_set(monkeypatch):
    from bot.enhanced_retry import EnhancedRetryManager

    _reset(monkeypatch)
    monkeypatch.setattr(tdisc, "_memory_cache", (["vendor/good-text:free"], 9e18))
    monkeypatch.setenv("TEXT_AUTO_DISCOVERY", "1")
    monkeypatch.setenv("TEXT_FALLBACK_MODELS", "openrouter|legacy/stale-text:free")

    manager = EnhancedRetryManager()
    models = [pc.model for pc in manager.provider_configs["text"]]

    assert models[0] == "vendor/good-text:free"
    assert "legacy/stale-text:free" in models


def test_text_ladder_falls_back_to_env_when_discovery_empty(monkeypatch):
    from bot.enhanced_retry import EnhancedRetryManager

    _reset(monkeypatch)
    monkeypatch.setattr(tdisc, "_memory_cache", ([], 9e18))
    monkeypatch.setenv("TEXT_AUTO_DISCOVERY", "1")
    monkeypatch.setenv("TEXT_FALLBACK_MODELS", "openrouter|legacy/stale-text:free")

    manager = EnhancedRetryManager()
    models = [pc.model for pc in manager.provider_configs["text"]]

    assert models == ["legacy/stale-text:free"]
