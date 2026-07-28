"""Budget knobs in .env must actually reach the code that reads them. [REH][IV]

`_build_config` returns an explicit allowlist dict, so a key that is never
registered there can never be read back: `config.get("TEXT_PER_ITEM_BUDGET",
120.0)` always returned the literal 120.0 and the user's `TEXT_PER_ITEM_BUDGET=240.0`
was silently dead. Same for MULTIMODAL_PER_ITEM_BUDGET (call-site default 30.0),
MULTIMODAL_TOTAL_BUDGET_S (240.0, arms the ambient deadline) and
MEDIA_PROVIDER_TIMEOUT. Only VISION_PER_ITEM_BUDGET had been wired up, which is
why vision.budget logged 300s while text.budget logged 120s from the same .env.
"""

from __future__ import annotations

import pytest

from bot.config._base import _build_config

# key -> (env value, expected parsed, call-site default when unset)
BUDGET_KEYS = {
    "TEXT_PER_ITEM_BUDGET": ("240.0", 240.0, 120.0),
    "MULTIMODAL_PER_ITEM_BUDGET": ("300.0", 300.0, 30.0),
    "MULTIMODAL_TOTAL_BUDGET_S": ("360.0", 360.0, 240.0),
    "VISION_PER_ITEM_BUDGET": ("300.0", 300.0, 120.0),
}


def _getter(values: dict[str, str]):
    def env_getter(key: str, default: str | None = None) -> str | None:
        return values.get(key, default)

    return env_getter


@pytest.mark.parametrize(("key", "raw", "expected", "unset_default"), [(k, *v) for k, v in BUDGET_KEYS.items()])
def test_budget_key_is_read_from_env(key: str, raw: str, expected: float, unset_default: float) -> None:
    cfg = _build_config(_getter({key: raw}))
    assert cfg.get(key, unset_default) == expected


@pytest.mark.parametrize(("key", "unset_default"), [(k, v[2]) for k, v in BUDGET_KEYS.items()])
def test_unset_budget_key_keeps_call_site_default(key: str, unset_default: float) -> None:
    """Registering the key must not change behaviour for users who never set it."""
    cfg = _build_config(_getter({}))
    assert cfg.get(key, unset_default) == unset_default


def test_inline_comment_is_stripped() -> None:
    cfg = _build_config(_getter({"TEXT_PER_ITEM_BUDGET": "240.0  # was 120"}))
    assert cfg["TEXT_PER_ITEM_BUDGET"] == 240.0


def test_malformed_budget_falls_back_to_default() -> None:
    cfg = _build_config(_getter({"TEXT_PER_ITEM_BUDGET": "not-a-number"}))
    assert cfg["TEXT_PER_ITEM_BUDGET"] == 120.0


class TestPresenceSensitiveMediaBudgets:
    """enhanced_retry branches on `is not None`, so these must stay absent when unset."""

    def test_absent_when_unset(self) -> None:
        cfg = _build_config(_getter({}))
        assert "MEDIA_PROVIDER_TIMEOUT" not in cfg
        assert "MEDIA_PER_ITEM_BUDGET" not in cfg

    def test_present_when_set(self) -> None:
        cfg = _build_config(_getter({"MEDIA_PROVIDER_TIMEOUT": "240", "MEDIA_PER_ITEM_BUDGET": "180"}))
        assert cfg["MEDIA_PROVIDER_TIMEOUT"] == 240.0
        assert cfg["MEDIA_PER_ITEM_BUDGET"] == 180.0

    def test_never_stores_none(self) -> None:
        """A stored None would crash float(config.get(key, "120.0")) at the call site."""
        cfg = _build_config(_getter({"MEDIA_PER_ITEM_BUDGET": ""}))
        assert cfg.get("MEDIA_PER_ITEM_BUDGET") is not None or "MEDIA_PER_ITEM_BUDGET" not in cfg

    def test_malformed_is_ignored_not_stored(self) -> None:
        cfg = _build_config(_getter({"MEDIA_PROVIDER_TIMEOUT": "soon"}))
        assert "MEDIA_PROVIDER_TIMEOUT" not in cfg


def test_media_timeout_derivation_unchanged_when_unset(monkeypatch) -> None:
    """The 100s default path in enhanced_retry must survive this change."""
    from bot.enhanced_retry import EnhancedRetryManager

    monkeypatch.delenv("MEDIA_PROVIDER_TIMEOUT", raising=False)
    monkeypatch.delenv("MEDIA_PER_ITEM_BUDGET", raising=False)
    monkeypatch.setattr("bot.enhanced_retry.load_config", lambda: _build_config(_getter({})), raising=False)

    mgr = EnhancedRetryManager()
    media = mgr.provider_configs.get("media") or []
    assert media and media[0].timeout == 100.0
