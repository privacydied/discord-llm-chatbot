"""Regression test: the free OpenRouter img2img fallback (added alongside
OpenRouterPlugin.capabilities()'s IMAGE_TO_IMAGE mode) was silently never
reachable in the common deployment shape (no VISION_PROVIDER_CONFIG_PATH
override file). UnifiedVisionAdapter.submit()'s `policy.get("provider_order",
[..., "openrouter"])` fallback default is dead code whenever `policy` already
has the key -- which it always does via _default_provider_config()'s own
hardcoded list. That embedded list is the one that actually needs
"openrouter" in it; this test locks that in, plus an end-to-end check that a
quota-exhausted Novita actually falls through to OpenRouter. [REH][CMV]
"""

from unittest.mock import AsyncMock

import pytest

from bot.vision.types import VisionError, VisionErrorType, VisionProvider, VisionRequest, VisionTask
from bot.vision.unified_adapter import UnifiedVisionAdapter, _resolve_openrouter_api_key


@pytest.fixture(autouse=True)
def _clean_openrouter_env(monkeypatch):
    """These vars leaking from the real shell/.env would silently mask the
    bug this file exists to catch. [IV]
    """
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "VISION_API_KEY", "VISION_API_BASE"):
        monkeypatch.delenv(var, raising=False)


def test_default_provider_config_provider_order_includes_openrouter() -> None:
    """The embedded default_policy.provider_order -- the one actually used
    whenever no VISION_PROVIDER_CONFIG_PATH override file is configured --
    must list openrouter, not just the .get() fallback default deeper in
    submit() (which is dead code once this key is present).
    """
    adapter = UnifiedVisionAdapter({"VISION_ALLOWED_PROVIDERS": ["novita", "together", "openrouter"]})
    provider_order = adapter.provider_config["vision"]["default_policy"]["provider_order"]
    assert any(entry.split(":")[0] == "openrouter" for entry in provider_order)


@pytest.mark.asyncio
async def test_openrouter_used_when_novita_quota_exhausted() -> None:
    """Reproduces the real failure: Novita returns a quota/balance error for
    IMAGE_TO_IMAGE and there is no VISION_PROVIDER_CONFIG_PATH override --
    OpenRouter must be tried next instead of the job failing outright.
    """
    config = {
        "VISION_ALLOWED_PROVIDERS": ["novita", "openrouter"],
        "VISION_API_KEY": "test-vision-api-key-0123456789",
        "OPENROUTER_API_KEY": "test-openrouter-api-key-0123456789",
    }
    adapter = UnifiedVisionAdapter(config)
    adapter.providers["novita"].submit = AsyncMock(
        side_effect=VisionError(
            error_type=VisionErrorType.QUOTA_EXCEEDED,
            message="Novita.ai balance/quota exhausted (403)",
            user_message="Out of credit.",
        )
    )
    adapter.providers["openrouter"].submit = AsyncMock(return_value="openrouter-task-id")

    request = VisionRequest(
        task=VisionTask.IMAGE_TO_IMAGE,
        prompt="make him a superhero",
        user_id="1",
        input_image_data=b"fake-image-bytes",
    )

    response = await adapter.submit(request)

    adapter.providers["novita"].submit.assert_awaited_once()
    adapter.providers["openrouter"].submit.assert_awaited_once()
    assert response.success is True
    assert response.provider == VisionProvider.OPENROUTER


# ---------------------------------------------------------------------------
# _resolve_openrouter_api_key -- the second bug found live: OpenRouter's
# provider entry resolved its key from VISION_API_KEY (a Together/Novita
# key), so even once reachable it failed with "authentication failed". A
# deployment's real OpenRouter key commonly lives under OPENAI_API_KEY
# (the text-backend setup), not a dedicated OPENROUTER_API_KEY.
# ---------------------------------------------------------------------------


class TestResolveOpenrouterApiKey:
    def test_prefers_dedicated_var(self) -> None:
        config = {"OPENROUTER_API_KEY": "or-dedicated-key", "OPENAI_API_KEY": "or-openai-key", "OPENAI_API_BASE": "https://openrouter.ai/api/v1"}
        assert _resolve_openrouter_api_key(config) == "or-dedicated-key"

    def test_falls_back_to_openai_key_when_its_base_is_openrouter(self) -> None:
        """The real deployment shape: OpenRouter doubles as the text backend,
        so OPENAI_API_KEY/OPENAI_API_BASE are the only place the key lives.
        """
        config = {
            "OPENAI_API_KEY": "sk-or-v1-the-real-openrouter-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
            "VISION_API_KEY": "a-together-or-novita-key",
        }
        assert _resolve_openrouter_api_key(config) == "sk-or-v1-the-real-openrouter-key"

    def test_ignores_vision_api_key_when_its_base_is_not_openrouter(self) -> None:
        """VISION_API_KEY alone must never be treated as an OpenRouter key --
        it's the Together/Novita-flavored key by convention in this codebase.
        """
        config = {"VISION_API_KEY": "a-together-or-novita-key"}
        assert _resolve_openrouter_api_key(config) == ""

    def test_uses_vision_api_key_only_when_its_base_is_openrouter(self) -> None:
        config = {"VISION_API_KEY": "a-real-openrouter-key", "VISION_API_BASE": "https://openrouter.ai/api/v1"}
        assert _resolve_openrouter_api_key(config) == "a-real-openrouter-key"

    def test_returns_empty_when_nothing_resolves(self) -> None:
        assert _resolve_openrouter_api_key({}) == ""


@pytest.mark.asyncio
async def test_openrouter_plugin_gets_openai_key_not_vision_api_key() -> None:
    """End-to-end repro of the real incident: OPENAI_API_KEY/OPENAI_API_BASE
    hold the real OpenRouter credentials (text-backend setup), VISION_API_KEY
    holds an unrelated Together/Novita key, and no OPENROUTER_API_KEY is set
    at all. The openrouter plugin must end up with the OPENAI_API_KEY value,
    not VISION_API_KEY -- previously it always got VISION_API_KEY.
    """
    config = {
        "VISION_ALLOWED_PROVIDERS": ["openrouter"],
        "VISION_API_KEY": "a-together-or-novita-key",
        "OPENAI_API_KEY": "sk-or-v1-the-real-openrouter-key",
        "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
    }
    adapter = UnifiedVisionAdapter(config)

    assert adapter.providers["openrouter"].api_key == "sk-or-v1-the-real-openrouter-key"


@pytest.mark.asyncio
async def test_hot_reload_does_not_regress_openrouter_key() -> None:
    """update_config() (the .env hot-reload path) used to re-derive every
    provider's api_key with the same buggy VISION_API_KEY fallback right
    after _initialize_providers() resolved it correctly, silently undoing
    the fix on the next config reload.
    """
    config = {
        "VISION_ALLOWED_PROVIDERS": ["openrouter"],
        "VISION_API_KEY": "a-together-or-novita-key",
        "OPENAI_API_KEY": "sk-or-v1-the-real-openrouter-key",
        "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
    }
    adapter = UnifiedVisionAdapter(config)
    assert adapter.providers["openrouter"].api_key == "sk-or-v1-the-real-openrouter-key"

    adapter.update_config(dict(config))  # simulate a hot-reload with unchanged env

    assert adapter.providers["openrouter"].api_key == "sk-or-v1-the-real-openrouter-key"
