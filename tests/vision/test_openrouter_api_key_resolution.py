"""Regression tests for OpenRouter provider key resolution, plus a guard
against re-introducing a free IMAGE_TO_IMAGE default with no real model
behind it.

History: a prior commit added OpenRouter as a "free img2img fallback" using
qwen/qwen2.5-vl-72b-instruct:free. That was reverted after two live
failures: (1) the provider was unreachable in the default deployment shape
because two separate hardcoded provider_order lists existed and only one got
"openrouter" added, and (2) once reachable, it authenticated with
VISION_API_KEY (a Together/Novita key) instead of a real OpenRouter key.
Fixing (2) surfaced a third, unfixable problem: verified against OpenRouter's
live catalogue (2026-08-27) that qwen2.5-vl-72b-instruct has
output_modalities=["text"] only -- an understanding model, not a
generation/edit one -- and that every model OpenRouter currently lists with
"image" in output_modalities is paid. There is no free model to default to,
so IMAGE_TO_IMAGE support was reverted entirely rather than left half-broken.

The key-resolution fix itself (_resolve_openrouter_api_key) remains correct
and generically useful regardless of which tasks OpenRouter ends up serving,
so those tests stay. [REH][SFT][CMV]
"""

from unittest.mock import AsyncMock

import pytest

from bot.vision.types import VisionRequest, VisionTask
from bot.vision.unified_adapter import UnifiedVisionAdapter, _resolve_openrouter_api_key


@pytest.fixture(autouse=True)
def _clean_openrouter_env(monkeypatch):
    """These vars leaking from the real shell/.env would silently mask the
    bug this file exists to catch. [IV]
    """
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "VISION_API_KEY", "VISION_API_BASE"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Guard: no free-model IMAGE_TO_IMAGE default. See module docstring.
# ---------------------------------------------------------------------------


def test_openrouter_does_not_claim_image_to_image_capability() -> None:
    """There is currently no free OpenRouter model that can generate/edit
    images (verified against the live catalogue). Advertising IMAGE_TO_IMAGE
    support with no real model behind it just moves the "always fails" bug
    around instead of fixing it -- don't re-add this without a specific,
    deliberately chosen model in OpenRouterPlugin.model_map alongside it.
    """
    adapter = UnifiedVisionAdapter({"VISION_ALLOWED_PROVIDERS": ["openrouter"], "OPENROUTER_API_KEY": "test-key-0123456789"})
    modes = adapter.providers["openrouter"].capabilities()["modes"]
    assert VisionTask.IMAGE_TO_IMAGE not in modes


@pytest.mark.asyncio
async def test_openrouter_rejects_image_to_image_with_no_configured_model() -> None:
    adapter = UnifiedVisionAdapter({"VISION_ALLOWED_PROVIDERS": ["openrouter"], "OPENROUTER_API_KEY": "test-key-0123456789"})
    adapter.providers["openrouter"].startup = AsyncMock()
    from bot.vision.unified_adapter import NormalizedRequest

    request = NormalizedRequest(task=VisionTask.IMAGE_TO_IMAGE, prompt="edit it", input_image_data=b"fake")
    with pytest.raises(Exception, match="not supported"):
        await adapter.providers["openrouter"].submit(request)


def test_default_provider_order_omits_openrouter() -> None:
    """The embedded default_policy.provider_order (the one actually used
    whenever no VISION_PROVIDER_CONFIG_PATH override file is configured)
    must not list openrouter -- it can't serve the tasks this ladder is for
    without a real model configured (see module docstring).
    """
    adapter = UnifiedVisionAdapter({"VISION_ALLOWED_PROVIDERS": ["novita", "together"]})
    provider_order = adapter.provider_config["vision"]["default_policy"]["provider_order"]
    assert not any(entry.split(":")[0] == "openrouter" for entry in provider_order)


@pytest.mark.asyncio
async def test_image_to_image_job_fails_honestly_with_no_openrouter_in_the_mix() -> None:
    """With OpenRouter no longer claiming IMAGE_TO_IMAGE, a job with no
    capable provider must fail with the existing honest "no provider"
    message, not a confusing downstream auth/model error.
    """
    from bot.vision.types import VisionError

    adapter = UnifiedVisionAdapter({"VISION_ALLOWED_PROVIDERS": ["nvidia"], "NVIDIA_NIM_API_KEY": "test-key-0123456789"})
    request = VisionRequest(
        task=VisionTask.IMAGE_TO_IMAGE,
        prompt="edit it",
        user_id="1",
        input_image_data=b"fake-image-bytes",
    )
    with pytest.raises(VisionError):
        await adapter.submit(request)


# ---------------------------------------------------------------------------
# _resolve_openrouter_api_key -- OpenRouter's provider entry used to resolve
# its key from VISION_API_KEY (a Together/Novita key), so even when
# reachable it failed with "authentication failed". A deployment's real
# OpenRouter key commonly lives under OPENAI_API_KEY (the text-backend
# setup), not a dedicated OPENROUTER_API_KEY. Still relevant regardless of
# IMAGE_TO_IMAGE: OpenRouter remains a configured TEXT_TO_IMAGE provider.
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
