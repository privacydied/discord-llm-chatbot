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
from bot.vision.unified_adapter import UnifiedVisionAdapter


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
