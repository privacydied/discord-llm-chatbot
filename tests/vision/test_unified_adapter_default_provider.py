"""Regression test: VISION_DEFAULT_PROVIDER used to be silently ignored for
every value except "openrouter" - UnifiedVisionAdapter.submit() only
special-cased `if self.default_provider == "openrouter"` when promoting a
provider to the front of provider_order. Generalized to promote whichever
provider is configured; providers that can't handle the requested task are
still safely skipped via the existing capabilities() check. [REH][CMV]
"""

from unittest.mock import AsyncMock

import pytest

from bot.vision.types import VisionProvider, VisionRequest, VisionTask
from bot.vision.unified_adapter import UnifiedVisionAdapter


@pytest.fixture
def adapter():
    config = {
        "VISION_ALLOWED_PROVIDERS": ["nvidia", "novita", "together"],
        "VISION_DEFAULT_PROVIDER": "nvidia",
        "VISION_API_KEY": "test-vision-api-key-0123456789",
        "NVIDIA_NIM_API_KEY": "test-nvidia-api-key-0123456789",
    }
    a = UnifiedVisionAdapter(config)
    # Stub every provider's submit() so no real HTTP calls happen; each
    # returns a distinct provider-tagged task id so we can see which one won.
    for name, plugin in a.providers.items():
        plugin.submit = AsyncMock(return_value=f"{name}-task-id")
    return a


@pytest.mark.asyncio
async def test_default_provider_promoted_for_supported_task(adapter) -> None:
    """nvidia supports TEXT_TO_IMAGE, so promoting it to the front means it's
    the first (and here, only) provider actually attempted."""
    request = VisionRequest(
        task=VisionTask.TEXT_TO_IMAGE,
        prompt="a red bicycle",
        user_id="1",
    )

    response = await adapter.submit(request)

    assert response.provider == VisionProvider.NVIDIA
    adapter.providers["nvidia"].submit.assert_awaited_once()
    adapter.providers["novita"].submit.assert_not_awaited()
    adapter.providers["together"].submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_provider_skipped_when_it_cant_do_the_task(adapter) -> None:
    """nvidia has no IMAGE_TO_IMAGE capability - promoting it to the front
    must NOT break editing; it should be silently skipped and the next
    capable provider (novita) used instead."""
    request = VisionRequest(
        task=VisionTask.IMAGE_TO_IMAGE,
        prompt="give this man a beard",
        user_id="1",
        input_image_data=b"fake-image-bytes",
    )

    response = await adapter.submit(request)

    assert response.provider != VisionProvider.NVIDIA
    adapter.providers["nvidia"].submit.assert_not_awaited()


def test_default_provider_empty_string_is_a_no_op() -> None:
    """No VISION_DEFAULT_PROVIDER configured -> provider_order construction
    must not crash and must not promote an empty string into the list."""
    config = {
        "VISION_ALLOWED_PROVIDERS": ["novita", "together"],
        "VISION_API_KEY": "test-vision-api-key-0123456789",
    }
    a = UnifiedVisionAdapter(config)
    assert a.default_provider == ""
