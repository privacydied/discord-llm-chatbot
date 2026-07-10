"""NvidiaPlugin's IMAGE_TO_IMAGE support (FLUX.1 Kontext [dev]) was tried and
reverted: a live request to NVIDIA's hosted GenAI endpoint
(ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev) came back
422 `{"detail":"Expected: example_id, got: base64"}` - this hosted tier only
accepts one of NVIDIA's 3 canned demo images, not an arbitrary user-uploaded
photo. NVIDIA does ship a real self-hostable NIM microservice for Kontext
editing, but that's a different deployment (your own GPU/endpoint), not this
public API. These tests guard against silently re-introducing IMAGE_TO_IMAGE
here without first confirming a genai_base_url that actually accepts
uploaded images. [CMV][REH]
"""

import pytest

from bot.vision.types import VisionErrorType, VisionTask
from bot.vision.unified_adapter import NormalizedRequest, NvidiaPlugin, VisionError


def test_nvidia_only_advertises_text_to_image() -> None:
    plugin = NvidiaPlugin("nvidia", {}, "test-nvidia-key-0123456789")
    modes = plugin.capabilities()["modes"]
    assert modes == [VisionTask.TEXT_TO_IMAGE]
    assert VisionTask.IMAGE_TO_IMAGE not in modes


def test_nvidia_model_map_has_no_image_to_image_entry() -> None:
    plugin = NvidiaPlugin("nvidia", {}, "test-nvidia-key-0123456789")
    assert VisionTask.IMAGE_TO_IMAGE not in plugin.model_map
    assert plugin.model_map[VisionTask.TEXT_TO_IMAGE] == "black-forest-labs/flux.1-dev"


@pytest.mark.asyncio
async def test_image_to_image_request_rejected_before_any_network_call() -> None:
    """submit() must reject IMAGE_TO_IMAGE immediately (UNSUPPORTED_TASK) -
    never attempt an HTTP call the provider is known to reject."""
    plugin = NvidiaPlugin("nvidia", {}, "test-nvidia-key-0123456789")
    request = NormalizedRequest(
        task=VisionTask.IMAGE_TO_IMAGE,
        prompt="give this man a beard",
        input_image_data=b"fake-source-image-bytes",
    )

    try:
        with pytest.raises(VisionError) as excinfo:
            await plugin.submit(request)
        assert excinfo.value.error_type == VisionErrorType.UNSUPPORTED_TASK
    finally:
        await plugin.shutdown()
