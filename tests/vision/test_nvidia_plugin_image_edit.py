"""NvidiaPlugin previously supported TEXT_TO_IMAGE only (flux.1-dev); this adds
FLUX.1 Kontext [dev] for IMAGE_TO_IMAGE (NVIDIA-hosted image editing), reusing
the same {genai_base_url}/genai/{model} endpoint pattern. No live NVIDIA
account was used to verify the exact request/response shape - these tests
mock the HTTP layer against the shape documented at
docs.nvidia.com/nim/visual-genai. [CMV][REH]
"""

import base64
import json
from contextlib import asynccontextmanager

import pytest

from bot.vision.types import VisionErrorType, VisionTask
from bot.vision.unified_adapter import NormalizedRequest, NvidiaPlugin, VisionError


class _FakeResponse:
    def __init__(self, status: int, body: dict | str) -> None:
        self.status = status
        self._body = body if isinstance(body, str) else json.dumps(body)

    async def text(self) -> str:
        return self._body

    async def json(self) -> dict:
        return json.loads(self._body)


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_url = None
        self.last_json = None

    @asynccontextmanager
    async def _cm(self):
        yield self._response

    def post(self, url, json=None, headers=None):  # noqa: A002 - matches aiohttp signature
        self.last_url = url
        self.last_json = json
        return self._cm()


def _plugin(response: _FakeResponse) -> tuple[NvidiaPlugin, _FakeSession]:
    plugin = NvidiaPlugin("nvidia", {}, "test-nvidia-key-0123456789")
    session = _FakeSession(response)
    plugin.session = session
    return plugin, session


def _edit_request(**overrides) -> NormalizedRequest:
    defaults = dict(
        task=VisionTask.IMAGE_TO_IMAGE,
        prompt="give this man a beard",
        negative_prompt="",
        width=1024,
        height=1024,
        steps=30,
        guidance_scale=7.0,
        seed=None,
        input_image_data=b"fake-source-image-bytes",
        input_image_url=None,
        batch_size=1,
        safety_mode="strict",
        preferred_model=None,
    )
    defaults.update(overrides)
    return NormalizedRequest(**defaults)


class TestNvidiaPluginCapabilities:
    def test_capabilities_now_include_image_to_image(self) -> None:
        plugin = NvidiaPlugin("nvidia", {}, "test-nvidia-key-0123456789")
        modes = plugin.capabilities()["modes"]
        assert VisionTask.TEXT_TO_IMAGE in modes
        assert VisionTask.IMAGE_TO_IMAGE in modes


@pytest.mark.asyncio
class TestNvidiaPluginImageEdit:
    async def test_submit_sends_base64_image_to_kontext_endpoint(self) -> None:
        plugin, session = _plugin(_FakeResponse(200, {"artifacts": [{"base64": "ZmFrZQ=="}]}))

        job_id = await plugin.submit(_edit_request())

        assert job_id
        assert session.last_url == "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev"
        assert session.last_json["prompt"] == "give this man a beard"
        expected_b64 = base64.b64encode(b"fake-source-image-bytes").decode()
        assert session.last_json["image"] == f"data:image/png;base64,{expected_b64}"
        assert session.last_json["cfg_scale"] == 7.0
        assert session.last_json["steps"] == 30

    async def test_submit_stores_completed_result_for_polling(self) -> None:
        plugin, _ = _plugin(_FakeResponse(200, {"artifacts": [{"base64": "ZmFrZQ=="}]}))

        job_id = await plugin.submit(_edit_request())
        status = await plugin.poll(job_id)
        result = await plugin.fetch_result(job_id)

        assert status.status.value == "completed"
        assert result.assets == ["data:image/jpeg;base64,ZmFrZQ=="]

    async def test_missing_input_image_data_is_a_validation_error(self) -> None:
        plugin, _ = _plugin(_FakeResponse(200, {"artifacts": []}))

        with pytest.raises(VisionError) as excinfo:
            await plugin.submit(_edit_request(input_image_data=None))

        assert excinfo.value.error_type == VisionErrorType.VALIDATION_ERROR

    async def test_cfg_scale_and_steps_are_clamped_to_documented_ranges(self) -> None:
        plugin, session = _plugin(_FakeResponse(200, {"artifacts": [{"base64": "ZmFrZQ=="}]}))

        await plugin.submit(_edit_request(guidance_scale=99.0, steps=5))

        assert session.last_json["cfg_scale"] == 9.0  # clamped to documented max
        assert session.last_json["steps"] == 20  # clamped to documented min

    async def test_422_maps_to_validation_error_with_raw_body_surfaced(self) -> None:
        plugin, _ = _plugin(_FakeResponse(422, "unprocessable: bad aspect_ratio"))

        with pytest.raises(VisionError) as excinfo:
            await plugin.submit(_edit_request())

        assert excinfo.value.error_type == VisionErrorType.VALIDATION_ERROR
        assert "unprocessable" in excinfo.value.message

    async def test_text_to_image_still_works_unchanged(self) -> None:
        plugin, session = _plugin(_FakeResponse(200, {"artifacts": [{"base64": "ZmFrZQ=="}]}))
        request = _edit_request(task=VisionTask.TEXT_TO_IMAGE, input_image_data=None)

        await plugin.submit(request)

        assert session.last_url == "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"
        assert "image" not in session.last_json
        assert "cfg_scale" not in session.last_json
