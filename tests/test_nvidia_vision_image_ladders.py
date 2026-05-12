import pytest

from bot.enhanced_retry import ProviderConfig, get_retry_manager
from bot.openai_backend import generate_vl_response
from bot.vision.types import (
    VisionError,
    VisionErrorType,
    VisionProvider,
    VisionRequest,
    VisionTask,
)
from bot.vision.unified_adapter import UnifiedVisionAdapter


class FakeChatCompletions:
    def __init__(self, create_fn):
        self._create = create_fn

    async def create(self, **kwargs):
        return await self._create(**kwargs)


class FakeChat:
    def __init__(self, create_fn):
        self.completions = FakeChatCompletions(create_fn)


class FakeOpenAIClient:
    def __init__(self, create_fn):
        self.chat = FakeChat(create_fn)


@pytest.mark.asyncio
async def test_vl_ladder_can_mix_openrouter_and_nvidia_endpoints(monkeypatch, tmp_path):
    prompt_file = tmp_path / "vl_prompt.txt"
    prompt_file.write_text("You describe images.", encoding="utf-8")

    def fake_load_config():
        return {
            "OPENAI_API_KEY": "openrouter-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
            "NVIDIA_NIM_API_KEY": "nvidia-test-key",
            "NVIDIA_NIM_API_BASE": "https://integrate.api.nvidia.com/v1",
            "VL_PROMPT_FILE": str(prompt_file),
            "MAX_RESPONSE_TOKENS": 1000,
            "VISION_PER_ITEM_BUDGET": 10.0,
            "TEMPERATURE": 0.1,
        }

    monkeypatch.setattr("bot.openai_backend.load_config", fake_load_config)

    async def fake_get_base64_image(_path):
        return "data:image/png;base64,AAAA"

    monkeypatch.setattr("bot.openai_backend.get_base64_image", fake_get_base64_image)

    mgr = get_retry_manager()
    mgr.circuit_breakers.clear()
    mgr.provider_configs["vision"] = [
        ProviderConfig("openrouter", "openrouter-vl-a", timeout=2.0, max_attempts=1),
        ProviderConfig("nvidia", "nvidia-vl-b", timeout=2.0, max_attempts=1),
    ]

    client_calls = []

    async def fake_create(**kwargs):
        model = kwargs["model"]
        if model == "openrouter-vl-a":
            raise Exception("429 Too Many Requests")

        class _Usage:
            prompt_tokens = 1
            completion_tokens = 2
            total_tokens = 3

        class _Msg:
            content = "OK NVIDIA VISION"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()

    def fake_client_ctor(**kwargs):
        client_calls.append(kwargs)
        return FakeOpenAIClient(fake_create)

    monkeypatch.setattr("bot.openai_backend.openai.AsyncOpenAI", fake_client_ctor)

    result = await generate_vl_response(
        image_url=str(tmp_path / "image.png"),
        user_prompt="what is this?",
    )

    assert result["text"] == "OK NVIDIA VISION"
    assert result["model"] == "nvidia-vl-b"
    assert client_calls[0]["api_key"] == "openrouter-key"
    assert (
        str(client_calls[0]["base_url"]).rstrip("/") == "https://openrouter.ai/api/v1"
    )
    assert client_calls[1]["api_key"] == "nvidia-test-key"
    assert (
        str(client_calls[1]["base_url"]).rstrip("/")
        == "https://integrate.api.nvidia.com/v1"
    )


@pytest.mark.asyncio
async def test_image_generation_ladder_tries_nvidia_models_in_order(monkeypatch):
    config = {
        "VISION_API_KEY": "generic-vision-key",
        "NVIDIA_NIM_API_KEY": "nvidia-test-key",
        "NVIDIA_NIM_API_BASE": "https://integrate.api.nvidia.com/v1",
        "VISION_ALLOWED_PROVIDERS": ["nvidia"],
        "VISION_DEFAULT_PROVIDER": "nvidia",
        "VISION_IMAGE_FALLBACK_MODELS": "nvidia|black-forest-labs/flux.1-dev,nvidia|black-forest-labs/flux.1-schnell",
    }
    adapter = UnifiedVisionAdapter(config)

    seen_models = []

    async def fake_submit(normalized_request):
        seen_models.append(normalized_request.preferred_model)
        if normalized_request.preferred_model == "black-forest-labs/flux.1-dev":
            raise VisionError(
                error_type=VisionErrorType.RATE_LIMITED,
                message="rate limited",
                user_message="try later",
                provider=VisionProvider.NVIDIA,
            )
        return "job-qwen"

    provider = adapter.providers["nvidia"]
    monkeypatch.setattr(provider, "submit", fake_submit)

    request = VisionRequest(
        task=VisionTask.TEXT_TO_IMAGE,
        prompt="a robot",
        user_id="u1",
    )

    response = await adapter.submit(request)

    assert response.success is True
    assert response.job_id == "nvidia:job-qwen"
    assert response.provider == VisionProvider.NVIDIA
    assert response.model_used == "black-forest-labs/flux.1-schnell"
    assert seen_models == [
        "black-forest-labs/flux.1-dev",
        "black-forest-labs/flux.1-schnell",
    ]


@pytest.mark.asyncio
async def test_nvidia_image_submit_uses_nvidia_endpoint_and_returns_fetchable_result(
    monkeypatch,
):
    config = {
        "NVIDIA_NIM_API_KEY": "nvidia-test-key",
        "NVIDIA_NIM_API_BASE": "https://integrate.api.nvidia.com/v1",
        "VISION_ALLOWED_PROVIDERS": ["nvidia"],
        "VISION_DEFAULT_PROVIDER": "nvidia",
        "VISION_IMAGE_FALLBACK_MODELS": "nvidia|black-forest-labs/flux.1-dev",
    }
    adapter = UnifiedVisionAdapter(config)
    provider = adapter.providers["nvidia"]
    captured = {}

    class FakeResponse:
        status = 200

        async def text(self):
            return '{"artifacts":[{"base64":"AAAA"}]}'

        async def json(self):
            return {"artifacts": [{"base64": "AAAA"}]}

    class FakePostContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakePostContext()

    provider.session = FakeSession()

    response = await adapter.submit(
        VisionRequest(task=VisionTask.TEXT_TO_IMAGE, prompt="a robot", user_id="u1")
    )
    result = await adapter.fetch_result(response.job_id)

    assert response.provider == VisionProvider.NVIDIA
    assert response.model_used == "black-forest-labs/flux.1-dev"
    assert (
        captured["url"]
        == "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"
    )
    assert captured["json"] == {"prompt": "a robot"}
    assert captured["headers"]["Authorization"] == "Bearer nvidia-test-key"
    assert result.provider_used == "nvidia"
    assert result.assets == ["data:image/jpeg;base64,AAAA"]


def test_unified_adapter_update_config_refreshes_hotloaded_vision_fields():
    adapter = UnifiedVisionAdapter(
        {
            "VISION_API_KEY": "generic-vision-key",
            "NVIDIA_NIM_API_KEY": "old-nvidia-key",
            "VISION_ALLOWED_PROVIDERS": ["novita"],
            "VISION_DEFAULT_PROVIDER": "novita",
            "VISION_MODEL": "novita:qwen-image",
        }
    )

    adapter.update_config(
        {
            "VISION_API_KEY": "generic-vision-key",
            "NVIDIA_NIM_API_KEY": "new-nvidia-key",
            "VISION_ALLOWED_PROVIDERS": "nvidia",
            "VISION_DEFAULT_PROVIDER": "nvidia",
            "VISION_MODEL": "nvidia:black-forest-labs/flux.1-schnell",
        }
    )

    assert adapter.allowed_providers == ["nvidia"]
    assert adapter.default_provider == "nvidia"
    assert adapter.vision_model_override == "nvidia:black-forest-labs/flux.1-schnell"
    assert adapter.providers["nvidia"].api_key == "new-nvidia-key"
