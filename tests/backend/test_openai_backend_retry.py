import types
import pytest
import httpx

from bot.exceptions import APIError
from bot import retry_utils as retry_utils_mod
from bot.openai_backend import generate_openai_response
from bot.nvidia_backend import generate_nvidia_response
from bot.enhanced_retry import (
    EnhancedRetryManager,
    ProviderStatus,
    get_retry_manager,
    ProviderConfig,
)


def make_httpx_429(retry_after: float) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(
        429, headers={"Retry-After": str(retry_after)}, request=request
    )
    return httpx.HTTPStatusError(
        "429 Too Many Requests", request=request, response=response
    )


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
async def test_text_backend_openrouter_alias_routes_to_openai_backend(monkeypatch):
    def fake_load_config():
        return {"TEXT_BACKEND": "openrouter"}

    async def fake_generate_openai_response(**kwargs):
        return {"text": "ok", "backend": "openai"}

    monkeypatch.setattr("bot.ai_backend.load_config", fake_load_config)
    monkeypatch.setattr(
        "bot.openai_backend.generate_openai_response", fake_generate_openai_response
    )

    from bot.ai_backend import generate_response

    result = await generate_response("hello", system_prompt="test")

    assert result["text"] == "ok"


def test_text_ladder_env_can_force_single_attempt_per_model(monkeypatch):
    monkeypatch.setenv(
        "TEXT_FALLBACK_MODELS",
        "nvidia|deepseek-ai/deepseek-v4-pro,nvidia|qwen/qwen3.5-397b-a17b",
    )
    monkeypatch.setenv("TEXT_FALLBACK_TIMEOUTS", "35,35")
    monkeypatch.setenv("TEXT_FALLBACK_MAX_ATTEMPTS", "1")

    mgr = EnhancedRetryManager()

    assert [
        (p.name, p.model, p.timeout, p.max_attempts)
        for p in mgr.provider_configs["text"]
    ] == [
        ("nvidia", "deepseek-ai/deepseek-v4-pro", 35.0, 1),
        ("nvidia", "qwen/qwen3.5-397b-a17b", 35.0, 1),
    ]


def test_text_ladder_env_dedupes_exact_duplicate_entries(monkeypatch):
    monkeypatch.setenv(
        "TEXT_FALLBACK_MODELS",
        "openrouter|minimax/minimax-m2.5:free,openrouter|tencent/hy3-preview:free,openrouter|minimax/minimax-m2.5:free",
    )
    monkeypatch.setenv("TEXT_FALLBACK_TIMEOUTS", "35,35,35,35")
    monkeypatch.setenv("TEXT_FALLBACK_MAX_ATTEMPTS", "1")

    mgr = EnhancedRetryManager()

    assert [p.model for p in mgr.provider_configs["text"]] == [
        "minimax/minimax-m2.5:free",
        "tencent/hy3-preview:free",
    ]


def test_empty_text_response_is_retryable_for_ladder():
    mgr = EnhancedRetryManager()

    assert mgr._is_retryable_error(
        APIError(
            "Response normalization failed: APIError: Empty text response from model tencent/hy3-preview:free"
        )
    )


@pytest.mark.asyncio
async def test_nvidia_backend_preserves_actual_ladder_model(monkeypatch):
    monkeypatch.setenv(
        "TEXT_FALLBACK_MODELS",
        "nvidia|deepseek-ai/deepseek-v4-pro,nvidia|z-ai/glm-5.1",
    )

    def fake_load_config():
        return {
            "TEXT_BACKEND": "nvidia",
            "NVIDIA_NIM_TEXT_MODEL": "z-ai/glm-5.1",
            "OPENAI_TEXT_MODEL": "deepseek-ai/deepseek-v4-pro",
        }

    async def fake_generate_openai_response(**kwargs):
        return {"text": "ok", "model": "deepseek-ai/deepseek-v4-pro"}

    monkeypatch.setattr("bot.nvidia_backend.load_config", fake_load_config)
    monkeypatch.setattr(
        "bot.openai_backend.generate_openai_response", fake_generate_openai_response
    )

    result = await generate_nvidia_response("hello", stream=False, system_prompt="test")

    assert result["backend"] == "nvidia_nim"
    assert result["model"] == "deepseek-ai/deepseek-v4-pro"
    assert result["configured_model"] == "deepseek-ai/deepseek-v4-pro"


@pytest.mark.asyncio
async def test_retry_after_affects_backoff_delay(monkeypatch):
    # Patch config to use OpenAI base (no OpenRouter fallback)
    def fake_load_config():
        return {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE": "https://api.openai.com/v1",
            "OPENAI_TEXT_MODEL": "gpt-4",
            "TEMPERATURE": 0.1,
        }

    monkeypatch.setattr("bot.openai_backend.load_config", fake_load_config)

    # Control retry config for determinism
    retry_utils_mod.API_RETRY_CONFIG.jitter = False
    retry_utils_mod.API_RETRY_CONFIG.max_attempts = 2

    delays = []

    async def fake_sleep(d):
        delays.append(d)
        # do not actually sleep

    # Patch asyncio.sleep used inside retry_utils
    monkeypatch.setattr(retry_utils_mod.asyncio, "sleep", fake_sleep)

    call_count = {"n": 0}

    async def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise make_httpx_429(4.0)

        # Success second time
        class _Usage:
            prompt_tokens = 1
            completion_tokens = 2
            total_tokens = 3

        class _Msg:
            content = "OK"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()

    # Patch openai.AsyncOpenAI constructor to return our fake client
    monkeypatch.setattr(
        "bot.openai_backend.openai.AsyncOpenAI",
        lambda **kwargs: FakeOpenAIClient(fake_create),
    )

    # Execute
    result = await generate_openai_response(
        "hello", stream=False, system_prompt="test-system"
    )

    assert result["text"] == "OK"
    # Ensure delay respected Retry-After (max(delay, retry_after))
    assert len(delays) == 1
    assert abs(delays[0] - 4.0) < 0.01


@pytest.mark.asyncio
async def test_rate_limit_apierror_carries_retry_after_seconds_when_no_retry(
    monkeypatch,
):
    def fake_load_config():
        return {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE": "https://api.openai.com/v1",
            "OPENAI_TEXT_MODEL": "gpt-4",
        }

    monkeypatch.setattr("bot.openai_backend.load_config", fake_load_config)

    # Disable retry so the first APIError is propagated
    retry_utils_mod.API_RETRY_CONFIG.max_attempts = 1

    async def fake_create(**kwargs):
        raise make_httpx_429(3.5)

    monkeypatch.setattr(
        "bot.openai_backend.openai.AsyncOpenAI",
        lambda **kwargs: FakeOpenAIClient(fake_create),
    )

    with pytest.raises(APIError) as ei:
        await generate_openai_response(
            "hello", stream=False, system_prompt="test-system"
        )

    e = ei.value
    assert hasattr(e, "retry_after_seconds")
    assert abs(float(e.retry_after_seconds) - 3.5) < 0.01


@pytest.mark.asyncio
async def test_nvidia_base_uses_text_fallback_ladder_and_nvidia_key(monkeypatch):
    def fake_load_config():
        return {
            "TEXT_BACKEND": "nvidia",
            "OPENAI_API_KEY": "old-openrouter-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
            "OPENAI_TEXT_MODEL": "old-openrouter-model",
            "NVIDIA_NIM_API_KEY": "nvapi-test-key",
            "NVIDIA_NIM_API_BASE": "https://integrate.api.nvidia.com/v1",
            "NVIDIA_NIM_TEXT_MODEL": "model-a",
            "TEXT_PER_ITEM_BUDGET": 10.0,
            "TEMPERATURE": 0.1,
        }

    monkeypatch.setattr("bot.openai_backend.load_config", fake_load_config)

    mgr = get_retry_manager()
    mgr.circuit_breakers.clear()
    mgr.provider_configs["text"] = [
        ProviderConfig("nvidia", "model-a", timeout=2.0, max_attempts=1),
        ProviderConfig("nvidia", "model-b", timeout=2.0, max_attempts=1),
    ]

    captured_client_kwargs = {}

    async def fake_create(**kwargs):
        model = kwargs.get("model")
        if model == "model-a":
            raise make_httpx_429(1.0)

        class _Usage:
            prompt_tokens = 1
            completion_tokens = 2
            total_tokens = 3

        class _Msg:
            content = "OK-NVIDIA-B"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()

    def fake_client_ctor(**kwargs):
        captured_client_kwargs.update(kwargs)
        return FakeOpenAIClient(fake_create)

    monkeypatch.setattr("bot.openai_backend.openai.AsyncOpenAI", fake_client_ctor)

    result = await generate_openai_response(
        "hello", stream=False, system_prompt="test-system"
    )

    assert captured_client_kwargs["api_key"] == "nvapi-test-key"
    assert str(captured_client_kwargs["base_url"]).rstrip("/") == "https://integrate.api.nvidia.com/v1"
    assert result["text"] == "OK-NVIDIA-B"
    assert result["model"] == "model-b"


@pytest.mark.asyncio
async def test_text_ladder_can_mix_openrouter_and_nvidia_endpoints(monkeypatch):
    def fake_load_config():
        return {
            "OPENAI_API_KEY": "openrouter-key",
            "OPENROUTER_API_KEY": "dedicated-openrouter-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
            "OPENAI_TEXT_MODEL": "openrouter-model-a",
            "NVIDIA_NIM_API_KEY": "nvidia-key",
            "NVIDIA_NIM_API_BASE": "https://integrate.api.nvidia.com/v1",
            "TEXT_PER_ITEM_BUDGET": 10.0,
            "TEMPERATURE": 0.1,
        }

    monkeypatch.setattr("bot.openai_backend.load_config", fake_load_config)

    mgr = get_retry_manager()
    mgr.circuit_breakers.clear()
    mgr.provider_configs["text"] = [
        ProviderConfig("openrouter", "openrouter-model-a", timeout=2.0, max_attempts=1),
        ProviderConfig("nvidia", "nvidia-model-b", timeout=2.0, max_attempts=1),
    ]

    client_calls = []

    async def fake_create(**kwargs):
        model = kwargs.get("model")
        if model == "openrouter-model-a":
            raise make_httpx_429(1.0)

        class _Usage:
            prompt_tokens = 1
            completion_tokens = 2
            total_tokens = 3

        class _Msg:
            content = "OK-MIXED-NVIDIA"

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

    result = await generate_openai_response(
        "hello", stream=False, system_prompt="test-system"
    )

    assert result["text"] == "OK-MIXED-NVIDIA"
    assert result["model"] == "nvidia-model-b"
    ladder_calls = client_calls[-2:]
    assert ladder_calls[0]["api_key"] == "dedicated-openrouter-key"
    assert str(ladder_calls[0]["base_url"]).rstrip("/") == "https://openrouter.ai/api/v1"
    assert ladder_calls[1]["api_key"] == "nvidia-key"
    assert str(ladder_calls[1]["base_url"]).rstrip("/") == "https://integrate.api.nvidia.com/v1"


@pytest.mark.asyncio
async def test_text_ladder_falls_back_after_empty_model_response(monkeypatch):
    def fake_load_config():
        return {
            "OPENAI_API_KEY": "openrouter-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
            "OPENAI_TEXT_MODEL": "model-a",
            "TEXT_PER_ITEM_BUDGET": 10.0,
            "TEMPERATURE": 0.1,
        }

    monkeypatch.setattr("bot.openai_backend.load_config", fake_load_config)

    mgr = get_retry_manager()
    mgr.circuit_breakers.clear()
    mgr.provider_configs["text"] = [
        ProviderConfig("openrouter", "model-empty", timeout=2.0, max_attempts=1),
        ProviderConfig("openrouter", "model-good", timeout=2.0, max_attempts=1),
    ]

    calls = []

    async def fake_create(**kwargs):
        model = kwargs.get("model")
        calls.append(model)

        class _Usage:
            prompt_tokens = 1
            completion_tokens = 2
            total_tokens = 3

        class _Msg:
            content = "" if model == "model-empty" else "OK-GOOD"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()

    monkeypatch.setattr(
        "bot.openai_backend.openai.AsyncOpenAI",
        lambda **kwargs: FakeOpenAIClient(fake_create),
    )

    result = await generate_openai_response(
        "hello", stream=False, system_prompt="test-system"
    )

    assert calls[-2:] == ["model-empty", "model-good"]
    assert result["text"] == "OK-GOOD"
    assert result["model"] == "model-good"


@pytest.mark.asyncio
async def test_openrouter_fallback_ladder_selects_second_model(monkeypatch):
    # Force OpenRouter base to engage fallback ladder in openai_backend
    def fake_load_config():
        return {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
            "OPENAI_TEXT_MODEL": "text-model-a",
            "TEXT_PER_ITEM_BUDGET": 10.0,
        }

    monkeypatch.setattr("bot.openai_backend.load_config", fake_load_config)

    # Configure text ladder: first fails, second succeeds
    mgr = get_retry_manager()
    mgr.circuit_breakers.clear()
    mgr.provider_configs["text"] = [
        ProviderConfig("openrouter", "text-model-a", timeout=2.0, max_attempts=1),
        ProviderConfig("openrouter", "text-model-b", timeout=2.0, max_attempts=1),
    ]

    async def fake_create(**kwargs):
        model = kwargs.get("model")
        if model == "text-model-a":
            raise make_httpx_429(1.0)

        # success path
        class _Usage:
            prompt_tokens = 1
            completion_tokens = 2
            total_tokens = 3

        class _Msg:
            content = "OK-B"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()

    monkeypatch.setattr(
        "bot.openai_backend.openai.AsyncOpenAI",
        lambda **kwargs: FakeOpenAIClient(fake_create),
    )

    result = await generate_openai_response(
        "hello", stream=False, system_prompt="test-system"
    )

    assert result["text"] == "OK-B"
    assert result["model"] == "text-model-b"


@pytest.mark.asyncio
async def test_enhanced_retry_manager_attempts_and_fallback_flags_text():
    mgr = get_retry_manager()
    mgr.circuit_breakers.clear()
    mgr.provider_configs["text"] = [
        ProviderConfig("openrouter", "fail-a", timeout=1.0, max_attempts=1),
        ProviderConfig("openrouter", "ok-b", timeout=1.0, max_attempts=1),
    ]

    def factory(pc: ProviderConfig):
        async def run():
            if pc.model == "fail-a":
                raise Exception("429 Too Many Requests")
            return "OK"

        return run

    res = await mgr.run_with_fallback("text", factory, per_item_budget=5.0)
    assert res.success is True
    assert res.provider_used.endswith(":ok-b")
    assert res.attempts == 2
    assert res.fallback_occurred is True


@pytest.mark.asyncio
async def test_streaming_bypasses_fallback_and_streams_chunks(monkeypatch):
    # Even if OpenRouter base, streaming path bypasses fallback ladder
    def fake_load_config():
        return {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
            "OPENAI_TEXT_MODEL": "any-model",
        }

    monkeypatch.setattr("bot.openai_backend.load_config", fake_load_config)

    call_log = {"count": 0}

    async def stream_iter():
        # Yield three chunks resembling OpenAI streaming deltas
        class _Delta:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, content):
                self.delta = _Delta(content)

        # Yield a few chunks
        yield types.SimpleNamespace(choices=[_Choice("Hello")])
        yield types.SimpleNamespace(choices=[_Choice(" ")])
        yield types.SimpleNamespace(choices=[_Choice("world!")])

    async def fake_create(**kwargs):
        call_log["count"] += 1
        assert kwargs.get("stream") is True
        return stream_iter()

    monkeypatch.setattr(
        "bot.openai_backend.openai.AsyncOpenAI",
        lambda **kwargs: FakeOpenAIClient(fake_create),
    )

    gen = await generate_openai_response(
        "hello", stream=True, system_prompt="test-system"
    )
    outs = []
    async for item in gen:
        outs.append(item)

    # Ensure we streamed expected chunks
    text_stream = "".join(x["text"] for x in outs if not x["finished"])
    assert text_stream == "Hello world!"
    assert call_log["count"] == 1


def test_text_fallback_env_ladder_is_not_overridden_by_openai_text_model(monkeypatch):
    monkeypatch.setenv(
        "TEXT_FALLBACK_MODELS", "openrouter|model-a,openrouter|model-b"
    )
    monkeypatch.setenv("OPENAI_TEXT_MODEL", "dead-model-not-in-ladder")
    monkeypatch.delenv("TEXT_FALLBACK_TIMEOUTS", raising=False)

    mgr = EnhancedRetryManager()
    ladder = mgr.provider_configs.get("text", [])
    assert [p.model for p in ladder][:2] == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_404_no_endpoints_opens_long_cooldown_and_skips_next_call(monkeypatch):
    monkeypatch.delenv("TEXT_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("TEXT_FALLBACK_TIMEOUTS", raising=False)
    monkeypatch.setenv("OPENROUTER_DEAD_MODEL_COOLDOWN_S", "1800")

    mgr = EnhancedRetryManager()
    mgr.provider_configs["text"] = [
        ProviderConfig("openrouter", "dead-model", timeout=1.0, max_attempts=1),
        ProviderConfig("openrouter", "ok-model", timeout=1.0, max_attempts=1),
    ]

    calls = {"dead": 0, "ok": 0}

    def factory(pc: ProviderConfig):
        async def run():
            if pc.model == "dead-model":
                calls["dead"] += 1
                raise Exception(
                    "NotFoundError: Error code: 404 - {'error': {'message': 'No endpoints found for dead-model.'}}"
                )
            calls["ok"] += 1
            return "OK"

        return run

    res1 = await mgr.run_with_fallback("text", factory, per_item_budget=5.0)
    assert res1.success is True
    assert calls["dead"] == 1
    assert calls["ok"] == 1

    breaker = mgr._get_circuit_breaker("openrouter:dead-model")
    assert breaker.status == ProviderStatus.CIRCUIT_OPEN
    assert breaker.cooldown_duration >= 1800.0

    res2 = await mgr.run_with_fallback("text", factory, per_item_budget=5.0)
    assert res2.success is True
    # dead model should be skipped on immediate next call due long cooldown
    assert calls["dead"] == 1
    assert calls["ok"] == 2


@pytest.mark.asyncio
async def test_circuit_probe_window_prevents_zero_attempt_exhaustion(monkeypatch):
    monkeypatch.setenv("CIRCUIT_PROBE_WINDOW_S", "0.5")

    mgr = EnhancedRetryManager()
    mgr.provider_configs["text"] = [
        ProviderConfig("openrouter", "model-a", timeout=1.0, max_attempts=1),
        ProviderConfig("openrouter", "model-b", timeout=1.0, max_attempts=1),
    ]

    now = __import__("time").time()
    # Simulate both providers circuit-open but close to cooldown expiry.
    for key in ("openrouter:model-a", "openrouter:model-b"):
        breaker = mgr._get_circuit_breaker(key)
        breaker.status = ProviderStatus.CIRCUIT_OPEN
        breaker.failure_count = breaker.failure_threshold
        breaker.cooldown_duration = 2.0
        breaker.last_failure_time = now - 1.8  # remaining ~= 0.2s (inside probe window)

    calls = {"a": 0, "b": 0}

    def factory(pc: ProviderConfig):
        async def run():
            if pc.model == "model-a":
                calls["a"] += 1
                raise Exception("429 Too Many Requests")
            calls["b"] += 1
            return "OK-B"

        return run

    res = await mgr.run_with_fallback("text", factory, per_item_budget=5.0)
    assert res.success is True
    assert res.result == "OK-B"
    assert res.attempts >= 1
    assert calls["a"] >= 1
    assert calls["b"] == 1


@pytest.mark.asyncio
async def test_auth_401_aborts_ladder_and_sets_provider_used():
    mgr = EnhancedRetryManager()
    mgr.circuit_breakers.clear()
    mgr.provider_configs["text"] = [
        ProviderConfig("openrouter", "auth-fail-model", timeout=1.0, max_attempts=2),
        ProviderConfig("openrouter", "should-not-run", timeout=1.0, max_attempts=1),
    ]

    calls = {"auth": 0, "next": 0}

    def factory(pc: ProviderConfig):
        async def run():
            if pc.model == "auth-fail-model":
                calls["auth"] += 1
                raise Exception(
                    "AuthenticationError: Error code: 401 - {'error': {'message': 'User not found.', 'code': 401}}"
                )
            calls["next"] += 1
            return "OK"

        return run

    res = await mgr.run_with_fallback("text", factory, per_item_budget=10.0)
    assert res.success is False
    assert res.attempts == 1
    assert res.provider_used == "openrouter:auth-fail-model"
    assert calls["auth"] == 1
    assert calls["next"] == 0
