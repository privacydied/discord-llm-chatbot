import types
import pytest
import httpx

from bot.exceptions import APIError
from bot import retry_utils as retry_utils_mod
from bot.openai_backend import generate_openai_response
from bot.enhanced_retry import get_retry_manager, ProviderConfig


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
