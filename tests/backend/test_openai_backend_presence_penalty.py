"""Regression test: PRESENCE_PENALTY was parsed into config but never passed
into the actual chat.completions.create() call - it was a dead config value
for the OpenAI/OpenRouter text backend (unlike TEMPERATURE, which was already
wired). generate_openai_response() now resolves and forwards it the same way
temperature is resolved/forwarded. [CMV]
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bot.openai_backend import generate_openai_response


def _fake_completion_response(text: str = "hello there"):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[choice], usage=usage)


def _fake_client(create_mock):
    completions = SimpleNamespace(create=create_mock)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


@pytest.mark.asyncio
async def test_presence_penalty_from_config_is_forwarded_to_api_call() -> None:
    create_mock = AsyncMock(return_value=_fake_completion_response())
    fake_client = _fake_client(create_mock)

    config = {
        "TEXT_BACKEND": "openai",
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "OPENAI_TEXT_MODEL": "gpt-4",
        "TEMPERATURE": 0.2,
        "PRESENCE_PENALTY": 0.5,
        "MAX_RESPONSE_TOKENS": 500,
    }

    with (
        patch("bot.openai_backend.load_config", return_value=config),
        patch("bot.openai_backend._make_openai_async_client", return_value=fake_client),
        patch("bot.openai_backend._safe_aclose_openai_client", new=AsyncMock()),
    ):
        result = await generate_openai_response(
            prompt="hi",
            system_prompt="be nice",
        )

    assert result["text"] == "hello there"
    create_mock.assert_awaited_once()
    call_kwargs = create_mock.await_args.kwargs
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["presence_penalty"] == 0.5


@pytest.mark.asyncio
async def test_presence_penalty_defaults_to_zero_when_unset() -> None:
    create_mock = AsyncMock(return_value=_fake_completion_response())
    fake_client = _fake_client(create_mock)

    config = {
        "TEXT_BACKEND": "openai",
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "OPENAI_TEXT_MODEL": "gpt-4",
    }

    with (
        patch("bot.openai_backend.load_config", return_value=config),
        patch("bot.openai_backend._make_openai_async_client", return_value=fake_client),
        patch("bot.openai_backend._safe_aclose_openai_client", new=AsyncMock()),
    ):
        await generate_openai_response(prompt="hi", system_prompt="be nice")

    assert create_mock.await_args.kwargs["presence_penalty"] == 0.0


@pytest.mark.asyncio
async def test_explicit_presence_penalty_argument_overrides_config() -> None:
    create_mock = AsyncMock(return_value=_fake_completion_response())
    fake_client = _fake_client(create_mock)

    config = {
        "TEXT_BACKEND": "openai",
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "OPENAI_TEXT_MODEL": "gpt-4",
        "PRESENCE_PENALTY": 0.5,
    }

    with (
        patch("bot.openai_backend.load_config", return_value=config),
        patch("bot.openai_backend._make_openai_async_client", return_value=fake_client),
        patch("bot.openai_backend._safe_aclose_openai_client", new=AsyncMock()),
    ):
        await generate_openai_response(prompt="hi", system_prompt="be nice", presence_penalty=1.0)

    assert create_mock.await_args.kwargs["presence_penalty"] == 1.0
