from types import SimpleNamespace
from typing import Never

import pytest

from bot.router import Router


@pytest.mark.asyncio
async def test_resolve_reference_message_prefers_fallback() -> None:
    router = object.__new__(Router)
    fallback = SimpleNamespace(id=10)
    message = SimpleNamespace(reference=None)
    out = await router._resolve_reference_message(message, fallback=fallback)
    assert out is fallback


@pytest.mark.asyncio
async def test_resolve_reference_message_no_reference() -> None:
    router = object.__new__(Router)
    message = SimpleNamespace(reference=None)
    out = await router._resolve_reference_message(message)
    assert out is None


@pytest.mark.asyncio
async def test_resolve_reference_message_uses_resolved() -> None:
    router = object.__new__(Router)
    resolved = SimpleNamespace(id=20)
    message = SimpleNamespace(reference=SimpleNamespace(resolved=resolved, message_id=30))
    out = await router._resolve_reference_message(message)
    assert out is resolved


@pytest.mark.asyncio
async def test_resolve_reference_message_fetches_when_needed() -> None:
    router = object.__new__(Router)
    fetched = SimpleNamespace(id=40)

    class _Channel:
        async def fetch_message(self, _msg_id):
            return fetched

    message = SimpleNamespace(
        reference=SimpleNamespace(resolved=None, message_id=40),
        channel=_Channel(),
    )
    out = await router._resolve_reference_message(message)
    assert out is fetched


@pytest.mark.asyncio
async def test_resolve_reference_message_fetch_failure_returns_none() -> None:
    router = object.__new__(Router)

    class _Channel:
        async def fetch_message(self, _msg_id) -> Never:
            msg = "boom"
            raise RuntimeError(msg)

    message = SimpleNamespace(
        reference=SimpleNamespace(resolved=None, message_id=50),
        channel=_Channel(),
    )
    out = await router._resolve_reference_message(message)
    assert out is None
