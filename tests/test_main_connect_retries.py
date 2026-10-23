"""bot.main.connect_with_retries: survive Discord gateway blips without re-running setup. [REH]"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import discord
import pytest

from bot.main import _is_transient_connect_error, connect_with_retries


def _handshake_503_then_attribute_error() -> AttributeError:
    """The real 2026-10-23 failure: discord.py's resume path hits ws=None while handling a 503."""
    try:
        try:
            raise aiohttp.WSServerHandshakeError(request_info=MagicMock(), history=(), status=503, message="Invalid response status")
        except aiohttp.WSServerHandshakeError:
            None.sequence  # noqa: B018 - reproduce discord.py's AttributeError
    except AttributeError as exc:
        return exc
    raise AssertionError("unreachable")


def test_transient_classification():
    assert _is_transient_connect_error(_handshake_503_then_attribute_error())
    assert _is_transient_connect_error(aiohttp.ClientConnectorError(MagicMock(), OSError("down")))
    assert _is_transient_connect_error(discord.HTTPException(SimpleNamespace(status=502, reason="bad gateway"), "x"))
    assert not _is_transient_connect_error(discord.HTTPException(SimpleNamespace(status=403, reason="no"), "x"))
    assert not _is_transient_connect_error(discord.LoginFailure("bad token"))
    assert not _is_transient_connect_error(ValueError("real bug"))


@pytest.mark.asyncio
async def test_retries_connect_only_and_logs_in_once():
    bot = SimpleNamespace(login=AsyncMock(), connect=AsyncMock(side_effect=[_handshake_503_then_attribute_error(), None]))
    sleep = AsyncMock()
    assert await connect_with_retries(bot, "tok", MagicMock(), sleep=sleep) == 0
    bot.login.assert_awaited_once_with("tok")  # setup_hook must not run twice
    assert bot.connect.await_count == 2
    sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_bad_token_and_real_bugs_exit_without_retry():
    sleep = AsyncMock()
    bad_token = SimpleNamespace(login=AsyncMock(side_effect=discord.LoginFailure("no")), connect=AsyncMock())
    assert await connect_with_retries(bad_token, "tok", MagicMock(), sleep=sleep) == 1
    bug = SimpleNamespace(login=AsyncMock(), connect=AsyncMock(side_effect=ValueError("bug")))
    assert await connect_with_retries(bug, "tok", MagicMock(), sleep=sleep) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts_with_capped_backoff():
    bot = SimpleNamespace(login=AsyncMock(), connect=AsyncMock(side_effect=OSError("network down")))
    sleep = AsyncMock()
    assert await connect_with_retries(bot, "tok", MagicMock(), sleep=sleep) == 1
    assert [c.args[0] for c in sleep.await_args_list] == [5.0, 10.0, 20.0, 40.0, 60.0]
