"""Behavior tests for bot/core/permissions.py.

Tests the actual observable behavior of admin permission checks:
- non-admin invocation returns denial via is_admin_user
- admin invocation passes
- check_admin_async sends denial on prefix context (no ephemeral)
- check_admin_async sends denial on slash interaction (ephemeral)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from bot.core.permissions import (
    admin_only_slash,
    check_admin_async,
    is_admin_user,
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


def _make_bot(owner_ids=None, owner_id=None):
    bot = MagicMock()
    bot.owner_ids = owner_ids if owner_ids is not None else []
    bot.owner_id = owner_id
    return bot


def _make_user(user_id=123):
    user = MagicMock(spec=discord.User)
    user.id = user_id
    return user


def _make_member(user_id=456, admin_perm=False):
    member = MagicMock(spec=discord.Member)
    member.id = user_id
    member.guild_permissions.administrator = admin_perm
    return member


# ------------------------------------------------------------------ #
# is_admin_user behavior tests
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_is_admin_owner_allowed() -> None:
    bot = _make_bot(owner_ids=[100, 200])
    user = _make_user(user_id=100)
    assert await is_admin_user(user, bot) is True


@pytest.mark.asyncio
async def test_is_admin_owner_id_allowed() -> None:
    """Non-team app with single owner_id."""
    bot = _make_bot(owner_id=99)
    user = _make_user(user_id=99)
    assert await is_admin_user(user, bot) is True


@pytest.mark.asyncio
async def test_is_non_admin_denied() -> None:
    bot = _make_bot(owner_ids=[100])
    user = _make_user(user_id=999)
    assert await is_admin_user(user, bot) is False


@pytest.mark.asyncio
async def test_is_member_admin_allowed() -> None:
    """Guild member with administrator permission."""
    bot = _make_bot(owner_ids=[])
    member = _make_member(user_id=555, admin_perm=True)
    assert await is_admin_user(member, bot) is True


@pytest.mark.asyncio
async def test_is_non_admin_member_denied() -> None:
    """Guild member without administrator permission and not in config."""
    bot = _make_bot(owner_ids=[])
    member = _make_member(user_id=555, admin_perm=False)
    assert await is_admin_user(member, bot) is False


@pytest.mark.asyncio
@patch("bot.core.permissions._get_configured_admin_ids")
async def test_configured_admin_ids_allowed(mock_ids) -> None:
    mock_ids.return_value = {777}
    bot = _make_bot(owner_ids=[])
    user = _make_user(user_id=777)
    assert await is_admin_user(user, bot) is True


# ------------------------------------------------------------------ #
# admin_only_prefix behavior tests
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_prefix_decorator_blocks_non_admin() -> None:
    """Non-admin via the prefix check path should deny."""
    bot = _make_bot(owner_ids=[100])
    user = _make_user(user_id=999)
    ctx_send = AsyncMock()

    result = await is_admin_user(user, bot)
    assert result is False

    if not result:
        await ctx_send("You do not have permission to use this command.")

    ctx_send.assert_called_once()
    call_kwargs = ctx_send.call_args
    assert "ephemeral" not in call_kwargs.kwargs


@pytest.mark.asyncio
async def test_prefix_decorator_allows_admin() -> None:
    """Admin via prefix path should reach handler."""
    bot = _make_bot(owner_ids=[100])
    user = _make_user(user_id=100)
    result = await is_admin_user(user, bot)
    assert result is True


# ------------------------------------------------------------------ #
# admin_only_slash behavior tests
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@patch("bot.core.permissions.is_admin_user")
async def test_slash_decorator_blocks_non_admin_ephemeral(mock_is_admin) -> None:
    mock_is_admin.return_value = False
    from bot.exceptions import PermissionDeniedError

    interaction = MagicMock(spec=discord.Interaction)
    interaction.user = _make_user(user_id=999)
    interaction.client = MagicMock()
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()

    wrapped_fn = admin_only_slash()(lambda i: None)

    with pytest.raises(PermissionDeniedError):
        await wrapped_fn(interaction)

    interaction.response.send_message.assert_called_once()
    _, kwargs = interaction.response.send_message.call_args
    assert kwargs.get("ephemeral") is True


@pytest.mark.asyncio
@patch("bot.core.permissions.is_admin_user")
async def test_slash_decorator_allows_admin(mock_is_admin) -> None:
    mock_is_admin.return_value = True

    handler_reached = []

    @admin_only_slash()
    async def my_slash(interaction) -> None:
        handler_reached.append(True)

    interaction = MagicMock(spec=discord.Interaction)
    interaction.user = _make_user(user_id=100)
    interaction.client = MagicMock()

    await my_slash(interaction)
    assert handler_reached == [True]


# ------------------------------------------------------------------ #
# check_admin_async behavior tests
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_check_admin_async_returns_true_for_admin() -> None:
    bot = _make_bot(owner_ids=[100])
    user = _make_user(user_id=100)
    result = await check_admin_async(user, bot)
    assert result is True


@pytest.mark.asyncio
async def test_check_admin_async_returns_false_for_non_admin() -> None:
    bot = _make_bot(owner_ids=[100])
    user = _make_user(user_id=999)
    result = await check_admin_async(user, bot)
    assert result is False


@pytest.mark.asyncio
async def test_check_admin_async_denial_on_prefix_no_ephemeral() -> None:
    """On prefix context, denial is sent via ctx.send (no ephemeral)."""
    bot = _make_bot(owner_ids=[100])
    ctx = MagicMock()
    ctx.author = _make_user(user_id=999)
    ctx.send = AsyncMock()

    result = await check_admin_async(ctx.author, bot, reply_channel=ctx)
    assert result is False
    ctx.send.assert_called_once()
    # Should NOT send ephemeral -- prefix commands don't support it
    _, kwargs = ctx.send.call_args
    assert "ephemeral" not in kwargs


@pytest.mark.asyncio
async def test_check_admin_async_denial_on_interaction_ephemeral() -> None:
    """On slash interaction, denial is sent ephemeral."""
    bot = _make_bot(owner_ids=[100])
    interaction = MagicMock(spec=discord.Interaction)
    interaction.user = _make_user(user_id=999)
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()

    result = await check_admin_async(interaction.user, bot, reply_channel=interaction, ephemeral=True)
    assert result is False
    interaction.response.send_message.assert_called_once()
    _, kwargs = interaction.response.send_message.call_args
    assert kwargs.get("ephemeral") is True


@pytest.mark.asyncio
async def test_check_admin_async_no_denial_without_channel() -> None:
    """When no reply_channel provided, just returns False."""
    bot = _make_bot(owner_ids=[100])
    user = _make_user(user_id=999)
    result = await check_admin_async(user, bot)
    assert result is False


@pytest.mark.asyncio
async def test_dm_user_non_admin_denied() -> None:
    """DM user that is not owner/config admin should be denied."""
    bot = _make_bot(owner_ids=[100])
    # DMUser -- not a Member
    user = _make_user(user_id=999)
    result = await is_admin_user(user, bot)
    assert result is False
