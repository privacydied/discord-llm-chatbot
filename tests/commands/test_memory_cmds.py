"""
Tests for the MemoryCommands cog.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

import discord

from bot.commands.memory_cmds import MemoryCommands

#pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_bot():
    """Fixture for a mocked bot object."""
    bot = MagicMock()
    bot.router = AsyncMock()
    bot.wait_for = AsyncMock()
    return bot

@pytest.fixture
def mock_ctx(mock_bot):
    """Fixture for a mocked context object."""
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.id = 12345
    ctx.author.name = "TestUser"
    ctx.guild = MagicMock()
    ctx.guild.id = 54321
    ctx.send = AsyncMock()
    ctx.send_help = AsyncMock()
    ctx.bot = mock_bot
    return ctx

@pytest.fixture
def memory_cog(mock_bot):
    """Fixture for the MemoryCommands cog."""
    cog = MemoryCommands(mock_bot)
    # Patch the config to avoid loading a real file
    cog.config = {
        "MAX_MEMORIES": 50,
        "MAX_SERVER_MEMORIES": 100
    }
    return cog

@pytest.mark.asyncio
async def test_add_memory_cmd_success(memory_cog, mock_ctx, monkeypatch):
    """Verify that a user can successfully add a memory."""
    # Mock the memory backend functions
    mock_get_profile = MagicMock(return_value={'memories': []})
    mock_save_profile = MagicMock(return_value=True)
    monkeypatch.setattr('bot.commands.memory_cmds.get_profile', mock_get_profile)
    monkeypatch.setattr('bot.commands.memory_cmds.save_profile', mock_save_profile)

    content = "This is a new memory."
    await memory_cog.add_memory_cmd.callback(memory_cog, mock_ctx, content=content)

    # Verify that the profile was loaded and saved correctly
    mock_get_profile.assert_called_once_with(str(mock_ctx.author.id), str(mock_ctx.author))
    mock_save_profile.assert_called_once()
    
    # Verify the content of the saved profile
    saved_profile = mock_save_profile.call_args[0][0]
    assert len(saved_profile['memories']) == 1
    assert saved_profile['memories'][0]['content'] == content

    # Verify the confirmation message
    mock_ctx.send.assert_called_once_with("✅ Memory added! You now have 1 memories.")

@pytest.mark.asyncio
async def test_add_memory_cmd_save_fails(memory_cog, mock_ctx, monkeypatch):
    """Verify error handling when saving a memory fails."""
    mock_get_profile = MagicMock(return_value={'memories': []})
    mock_save_profile = MagicMock(return_value=False)
    monkeypatch.setattr('bot.commands.memory_cmds.get_profile', mock_get_profile)
    monkeypatch.setattr('bot.commands.memory_cmds.save_profile', mock_save_profile)

    await memory_cog.add_memory_cmd.callback(memory_cog, mock_ctx, content="won't save")

    mock_ctx.send.assert_called_once_with("❌ Failed to save memory. Please try again.")

@pytest.mark.asyncio
async def test_list_memories_cmd_success(memory_cog, mock_ctx, monkeypatch):
    """Verify that a user can successfully list their memories."""
    memories = [{'content': 'memory 1'}, {'content': 'memory 2'}]
    mock_get_profile = MagicMock(return_value={'memories': memories})
    monkeypatch.setattr('bot.commands.memory_cmds.get_profile', mock_get_profile)

    await memory_cog.list_memories_cmd.callback(memory_cog, mock_ctx, limit=5)

    mock_get_profile.assert_called_once_with(str(mock_ctx.author.id))
    mock_ctx.send.assert_called_once()
    # Check that an embed was sent
    assert isinstance(mock_ctx.send.call_args[1]['embed'], discord.Embed)

@pytest.mark.asyncio
async def test_list_memories_cmd_no_memories(memory_cog, mock_ctx, monkeypatch):
    """Verify the response when a user has no memories."""
    mock_get_profile = MagicMock(return_value={'memories': []})
    monkeypatch.setattr('bot.commands.memory_cmds.get_profile', mock_get_profile)

    await memory_cog.list_memories_cmd.callback(memory_cog, mock_ctx, limit=5)

    mock_ctx.send.assert_called_once_with("You don't have any memories yet. Use `!memory add <content>` to add one!")

@pytest.mark.asyncio
async def test_clear_memories_cmd_confirmed(memory_cog, mock_ctx, monkeypatch):
    """Verify that memories are cleared after confirmation."""
    mock_get_profile = MagicMock(return_value={'memories': [{'content': 'a memory'}]})
    mock_save_profile = MagicMock(return_value=True)
    monkeypatch.setattr('bot.commands.memory_cmds.get_profile', mock_get_profile)
    monkeypatch.setattr('bot.commands.memory_cmds.save_profile', mock_save_profile)

    # Simulate user confirming by returning a mock message
    mock_ctx.bot.wait_for.return_value = MagicMock()

    await memory_cog.clear_memories_cmd.callback(memory_cog, mock_ctx)

    # Verify that the profile was saved with an empty memory list
    saved_profile = mock_save_profile.call_args[0][0]
    assert saved_profile['memories'] == []
    mock_ctx.send.assert_called_with("✅ Successfully cleared 1 memories.")

@pytest.mark.asyncio
async def test_clear_memories_cmd_timeout(memory_cog, mock_ctx, monkeypatch):
    """Verify that clearing is cancelled on timeout."""
    # Simulate a timeout
    mock_ctx.bot.wait_for.side_effect = asyncio.TimeoutError
    confirm_msg = AsyncMock()
    mock_ctx.send.return_value = confirm_msg

    await memory_cog.clear_memories_cmd.callback(memory_cog, mock_ctx)

    # Verify the confirmation message was edited to show cancellation
    confirm_msg.edit.assert_called_once_with(content="Memory clear cancelled due to timeout.")

@pytest.mark.asyncio
async def test_memory_group_no_subcommand(memory_cog, mock_ctx):
    """Verify that the help command is sent when no subcommand is given."""
    mock_ctx.invoked_subcommand = None
    await memory_cog.memory_group.callback(memory_cog, mock_ctx)
    mock_ctx.send_help.assert_called_once_with(mock_ctx.command)

# --- Server Memory Command Tests ---

@pytest.fixture
def mock_admin_ctx(mock_ctx):
    """Fixture for a mocked context with admin permissions."""
    mock_ctx.author.guild_permissions.administrator = True
    return mock_ctx

@pytest.mark.asyncio
async def test_server_memory_add_success(memory_cog, mock_admin_ctx, monkeypatch):
    """Verify an admin can successfully add a server memory."""
    mock_get = MagicMock(return_value={'memories': []})
    mock_save = MagicMock(return_value=True)
    monkeypatch.setattr('bot.commands.memory_cmds.get_server_profile', mock_get)
    monkeypatch.setattr('bot.commands.memory_cmds.save_server_profile', mock_save)

    content = "This is a new server memory."
    await memory_cog.server_memory_add.callback(memory_cog, mock_admin_ctx, content=content)

    mock_get.assert_called_once_with(str(mock_admin_ctx.guild.id), mock_admin_ctx.guild.name)
    mock_save.assert_called_once()
    saved_profile = mock_save.call_args[0][1]
    assert len(saved_profile['memories']) == 1
    assert saved_profile['memories'][0]['content'] == content
    mock_admin_ctx.send.assert_called_once_with("✅ Server memory added! There are now 1 server memories.")

@pytest.mark.asyncio
async def test_server_memory_list_success(memory_cog, mock_admin_ctx, monkeypatch):
    """Verify an admin can successfully list server memories."""
    memories = [{'content': 'server memory 1'}]
    mock_get = MagicMock(return_value={'memories': memories})
    monkeypatch.setattr('bot.commands.memory_cmds.get_server_profile', mock_get)

    await memory_cog.server_memory_list.callback(memory_cog, mock_admin_ctx)

    mock_get.assert_called_once_with(str(mock_admin_ctx.guild.id))
    mock_admin_ctx.send.assert_called_once()
    assert isinstance(mock_admin_ctx.send.call_args[1]['embed'], discord.Embed)

@pytest.mark.asyncio
async def test_server_memory_list_no_memories(memory_cog, mock_admin_ctx, monkeypatch):
    """Verify the response when a server has no memories."""
    mock_get = MagicMock(return_value={'memories': []})
    monkeypatch.setattr('bot.commands.memory_cmds.get_server_profile', mock_get)

    await memory_cog.server_memory_list.callback(memory_cog, mock_admin_ctx)

    mock_admin_ctx.send.assert_called_once_with("No server memories found. Use `!server-memory add <content>` to add one!")

@pytest.mark.asyncio
async def test_server_memory_clear_confirmed(memory_cog, mock_admin_ctx, monkeypatch):
    """Verify that server memories are cleared after admin confirmation."""
    mock_get = MagicMock(return_value={'memories': [{'content': 'a memory'}]})
    mock_save = MagicMock(return_value=True)
    monkeypatch.setattr('bot.commands.memory_cmds.get_server_profile', mock_get)
    monkeypatch.setattr('bot.commands.memory_cmds.save_server_profile', mock_save)

    mock_admin_ctx.bot.wait_for.return_value = MagicMock()

    await memory_cog.server_memory_clear.callback(memory_cog, mock_admin_ctx)

    saved_profile = mock_save.call_args[0][1]
    assert saved_profile['memories'] == []
    mock_admin_ctx.send.assert_called_with("✅ Successfully cleared 1 server memories.")

@pytest.mark.asyncio
async def test_server_memory_group_no_subcommand(memory_cog, mock_admin_ctx):
    """Verify help is sent when no server-memory subcommand is given."""
    mock_admin_ctx.invoked_subcommand = None
    await memory_cog.server_memory_group.callback(memory_cog, mock_admin_ctx)
    mock_admin_ctx.send_help.assert_called_once_with(mock_admin_ctx.command)
