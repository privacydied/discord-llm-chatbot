"""
Test suite for bot initialization idempotency.

Ensures that bot setup runs exactly once, even if setup_hook() or on_ready()
are called multiple times, preventing duplicate initialization.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Any

import discord
from discord.ext import commands

from bot.core.bot import LLMBot


@pytest.fixture
def mock_config():
    """Provide a minimal test configuration."""
    return {
        "DISCORD_TOKEN": "test_token",
        "CONTEXT_FILE_PATH": "test_context.json",
        "MAX_CONTEXT_MESSAGES": 10,
        "ENHANCED_CONTEXT_FILE_PATH": "test_enhanced_context.json",
        "MAX_CONTEXT_TOKENS": 4000,
    }


@pytest.fixture
def mock_intents():
    """Provide test Discord intents."""
    intents = discord.Intents.default()
    intents.message_content = True
    return intents


@pytest.fixture
def test_bot(mock_config, mock_intents):
    """Create a test bot instance with mocked dependencies."""
    with patch('bot.core.bot.load_system_prompts', return_value={}), \
         patch('bot.core.bot.ContextManager'), \
         patch('bot.core.bot.EnhancedContextManager'):
        
        bot = LLMBot(
            config=mock_config,
            command_prefix='!',
            intents=mock_intents,
            help_command=None
        )
        
        # Mock all the setup methods to avoid actual initialization
        bot.load_profiles = AsyncMock()
        bot.setup_background_tasks = MagicMock()
        bot.setup_tts = AsyncMock()
        bot.setup_router = AsyncMock()
        bot.load_extensions = AsyncMock()
        
        return bot


@pytest.mark.asyncio
async def test_setup_hook_idempotency(test_bot):
    """Test that setup_hook() can be called multiple times but only runs once."""
    # Verify initial state
    assert not test_bot._boot_completed
    
    # First call should execute setup
    await test_bot.setup_hook()
    
    # Verify setup completed
    assert test_bot._boot_completed
    test_bot.load_profiles.assert_called_once()
    test_bot.setup_background_tasks.assert_called_once()
    test_bot.setup_tts.assert_called_once()
    test_bot.setup_router.assert_called_once()
    test_bot.load_extensions.assert_called_once()
    
    # Reset mocks to verify second call doesn't execute setup
    test_bot.load_profiles.reset_mock()
    test_bot.setup_background_tasks.reset_mock()
    test_bot.setup_tts.reset_mock()
    test_bot.setup_router.reset_mock()
    test_bot.load_extensions.reset_mock()
    
    # Second call should be skipped
    await test_bot.setup_hook()
    
    # Verify no additional setup calls were made
    test_bot.load_profiles.assert_not_called()
    test_bot.setup_background_tasks.assert_not_called()
    test_bot.setup_tts.assert_not_called()
    test_bot.setup_router.assert_not_called()
    test_bot.load_extensions.assert_not_called()


@pytest.mark.asyncio
async def test_on_ready_idempotency(test_bot):
    """Test that on_ready() can be called multiple times but only logs once."""
    # Mock the user object using patch since it's a property
    mock_user = MagicMock()
    mock_user.id = 12345
    mock_user.__str__ = MagicMock(return_value="TestBot#1234")
    
    with patch.object(type(test_bot), 'user', new_callable=lambda: mock_user):
        # Verify initial state
        assert not test_bot._is_ready.is_set()
        
        # First call should set ready state and log
        with patch.object(test_bot.logger, 'info') as mock_log:
            await test_bot.on_ready()
            
            # Verify ready state is set
            assert test_bot._is_ready.is_set()
            
            # Verify logging calls
            mock_log.assert_has_calls([
                call("🤖 Logged in as TestBot#1234 (ID: 12345)"),
                call("🎉 Bot is ready to receive commands!")
            ])
        
        # Second call should not log again
        with patch.object(test_bot.logger, 'info') as mock_log:
            await test_bot.on_ready()
            
            # Verify no additional logging
            mock_log.assert_not_called()


@pytest.mark.asyncio
async def test_setup_failure_recovery(test_bot):
    """Test that setup failure resets the completion flag for retry."""
    # Make setup_tts fail on first call
    test_bot.setup_tts.side_effect = [Exception("TTS setup failed"), None]
    
    # First call should fail and reset completion flag
    with pytest.raises(Exception, match="TTS setup failed"):
        await test_bot.setup_hook()
    
    # Verify flag was reset on failure
    assert not test_bot._boot_completed
    
    # Second call should succeed
    await test_bot.setup_hook()
    
    # Verify setup completed successfully
    assert test_bot._boot_completed
    assert test_bot.setup_tts.call_count == 2


@pytest.mark.asyncio
async def test_concurrent_setup_calls(test_bot):
    """Test that concurrent setup_hook() calls are handled correctly."""
    # Create a slow async operation to simulate race conditions
    original_load_profiles = test_bot.load_profiles
    call_count = 0
    
    async def slow_load_profiles():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # Simulate slow operation
        return await original_load_profiles()
    
    test_bot.load_profiles = slow_load_profiles
    
    # Start multiple concurrent setup calls
    tasks = [
        asyncio.create_task(test_bot.setup_hook()),
        asyncio.create_task(test_bot.setup_hook()),
        asyncio.create_task(test_bot.setup_hook())
    ]
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    # Verify setup only ran once despite concurrent calls
    assert test_bot._boot_completed
    assert call_count == 1  # load_profiles should only be called once


@pytest.mark.asyncio
async def test_logging_output_single_trace(test_bot, caplog):
    """Test that setup produces exactly one setup trace."""
    import logging
    caplog.set_level(logging.DEBUG)  # Set to DEBUG to catch skip message
    
    # Run setup
    await test_bot.setup_hook()
    
    # Verify single setup start and completion messages
    setup_start_messages = [record for record in caplog.records 
                           if "Starting bot setup" in record.message]
    setup_complete_messages = [record for record in caplog.records 
                              if "Bot setup complete" in record.message]
    
    assert len(setup_start_messages) == 1
    assert len(setup_complete_messages) == 1
    
    # Run setup again
    caplog.clear()
    await test_bot.setup_hook()
    
    # Verify no additional setup messages (only skip message)
    setup_messages = [record for record in caplog.records 
                     if "Starting bot setup" in record.message or 
                        "Bot setup complete" in record.message]
    assert len(setup_messages) == 0
    
    # Should have skip message (at DEBUG level)
    skip_messages = [record for record in caplog.records 
                    if "Setup hook called but boot already completed" in record.message]
    assert len(skip_messages) == 1


@pytest.mark.asyncio
async def test_initialization_order_preserved(test_bot):
    """Test that initialization steps occur in the correct order."""
    call_order = []
    
    # Wrap each setup method to track call order
    original_methods = {
        'load_profiles': test_bot.load_profiles,
        'setup_background_tasks': test_bot.setup_background_tasks,
        'setup_tts': test_bot.setup_tts,
        'setup_router': test_bot.setup_router,
        'load_extensions': test_bot.load_extensions,
    }
    
    async def async_track_call(name, original):
        call_order.append(name)
        return await original()
    
    def sync_track_call(name, original):
        call_order.append(name)
        return original()
    
    # Replace methods with tracking versions
    test_bot.load_profiles = lambda: async_track_call('load_profiles', original_methods['load_profiles'])
    test_bot.setup_background_tasks = lambda: sync_track_call('setup_background_tasks', original_methods['setup_background_tasks'])
    test_bot.setup_tts = lambda: async_track_call('setup_tts', original_methods['setup_tts'])
    test_bot.setup_router = lambda: async_track_call('setup_router', original_methods['setup_router'])
    test_bot.load_extensions = lambda: async_track_call('load_extensions', original_methods['load_extensions'])
    
    # Run setup
    await test_bot.setup_hook()
    
    # Verify correct initialization order
    expected_order = [
        'load_profiles',
        'setup_background_tasks', 
        'setup_tts',
        'setup_router',
        'load_extensions'
    ]
    
    assert call_order == expected_order
