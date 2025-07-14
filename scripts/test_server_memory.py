#!/usr/bin/env python3
"""
Test script for server memory and history functionality.
"""
import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from main import (
    get_server_profile,
    save_server_profile,
    SERVER_PROFILE_DIR,
    default_server_profile
)

def test_server_profile_creation():
    """Test creating a new server profile."""
    test_guild_id = "test_guild_123"
    
    # Clean up any existing test data
    test_profile_path = SERVER_PROFILE_DIR / f"{test_guild_id}.json"
    if test_profile_path.exists():
        test_profile_path.unlink()
    
    # Test creating a new profile
    profile = get_server_profile(test_guild_id)
    assert profile is not None
    assert isinstance(profile, dict)
    assert "memories" in profile
    assert "history" in profile
    assert "total_messages" in profile
    assert "last_updated" in profile
    
    # Verify the file was created
    assert test_profile_path.exists()
    print("✅ Test 1: Server profile creation passed")
    
    # Clean up
    test_profile_path.unlink()

def test_memory_operations():
    """Test adding and retrieving memories."""
    test_guild_id = "test_guild_456"
    test_memory = "This is a test memory"
    
    # Clean up
    test_profile_path = SERVER_PROFILE_DIR / f"{test_guild_id}.json"
    if test_profile_path.exists():
        test_profile_path.unlink()
    
    # Add a memory
    profile = get_server_profile(test_guild_id)
    profile["memories"].append(test_memory)
    save_server_profile(test_guild_id)
    
    # Verify it was saved
    profile = get_server_profile(test_guild_id, force_reload=True)
    assert test_memory in profile["memories"]
    print("✅ Test 2: Memory operations passed")
    
    # Clean up
    test_profile_path.unlink()

def test_history_tracking():
    """Test message history tracking."""
    test_guild_id = "test_guild_789"
    test_message = "This is a test message"
    
    # Clean up
    test_profile_path = SERVER_PROFILE_DIR / f"{test_guild_id}.json"
    if test_profile_path.exists():
        test_profile_path.unlink()
    
    # Simulate processing a message
    profile = get_server_profile(test_guild_id)
    
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": "test_user_123",
        "username": "testuser",
        "content": test_message,
        "channel_id": "test_channel_123",
        "message_id": "test_msg_123"
    }
    
    profile["history"].append(history_entry)
    save_server_profile(test_guild_id)
    
    # Verify history was saved
    profile = get_server_profile(test_guild_id, force_reload=True)
    assert len(profile["history"]) == 1
    assert profile["history"][0]["content"] == test_message
    print("✅ Test 3: History tracking passed")
    
    # Clean up
    test_profile_path.unlink()

def test_memory_pruning():
    """Test that memories are pruned to the limit."""
    test_guild_id = "test_guild_prune"
    max_memories = 5  # Use a small number for testing
    
    # Set up environment
    os.environ["MAX_SERVER_MEMORY"] = str(max_memories)
    
    # Clean up
    test_profile_path = SERVER_PROFILE_DIR / f"{test_guild_id}.json"
    if test_profile_path.exists():
        test_profile_path.unlink()
    
    # Add more memories than the limit
    profile = get_server_profile(test_guild_id)
    for i in range(max_memories + 3):  # 3 over the limit
        memory = f"Test memory {i}"
        profile["memories"].append(memory)
    
    # Save and reload to trigger pruning
    save_server_profile(test_guild_id)
    profile = get_server_profile(test_guild_id, force_reload=True)
    
    # Verify pruning
    assert len(profile["memories"]) == max_memories
    assert "Test memory 0" not in profile["memories"]  # Oldest should be pruned
    assert f"Test memory {max_memories + 2}" in profile["memories"]  # Newest should be kept
    print("✅ Test 4: Memory pruning passed")
    
    # Clean up
    test_profile_path.unlink()
    del os.environ["MAX_SERVER_MEMORY"]

def test_corrupted_profile_recovery():
    """Test recovery from a corrupted profile file."""
    test_guild_id = "test_corrupted"
    test_profile_path = SERVER_PROFILE_DIR / f"{test_guild_id}.json"
    
    # Create a corrupted JSON file
    with open(test_profile_path, 'w') as f:
        f.write("This is not valid JSON")
    
    # Try to load it - should recover gracefully
    try:
        profile = get_server_profile(test_guild_id, force_reload=True)
        assert isinstance(profile, dict)
        assert "memories" in profile
        print("✅ Test 5: Corrupted profile recovery passed")
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
    finally:
        if test_profile_path.exists():
            test_profile_path.unlink()

if __name__ == "__main__":
    print("🚀 Starting server memory tests...\n")
    
    # Create test directory if it doesn't exist
    SERVER_PROFILE_DIR.mkdir(exist_ok=True)
    
    # Run tests
    test_server_profile_creation()
    test_memory_operations()
    test_history_tracking()
    test_memory_pruning()
    test_corrupted_profile_recovery()
    
    print("\n✨ All tests completed!")
