"""
Tests for dynamic configuration reloading functionality.
"""
from unittest.mock import patch

import pytest

from bot.config_reload import (
    reload_env, get_current_config, get_config_version, get_config_for_debug,
    manual_reload_command, _generate_config_version, _redact_sensitive_values,
    _compare_configs, setup_config_reload
)


class TestConfigReload:
    """Test configuration reload functionality."""
    
    def test_generate_config_version(self):
        """Test configuration version generation."""
        config1 = {"KEY1": "value1", "KEY2": "value2"}
        config2 = {"KEY1": "value1", "KEY2": "value2"}
        config3 = {"KEY1": "value1", "KEY2": "different"}
        
        version1 = _generate_config_version(config1)
        version2 = _generate_config_version(config2)
        version3 = _generate_config_version(config3)
        
        # Same config should generate same version
        assert version1 == version2
        # Different config should generate different version
        assert version1 != version3
        # Version should be 12 characters (truncated SHA256)
        assert len(version1) == 12
    
    def test_redact_sensitive_values(self):
        """Test sensitive value redaction."""
        config = {
            "DISCORD_TOKEN": "abc123456789",
            "OPENAI_API_KEY": "sk-1234567890abcdef",
            "NORMAL_SETTING": "public_value",
            "SECRET_KEY": "secret123",
            "PASSWORD": "mypassword",
            "EMPTY_TOKEN": "",
            "NONE_VALUE": None
        }
        
        redacted = _redact_sensitive_values(config)
        
        # Sensitive values should be redacted
        assert redacted["DISCORD_TOKEN"] == "***6789"
        assert redacted["OPENAI_API_KEY"] == "***cdef"
        assert redacted["SECRET_KEY"] == "***123"
        assert redacted["PASSWORD"] == "***word"
        
        # Normal values should be unchanged
        assert redacted["NORMAL_SETTING"] == "public_value"
        
        # Empty/None values should be preserved
        assert redacted["EMPTY_TOKEN"] == ""
        assert redacted["NONE_VALUE"] is None
    
    def test_compare_configs(self):
        """Test configuration comparison."""
        old_config = {
            "KEY1": "old_value",
            "KEY2": "same_value",
            "KEY3": "to_be_removed"
        }
        
        new_config = {
            "KEY1": "new_value",
            "KEY2": "same_value",
            "KEY4": "added_value"
        }
        
        changes = _compare_configs(old_config, new_config)
        
        assert changes["added"] == {"KEY4": "added_value"}
        assert changes["removed"] == {"KEY3": "to_be_removed"}
        assert changes["modified"] == {
            "KEY1": {"old": "old_value", "new": "new_value"}
        }
        assert changes["unchanged_count"] == 1  # KEY2
    
    @patch('bot.config_reload.load_config')
    @patch('bot.config_reload._env_file_path')
    def test_reload_env_success(self, mock_env_path, mock_load_config):
        """Test successful environment reload."""
        # Mock file existence
        mock_env_path.exists.return_value = True
        
        # Mock config loading
        old_config = {"KEY1": "old", "KEY2": "same"}
        new_config = {"KEY1": "new", "KEY2": "same", "KEY3": "added"}
        
        # Set initial state
        with patch('bot.config_reload._current_config', old_config):
            mock_load_config.return_value = new_config
            
            result = reload_env()
            
            assert result["success"] is True
            assert "changes" in result
            assert "old_version" in result
            assert "new_version" in result
            assert result["old_version"] != result["new_version"]
    
    @patch('bot.config_reload.load_config')
    def test_reload_env_missing_required_vars(self, mock_load_config):
        """Test reload failure when required variables are missing."""
        # Mock config without required DISCORD_TOKEN
        mock_load_config.return_value = {"OTHER_KEY": "value"}
        
        result = reload_env()
        
        assert result["success"] is False
        assert "Missing required variables" in result["error"]
    
    def test_manual_reload_command_format(self):
        """Test manual reload command output formatting."""
        with patch('bot.config_reload.reload_env') as mock_reload:
            # Test successful reload with changes
            mock_reload.return_value = {
                "success": True,
                "changes": {
                    "added": {"KEY1": "value"},
                    "removed": {"KEY2": "old"},
                    "modified": {"KEY3": {"old": "old", "new": "new"}},
                    "unchanged_count": 5
                },
                "old_version": "abc123",
                "new_version": "def456"
            }
            
            result = manual_reload_command()
            
            assert "✅ Configuration reloaded successfully!" in result
            assert "+1 added, -1 removed, ~1 modified" in result
            assert "abc123 → def456" in result
    
    def test_manual_reload_command_no_changes(self):
        """Test manual reload command with no changes."""
        with patch('bot.config_reload.reload_env') as mock_reload:
            mock_reload.return_value = {
                "success": True,
                "changes": {
                    "added": {},
                    "removed": {},
                    "modified": {},
                    "unchanged_count": 10
                },
                "new_version": "abc123"
            }
            
            result = manual_reload_command()
            
            assert "✅ Configuration reloaded (no changes detected)" in result
            assert "abc123" in result
    
    def test_manual_reload_command_failure(self):
        """Test manual reload command failure handling."""
        with patch('bot.config_reload.reload_env') as mock_reload:
            mock_reload.return_value = {
                "success": False,
                "error": "Test error message"
            }
            
            result = manual_reload_command()
            
            assert "❌ Configuration reload failed" in result
            assert "Test error message" in result


class TestConfigReloadIntegration:
    """Integration tests for config reload system."""
    
    def test_setup_config_reload(self):
        """Test config reload system setup."""
        with patch('bot.config_reload.load_config') as mock_load_config:
            mock_load_config.return_value = {"TEST_KEY": "test_value"}
            
            setup_config_reload()
            
            # Should have initialized current config and version
            config = get_current_config()
            version = get_config_version()
            
            assert config["TEST_KEY"] == "test_value"
            assert len(version) == 12
    
    def test_get_config_for_debug(self):
        """Test debug config retrieval with redaction."""
        test_config = {
            "DISCORD_TOKEN": "secret123",
            "PUBLIC_SETTING": "public_value"
        }
        
        with patch('bot.config_reload._current_config', test_config):
            debug_config = get_config_for_debug()
            
            assert debug_config["DISCORD_TOKEN"] == "***123"
            assert debug_config["PUBLIC_SETTING"] == "public_value"


if __name__ == "__main__":
    pytest.main([__file__])
