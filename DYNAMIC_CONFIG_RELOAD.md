# Dynamic Configuration Reload System

## Overview

The bot now supports hot-reloading of environment variables from the `.env` file without requiring a restart. This addresses the documentation claim of "hot-reload" behavior that was previously only available for prompt files.

## Features

### 1. Multiple Reload Mechanisms

- **SIGHUP Signal Handler**: Send `kill -HUP <pid>` to reload configuration (Unix systems)
- **File Watcher**: Automatically detects `.env` file modifications and reloads
- **Manual Command**: Use `!reload-config` Discord command for on-demand reloading

### 2. Configuration Management (`bot/config_reload.py`)

Core functions:
- **`reload_env()`**: Main reload function with validation and change detection
- **`setup_config_reload()`**: Initialize the reload system with signal handlers
- **`start_file_watcher()`**: Start background file monitoring task
- **`manual_reload_command()`**: Discord command interface

### 3. Change Tracking & Logging

The system provides detailed logging of configuration changes:

```
📊 Configuration changes detected:
  ➕ Added: {'NEW_SETTING': 'value'}
  ➖ Removed: {'OLD_SETTING': 'removed_value'}
  🔄 Modified: MAX_USER_MEMORY = 20 → 30
  📈 Total: +1 -1 ~1 =45
✅ Configuration reload complete [version: abc123 → def456]
```

### 4. Security Features

- **Sensitive Value Redaction**: API keys, tokens, and passwords are redacted in logs
- **Version Tracking**: Each configuration gets a unique hash-based version identifier
- **Validation**: Required variables (like `DISCORD_TOKEN`) are validated after reload
- **Error Handling**: Graceful fallback to previous configuration on errors

## Discord Commands

### `!reload-config` (Admin Only)
Manually trigger configuration reload and show results:
```
✅ Configuration reloaded successfully!
📊 Changes: +1 added, ~2 modified
🔖 Version: abc123 → def456
```

### `!config-status` (Admin Only)
Show current configuration status and key settings:
- Configuration version
- Total number of settings
- Key non-sensitive settings display

### `!config-help`
Display help information about configuration commands and hot-reload features.

## Integration

### Bot Initialization
The system is integrated into the main bot startup sequence:

```python
# Set up dynamic configuration reload system
setup_config_reload()

# Start configuration file watcher
await start_file_watcher()
```

### Signal Handling
SIGHUP signal handler is automatically installed on Unix systems:
```bash
# Reload bot configuration without restart
kill -HUP $(pgrep -f "python.*bot")
```

## Configuration Versioning

Each configuration state gets a unique 12-character hash-based version:
- Generated from non-sensitive configuration values
- Used for tracking changes and debugging
- Displayed in status commands and logs

## File Watching

The file watcher monitors `.env` for changes with:
- **5-second polling interval**: Balance between responsiveness and resource usage
- **2-second debounce**: Prevents rapid successive reloads
- **Error resilience**: Continues monitoring even after temporary errors

## Supported Settings

Most settings take effect immediately upon reload:
- Model settings (`TEXT_MODEL`, `OPENAI_TEXT_MODEL`, etc.)
- Memory limits (`MAX_USER_MEMORY`, `MAX_SERVER_MEMORY`, etc.)
- TTS/STT engine settings
- Log levels and debug flags
- API endpoints and configurations

## Error Handling

The system includes comprehensive error handling:
- **Missing Required Variables**: Prevents reload if critical settings are missing
- **Malformed .env Files**: Logs errors and maintains previous configuration
- **Callback Failures**: Individual callback errors don't prevent overall reload
- **File System Errors**: File watcher continues operating despite temporary issues

## Testing

Comprehensive test coverage in `tests/test_config_reload.py`:
- Configuration version generation
- Sensitive value redaction
- Configuration comparison and change detection
- Reload success and failure scenarios
- Manual command output formatting

## Usage Examples

### 1. Change Model Settings
Edit `.env`:
```bash
OPENAI_TEXT_MODEL=gpt-4-turbo
MAX_USER_MEMORY=50
```

The bot automatically detects the change and logs:
```
🔄 Modified: OPENAI_TEXT_MODEL = gpt-4 → gpt-4-turbo
🔄 Modified: MAX_USER_MEMORY = 20 → 50
```

### 2. Manual Reload via Discord
```
User: !reload-config
Bot: ✅ Configuration reloaded successfully!
     📊 Changes: ~2 modified
     🔖 Version: abc123 → def456
```

### 3. Signal-Based Reload
```bash
# Find bot process
ps aux | grep python.*bot

# Send SIGHUP signal
kill -HUP 12345

# Check logs for reload confirmation
tail -f logs/bot.jsonl | grep "Configuration reload"
```

## Benefits

1. **Zero Downtime**: Configuration changes without bot restart
2. **Immediate Effect**: Most settings apply instantly
3. **Change Visibility**: Clear logging of what changed
4. **Multiple Interfaces**: Signal, file watcher, and manual command options
5. **Security Conscious**: Sensitive values are properly redacted
6. **Robust**: Comprehensive error handling and validation
7. **Debuggable**: Version tracking and detailed status information
