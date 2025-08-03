# Manual Test Plan: Mention Deduplication & Dynamic Config Reload

## Overview

This document provides step-by-step manual testing procedures to verify both the mention deduplication and dynamic configuration reload features work correctly.

## Prerequisites

1. Bot is running and connected to Discord
2. You have administrator permissions in the test server
3. Access to the bot's `.env` file and process (for config reload tests)
4. Terminal access to send signals (Unix systems)

## Test Suite 1: Mention Deduplication

### Test 1.1: Basic Mention Deduplication
**Objective**: Verify duplicate mentions are removed from bot replies

**Steps**:
1. Send a message to the bot in a guild channel: `Hello bot`
2. Observe the bot's reply format
3. Check that your username appears only once at the beginning

**Expected Result**: 
- Reply format: `@YourUsername [bot response]`
- No duplicate mentions like `@YourUsername @YourUsername [response]`

### Test 1.2: AI Model Already Including Mentions
**Objective**: Test when AI model response already contains mentions

**Steps**:
1. Send a message that might prompt the AI to mention you: `Can you mention me in your response?`
2. Observe the bot's reply
3. Verify only one mention appears at the start

**Expected Result**:
- Single mention at the beginning, even if AI tried to mention you in the response text
- Clean formatting without duplicate mentions

### Test 1.3: DM Channel Behavior
**Objective**: Verify mentions are not added in DM channels

**Steps**:
1. Send a direct message to the bot: `Hello in DM`
2. Observe the bot's reply format

**Expected Result**:
- No mention prefix in DM responses (since mentions aren't needed in DMs)
- Normal response without `@Username` prefix

### Test 1.4: Multiple Users in Thread/Channel
**Objective**: Verify mention deduplication works with multiple users

**Steps**:
1. Have multiple users interact with the bot in the same channel
2. Observe each user's replies
3. Verify each user gets mentioned exactly once in their respective replies

**Expected Result**:
- Each user mentioned exactly once in their own replies
- No cross-contamination of mentions between users

## Test Suite 2: Dynamic Configuration Reload

### Test 2.1: Manual Discord Command Reload
**Objective**: Test the `!reload-config` Discord command

**Steps**:
1. Edit the `.env` file to change a non-critical setting:
   ```bash
   # Change this value
   MAX_USER_MEMORY=25
   ```
2. In Discord, run: `!reload-config`
3. Observe the bot's response

**Expected Result**:
```
✅ Configuration reloaded successfully!
📊 Changes: ~1 modified
🔖 Version: [old_hash] → [new_hash]
```

### Test 2.2: File Watcher Automatic Reload
**Objective**: Test automatic reload when .env file is modified

**Steps**:
1. Monitor bot logs: `tail -f logs/bot.jsonl | grep -i config`
2. Edit `.env` file and save:
   ```bash
   # Add or modify a setting
   DEBUG=true
   ```
3. Wait up to 10 seconds
4. Check logs for automatic reload messages

**Expected Result**:
- Log message: `📁 .env file modification detected, triggering reload...`
- Followed by: `✅ File watcher configuration reload completed`
- Configuration changes logged with before/after values

### Test 2.3: SIGHUP Signal Reload (Unix Only)
**Objective**: Test SIGHUP signal handling for config reload

**Steps**:
1. Find the bot process ID: `ps aux | grep python.*bot`
2. Edit `.env` file to change a setting
3. Send SIGHUP signal: `kill -HUP [bot_pid]`
4. Check bot logs for reload confirmation

**Expected Result**:
- Log message: `📡 Received SIGHUP signal, triggering configuration reload...`
- Followed by: `✅ SIGHUP configuration reload completed successfully`
- Configuration changes logged

### Test 2.4: Configuration Status Command
**Objective**: Test the `!config-status` command

**Steps**:
1. In Discord, run: `!config-status`
2. Observe the embed response

**Expected Result**:
- Embed showing current configuration version
- Total number of settings
- Key non-sensitive settings displayed
- No sensitive values (tokens, keys) visible

### Test 2.5: Configuration Help Command
**Objective**: Test the `!config-help` command

**Steps**:
1. In Discord, run: `!config-help`
2. Review the help information

**Expected Result**:
- Comprehensive help embed explaining all config commands
- Information about automatic reload mechanisms
- Clear indication of admin-only commands

### Test 2.6: Error Handling - Missing Required Variables
**Objective**: Test graceful handling of invalid configuration

**Steps**:
1. Backup your current `.env` file
2. Edit `.env` to remove `DISCORD_TOKEN`
3. Try to reload: `!reload-config`
4. Restore the original `.env` file

**Expected Result**:
- Error message: `❌ Configuration reload failed: Missing required variables: ['DISCORD_TOKEN']`
- Bot continues operating with previous configuration
- No crash or service interruption

### Test 2.7: Sensitive Value Redaction
**Objective**: Verify sensitive values are redacted in logs

**Steps**:
1. Change a sensitive value in `.env` (e.g., add characters to `DISCORD_TOKEN`)
2. Trigger a reload via any method
3. Check logs for the change notification

**Expected Result**:
- Log shows: `🔄 Modified: DISCORD_TOKEN = [REDACTED]`
- Actual token values not visible in logs
- Other non-sensitive changes show actual values

## Test Suite 3: Integration & Regression Tests

### Test 3.1: Bot Startup with New Systems
**Objective**: Verify bot starts correctly with new systems enabled

**Steps**:
1. Stop the bot
2. Start the bot and monitor startup logs
3. Verify all systems initialize correctly

**Expected Result**:
- Log: `🔧 Configuration reload system initialized [version: ...]`
- Log: `📡 SIGHUP signal handler installed` (Unix)
- Log: `👁️ File watcher task started`
- Log: `✅ ConfigCommands cog loaded`
- No startup errors

### Test 3.2: Existing Functionality Preservation
**Objective**: Verify existing bot functionality still works

**Steps**:
1. Test basic chat functionality
2. Test TTS commands (if enabled)
3. Test memory commands
4. Test context commands
5. Verify all existing features work as before

**Expected Result**:
- All existing functionality works unchanged
- No regressions in posting behavior
- All commands respond correctly

### Test 3.3: Performance Impact Assessment
**Objective**: Verify new systems don't significantly impact performance

**Steps**:
1. Monitor bot response times before and after implementation
2. Check memory usage with `!config-status` or system monitoring
3. Observe file watcher resource usage

**Expected Result**:
- No noticeable increase in response times
- Minimal memory overhead from new systems
- File watcher uses minimal CPU (should be nearly zero when idle)

## Test Suite 4: Edge Cases & Stress Tests

### Test 4.1: Rapid Configuration Changes
**Objective**: Test debouncing and rapid reload handling

**Steps**:
1. Rapidly edit and save `.env` file multiple times (within 2 seconds)
2. Monitor logs for reload behavior

**Expected Result**:
- Only one reload occurs due to debouncing
- Log: `🔄 File change detected but debounced (last reload X.Xs ago)`
- No excessive reloading or resource usage

### Test 4.2: Large Configuration Changes
**Objective**: Test handling of many simultaneous changes

**Steps**:
1. Make multiple changes to `.env` at once:
   ```bash
   MAX_USER_MEMORY=100
   MAX_SERVER_MEMORY=500
   DEBUG=true
   LOG_LEVEL=DEBUG
   TEMPERATURE=0.8
   ```
2. Trigger reload and observe logging

**Expected Result**:
- All changes detected and logged individually
- Summary shows correct count: `📈 Total: +X -Y ~Z =N`
- No errors or missed changes

### Test 4.3: Permission Edge Cases
**Objective**: Test command permissions and error handling

**Steps**:
1. Have a non-admin user try `!reload-config`
2. Have a non-admin user try `!config-status`
3. Anyone can use `!config-help`

**Expected Result**:
- Non-admin users get permission denied messages
- Help command works for everyone
- Clear error messages, no crashes

## Success Criteria

### Mention Deduplication
- ✅ No duplicate user mentions in any bot replies
- ✅ Single mention appears at the start of guild messages
- ✅ No mentions in DM messages
- ✅ Existing posting behavior preserved
- ✅ All unit tests pass

### Dynamic Configuration Reload
- ✅ Manual `!reload-config` command works
- ✅ File watcher detects and reloads changes
- ✅ SIGHUP signal triggers reload (Unix)
- ✅ Configuration changes logged with before/after values
- ✅ Sensitive values redacted in logs
- ✅ Error handling for invalid configurations
- ✅ Status and help commands work correctly
- ✅ No service interruption during reloads

### Integration
- ✅ Bot starts successfully with new systems
- ✅ No regressions in existing functionality
- ✅ Minimal performance impact
- ✅ All edge cases handled gracefully

## Troubleshooting

### Common Issues

1. **File watcher not working**: Check file permissions and path
2. **SIGHUP not working**: Verify Unix system and proper signal handling
3. **Permission errors**: Ensure admin role for config commands
4. **Reload failures**: Check for required variables and valid syntax

### Debug Commands

- Check logs: `tail -f logs/bot.jsonl | grep -E "(config|mention|reload)"`
- Monitor file changes: `watch -n 1 stat .env`
- Test signal handling: `kill -HUP $(pgrep -f python.*bot)`
