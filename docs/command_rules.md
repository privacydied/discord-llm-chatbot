# Command Rules and Message Handling

## 1 IN > 1 OUT Principle
Every message event must produce exactly one response (string) or one error embed. This ensures:
- No silent failures (violations raise AssertionError)
- Clear audit trail (logged with trace ID)
- Predictable behavior (strict type checking)
- Non-empty responses (empty strings prohibited)

## Guild Message Handling
In guild channels:
- Bot must be mentioned (`@BotName`)
- First token after mention must start with `!` to be recognized as a command
- Example: `@BotName !ping` → command handler
- Example: `@BotName How are you?` → TEXT→TEXT flow

## DM Message Handling
In direct messages:
- `!` prefix is required for commands
- Mention is optional but allowed
- Example: `!ping` → command handler
- Example: `Hello` → TEXT→TEXT flow

## Mention/Prefix Matrix
| Context      | Format              | Result          | Notes                          |
|-------------|---------------------|-----------------|-------------------------------|
| Guild       | `@BotName !command` | Command handler | Must be mentioned with ! prefix |
| Guild       | `@BotName text`     | TEXT→TEXT flow  | Mention without ! prefix        |
| DM          | `!command`          | Command handler | ! prefix required for commands |
| DM          | `text`              | TEXT→TEXT flow  | Plain text in DMs              |

## Logging Behavior
All messages trigger logging with:
- Guild ID (or 'dm' for direct messages)
- Author ID
- Raw content
- Flow start/end timestamps
- Error embeds for any exceptions
- Response type validation (enforced via assert)
- Response content validation (non-empty check)

## Command Filtering
- Guild messages without mention are ignored early
- DM messages without `!` prefix are ignored
- This prevents CommandNotFound errors and spam