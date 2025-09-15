# Permissions

> **Fill me in**
> - [ ] Replace placeholder permissions integer and command mapping.
> - [ ] Link to actual Discord permissions calculator.

The bot requires a set of gateway permissions to operate. Use the [Discord Permissions Calculator](https://discordapi.com/permissions.html) to adjust as needed.

## Permissions Integer
- **Full**: `TODO_FULL_PERMISSIONS`
- **Minimal**: `TODO_MIN_PERMISSIONS`

## Per-command Requirements
| Command | Required Permissions |
| ------- | ------------------- |
| `/image` | Send Messages, Attach Files |
| `/imgedit` | Send Messages, Attach Files |
| `/video` | Send Messages, Attach Files |
| `!search` | Send Messages |
| `!rag` | Send Messages |
| `!tts` | Send Messages, Attach Files |
| `!memory` | Manage Messages |

Grant only the permissions your deployment needs. Reducing permissions lowers the risk surface.
