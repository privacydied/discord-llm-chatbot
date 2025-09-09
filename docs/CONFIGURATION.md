# Configuration

> **Fill me in**
> - [ ] Complete env var defaults and examples.
> - [ ] Confirm permissions integer and OAuth scopes.

## Environment Variables

| Name | Required? | Default | Example | Description | Security |
| ---- | --------- | ------- | ------- | ----------- | -------- |
| `DISCORD_TOKEN` | Yes | — | `A1B2...` | Discord bot token | Sensitive |
| `CLIENT_ID` | Yes | — | `1234567890` | Application client ID | Sensitive |
| `PUBLIC_KEY` | No | — | `abcdef` | Used for interactions endpoint | Sensitive |
| `TEXT_BACKEND` | No | `openai` | `openai` | Text model provider | Non-sensitive |
| `OPENAI_API_KEY` | Maybe | — | `sk-...` | API key for OpenAI/OpenRouter | Sensitive |
| `PROMPT_FILE` | Yes | — | `prompts/prompt.txt` | Path to system prompt file | Sensitive |
| `VL_PROMPT_FILE` | Yes | — | `prompts/vl-prompt.txt` | Vision prompt file | Sensitive |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` | Logging verbosity | Non-sensitive |
| `SHARD_COUNT` | No | `1` | `2` | Number of gateway shards | Non-sensitive |

Store sensitive variables in `.env` files or secret managers; avoid committing them to version control.

## Discord Values
- **Token**: `DISCORD_TOKEN`
- **Client ID**: `CLIENT_ID`
- **Public Key**: `PUBLIC_KEY`
- **Permissions Integer**: `TODO_PERMISSION_INT`
- **Scopes**: `bot`, `applications.commands`

## Privileged Intents
The bot may require privileged intents:

- **Message Content** – enables reading message bodies for prefix commands.
- **Guild Members** – required for member lookup and some admin features.
- **Presence** – optional; use to track user status.

Enable intents in the Developer Portal under **Bot → Privileged Gateway Intents**. If intents are disabled, related features will not function and commands may return errors.
