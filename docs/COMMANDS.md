# Commands

| Command | Description | Options | Example | Required Perms | Cooldown | Notes |
| ------- | ----------- | ------- | ------- | -------------- | -------- | ----- |
| `/image` | Generate an image | `prompt` | `/image prompt:cat` | Send Messages | 5s | Requires vision provider |
| `/imgedit` | Edit an image | `image`, `prompt` | `/imgedit image:1 prompt:hat` | Send Messages | 5s | |
| `/video` | Generate a video | `prompt` | `/video prompt:waves` | Send Messages | 10s | |
| `/vidref` | Animate an image into a video | `image` | `/vidref image:1` | Send Messages | 10s | |
| `!img <prompt>` | Generate images from text | `prompt` | `!img sunset` | Send Messages | 5s | alias `!image` |
| `!search <query>` | Search the web | `query` | `!search docs` | Send Messages | 3s | |
| `!ss <url>` | Screenshot a webpage | `url` | `!ss https://example.com` | Send Messages, Attach Files | 3s | Requires Playwright |
| `!watch <url>` | Transcribe video/audio | `url` | `!watch https://youtu.be/...` | Send Messages | 10s | aliases `!transcribe`,`!listen` |
| `!video-help` | Show video command help | – | `!video-help` | Send Messages | 0 | alias `!watch-help` |
| `!video-cache` | Show video cache info | – | `!video-cache` | Manage Guild | 0 | admin |
| `!tts <text>` | One-off TTS or toggle (`on`,`off`,`all`) | varies | `!tts hello` | Send Messages | 5s | group command |
| `!say <text>` | Echo text using TTS | `text` | `!say hi` | Send Messages | 5s | |
| `!speak [text]` | Next response TTS or speak text | `text` | `!speak hello` | Send Messages | 5s | |
| `!memory add/list/clear` | Manage personal memories | `content` | `!memory add likes cats` | Send Messages | 3s | |
| `!server-memory add/list/clear` | Manage server memories | `content` | `!server-memory list` | Manage Guild | 3s | admin |
| `!context_reset` | Clear conversation context | – | `!context_reset` | Send Messages | 3s | |
| `!context_stats` | Show context usage stats | – | `!context_stats` | Send Messages | 3s | |
| `!privacy_optout` | Disable memory collection | – | `!privacy_optout` | Send Messages | 0 | |
| `!privacy_optin` | Enable memory collection | – | `!privacy_optin` | Send Messages | 0 | |
| `!reload-config` | Reload configuration from disk | – | `!reload-config` | Manage Guild | 0 | admin |
| `!config-status` | Show current configuration | – | `!config-status` | Manage Guild | 0 | admin |
| `!config-help` | Show config help | – | `!config-help` | Manage Guild | 0 | admin |
| `!alert` | Compose admin alert in DMs | interactive | `!alert` | Admin | 0 | DM only |
| `!rag <subcommand>` | RAG management (`status`,`search`,`bootstrap`,...) | subcommand | `!rag status` | Admin | varies | admin |
| `!ping` | Check bot responsiveness | – | `!ping` | Send Messages | 0 | diagnostic |

Interaction components such as buttons or modals may be presented during command execution (screenshots TBD).

### Error Messages
- `This command is on cooldown.` – wait and retry.
- `I lack permissions to send messages.` – bot missing channel permissions.
- `Command not found` – slash commands may take time to propagate or bot lacks scope.
