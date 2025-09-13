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
| `VIDEO_MAX_DURATION` | No | `600` | `900` | Max video length in seconds | Non-sensitive |
| `VIDEO_MAX_CONCURRENT` | No | `3` | `5` | Max concurrent URL downloads | Non-sensitive |
| `VIDEO_SPEEDUP` | No | `1.5` | `1.25` | Audio atempo applied during processing | Non-sensitive |
| `VIDEO_CACHE_DIR` | No | `cache/video_audio` | `/mnt/cache/video_audio` | Cache directory for processed audio | Non-sensitive |
| `VIDEO_CACHE_EXPIRY_DAYS` | No | `7` | `3` | Days before cached entries expire | Non-sensitive |
| `VIDEO_COOKIES_FROM_BROWSER` | No | — | `firefox:default-release` | Pass browser cookies to yt-dlp (preferred). Enables access to TikTok age/region gated posts. | Sensitive |
| `VIDEO_COOKIES_FILE` | No | — | `/path/to/cookies.txt` | Netscape cookies file for yt-dlp. Alternative to `VIDEO_COOKIES_FROM_BROWSER`. | Sensitive |
| `VIDEO_COOKIES_SITES` | No | `tiktok` | `tiktok,youtube` | Apply cookies only to these sources. Case-insensitive. | Non-sensitive |

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

## Video ingestion requirements

To transcribe audio from URLs, the bot invokes external tools:

- `yt-dlp` for fetching media metadata and extracting audio.
- `ffmpeg` for audio conversion and normalization.

On Arch Linux (recommended), install via pacman:

```bash
sudo pacman -S yt-dlp ffmpeg
```

If you encounter TikTok messages like:

> This post may not be comfortable for some audiences. Log in for access.

configure authentication for yt-dlp using one of the following:

- Browser cookies (preferred):

```bash
export VIDEO_COOKIES_FROM_BROWSER="firefox:default-release"  # or "chromium:Default"
export VIDEO_COOKIES_SITES="tiktok"
```

- Cookies file (Netscape format):

```bash
export VIDEO_COOKIES_FILE="/path/to/cookies.txt"
export VIDEO_COOKIES_SITES="tiktok"
```

Only the sites listed in `VIDEO_COOKIES_SITES` will receive cookies (default is `tiktok`).
