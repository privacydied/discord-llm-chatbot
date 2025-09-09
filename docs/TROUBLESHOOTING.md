# Troubleshooting

> **Fill me in**
> - [ ] Expand with project-specific errors.

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| Bot stays offline | Invalid token | Regenerate token and update `DISCORD_TOKEN` |
| `Privileged intent required` | Missing intent in portal | Enable Message Content/Guild Members in developer portal |
| Slash commands missing | Bot not invited with `applications.commands` | Re-invite using OAuth2 URL with proper scopes |
| `This command is on cooldown` | Command used too quickly | Wait for cooldown to expire |
| `429 Too Many Requests` | Rate limited by Discord | Exponential backoff or reduce request frequency |
| `GatewayTimeoutError` | Network issue | Check connectivity and retry |
| `SSL error` | TLS handshake failed | Ensure system clock is correct and CA certs installed |
| Docker container can't reach Discord | Host firewall/DNS issue | Verify network access and DNS resolution |
| STT/TTS errors | Missing system deps (ffmpeg, whisper) | Install required system packages |
| OCR fails | Tesseract not installed | Install `tesseract` and language packs |
| Slash commands not updating | Global sync delay | Wait up to an hour or sync per guild |
| `ModuleNotFoundError` | Missing Python dependency | Reinstall requirements |
| `PlaywrightError` | Browser not installed | `python -m playwright install chromium` |
| `PermissionError` writing logs | Insufficient file perms | Ensure bot has write access to `logs/` |
| `ValueError` parsing env | Malformed value in `.env` | Check formatting |
| `clock skew` warnings | System time out of sync | Sync with NTP |
| TLS handshake issues in Docker | Missing CA certificates | Install `ca-certificates` in image |
| `Unknown command` | Prefix mismatch | Check `BOT_PREFIX` setting |
| `Intents required` on startup | Guild Member intent disabled | Enable intents or disable features |
| `Vision provider unavailable` | Missing API key or provider down | Set `VISION_API_KEY` or try later |

## Collecting Logs
```bash
tail -f logs/bot.jsonl
```
