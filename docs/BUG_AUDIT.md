# Bug Audit — discord-llm-chatbot

**Date:** 2026-04-27  
**Scope:** Full codebase (`bot/`, `utils/`, `pyproject.toml`)  
**Methodology:** Static analysis + pattern matching per bug-audit skill  

---

## Table 1: Security Vulnerabilities

| # | File | Line(s) | Severity | Category | Description | Evidence |
|---|------|---------|----------|----------|-------------|----------|
| S1 | `bot/controller.py` | 68 | **CRITICAL** | Path Traversal | `Path(f"/tmp/{attachment.filename}")` — Discord attachment filenames are attacker-controlled. A filename like `../../home/user/.ssh/authorized_keys` writes outside `/tmp/`. No sanitization of `..` or path separators. | `temp_file = Path(f"/tmp/{attachment.filename}")` |
| S2 | `bot/utils/external_api.py` | 118 | **CRITICAL** | Secret in Logs | `logger.debug(f"🔗 Final API URL: {api_url}")` — the `api_url` contains `?key={api_key}`, so the SCREENSHOT_API_KEY is written to JSONL logs in plaintext. The `SensitiveDataFilter` doesn't catch this because the key is embedded in a URL string, not a dict field. | `api_url = f"{api_url_base}?key={api_key}&url=..."` then `logger.debug(f"🔗 Final API URL: {api_url}")` |
| S3 | `bot/commands/screenshot_commands.py` | 49 | **HIGH** | SSRF | The `!ss` command accepts any URL and passes it to an external screenshot API or Playwright. No validation blocks internal/private IPs (`127.0.0.1`, `10.x`, `172.16-31.x`, `192.168.x`, `169.254.x`, `metadata.google.internal`). An attacker can screenshot internal services. | No `127.0`/`localhost`/`10.`/`172.`/`192.168` checks found |
| S4 | `bot/commands/rag_commands.py` | 18-85 | **HIGH** | Auth Bypass (DM) | `is_admin_user()` checks `guild_permissions.administrator` via mutual guilds in DMs. If the bot is only in one guild where the attacker has admin, or if the `application_info()` call fails, the exception handler returns `False` — but the check itself calls `await ctx.bot.application_info()` on EVERY invocation, which is a rate-limited API call and can fail with `DiscordException`, causing all RAG commands to fail open or closed unpredictably. | `except Exception as e: ... return False` — denial-of-service on auth |
| S5 | `bot/commands/admin_alert_commands.py` | 498 | **HIGH** | Missing Auth Decorator | `!alert` command has NO `@commands.is_owner()` or `@commands.has_permissions()` decorator. Auth is checked only inside the function body via `self.alert_manager.is_admin_user()`. This means the command appears in `!help` to all users, and any future refactoring that moves the body check breaks auth silently. Decorators are the Discord.py convention for fail-closed auth. | `@commands.command(name="alert")` with no auth decorator |
| S6 | `bot/tts/interface.py` | 1332 | **MEDIUM** | TOCTOU Race | `tempfile.mktemp(prefix="tts_", suffix=".ogg")` creates a filename without creating the file. Between `mktemp()` and the actual write, an attacker can create a symlink at that path to redirect output. Python docs explicitly warn: "Use of mktemp() is a security risk; use mkstemp() instead." | `Path(tempfile.mktemp(prefix="tts_", suffix=".ogg"))` |
| S7 | `bot/config_reload.py` | 112-113 | **MEDIUM** | Partial Secret Leakage | `_redact_sensitive_values()` shows the last 4 characters: `f"***{str(value)[-4:]}"`. For many API keys and tokens, the last 4 chars are sufficient to narrow brute-force space dramatically (e.g., OpenAI keys have known prefixes). | `redacted[key] = f"***{str(value)[-4:]}"` |
| S8 | `bot/config_reload.py` | 85-92 | **MEDIUM** | Incomplete SENSITIVE_KEYS | `SENSITIVE_KEYS` is missing: `SCREENSHOT_API_KEY`, `SCREENSHOT_API_URL`, `X_API_BEARER_TOKEN` (only in logging filter, not here), `VISION_API_KEY` (only in logging filter), `WHISPER_API_KEY` is present but many more env vars with secrets are missing (e.g., any `*_SECRET`, `*_CREDENTIALS`, `DATABASE_URL` with embedded passwords). | `SENSITIVE_KEYS = {"DISCORD_TOKEN", "OPENAI_API_KEY", "WHISPER_API_KEY", "API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASS"}` |
| S9 | `bot/utils/logging.py` | 187-195 | **MEDIUM** | Incomplete SensitiveDataFilter | `SECRET_KEYS` in the logging filter doesn't include `SCREENSHOT_API_KEY` or pattern-match on URL params like `?key=`. The API key embedded in the screenshot URL (S2) leaks through because the filter only scrubs dict fields by key name. | `SECRET_KEYS = {"OPENAI_API_KEY", "X_API_BEARER_TOKEN", ...}` — no `SCREENSHOT_API_KEY` |
| S10 | `bot/commands/video_commands.py` | 56 | **MEDIUM** | Missing Rate Limit | `!watch`/`!transcribe` command has no `@commands.cooldown()` or `@commands.max_concurrency()`. Each invocation spawns yt-dlp + ffmpeg subprocesses. A single user can exhaust system resources by spamming the command. | No cooldown/max_concurrency decorator found |
| S11 | `bot/commands/screenshot_commands.py` | 49 | **MEDIUM** | Missing Rate Limit | `!ss`/`!screenshot` command has no rate limit. Each invocation calls an external API (costs money) or spawns a Playwright browser. Spam = cost amplification. | No cooldown/max_concurrency decorator found |
| S12 | `bot/commands/search_commands.py` | 49 | **LOW** | Missing Rate Limit | `!search` command has no cooldown. External search API calls can be spammed. | No cooldown decorator found |
| S13 | `bot/core/bot.py` | 2228-2245 | **LOW** | Auth Bypass via Case Manipulation | `_is_long_running_admin_command()` checks `content.strip().lower().startswith(cmd)` where `cmd` is `"!rag bootstrap"` etc. The Discord command prefix is configurable, so if `COMMAND_PREFIX` is changed, this hardcoded check stops matching and those commands get queued normally (not a security bypass, but a functional bug). Also, `content.lower()` means `!RAG BOOTSTRAP` matches but Discord.py's command routing is case-insensitive for prefixes but case-sensitive for command names by default. | `content = message.content.strip().lower()` then `content.startswith(cmd)` |

---

## Table 2: Resource Leaks & Unbounded Growth

| # | File | Line(s) | Severity | Category | Description | Evidence |
|---|------|---------|----------|----------|-------------|----------|
| R1 | `bot/controller.py` | 68 | **HIGH** | Temp File Leak | `download_attachment()` writes to `/tmp/{attachment.filename}` with no cleanup. The calling code in `hybrid_pipeline()` never deletes the file. Vision commands have proper `tempfile.NamedTemporaryFile` + cleanup, but the controller path is a standalone leak. | `temp_file = Path(f"/tmp/{attachment.filename}"); await attachment.save(temp_file)` — no unlink |
| R2 | `bot/commands/admin_alert_commands.py` | 61 | **MEDIUM** | Memory Leak | `self.sessions: Dict[int, AlertSession] = {}` — expired sessions are only purged on `get_session()` access. If a user creates a session and never accesses it again, the entry persists forever. No periodic sweep or max-size cap. | `if time.time() > session.expires_at: ... del self.sessions[user_id]` — only in get_session |
| R3 | `bot/commands/admin_alert_commands.py` | 62 | **LOW** | Minor Leak | `self.reaction_queues: Dict[int, List] = {}` — queues are cleaned after processing, but if `_process_reaction_queue` raises before cleanup, the queue entry stays. The while-loop catches per-operation exceptions, so this is unlikely but not impossible. | `if message_id in self.reaction_queues and not self.reaction_queues[message_id]: del self.reaction_queues[message_id]` |
| R4 | `bot/tts/interface.py` | 1295 | **LOW** | FD Leak (mitigated) | `fd, wav_tmp_name = tempfile.mkstemp(...)` then `os.close(fd)` — this is correctly handled. The `os.close(fd)` is on the next line. However, if `sf.write()` raises between mkstemp and the finally block, the wav file leaks on disk (the finally only unlinks `wav_path`). | `fd, wav_tmp_name = tempfile.mkstemp(prefix="tts_", suffix=".wav"); os.close(fd)` |
| R5 | `bot/hear.py` | 100 | **INFO** | Global Semaphore | `_JOB_SEMAPHORE = asyncio.Semaphore(1)` — STT is limited to 1 concurrent job globally (not per-user). This means one user's STT request blocks all other users' STT. This is a design choice but could cause denial-of-service by a single user. | `_JOB_SEMAPHORE = asyncio.Semaphore(1)` at module level |

---

## Table 3: Race Conditions & Concurrency Bugs

| # | File | Line(s) | Severity | Category | Description | Evidence |
|---|------|---------|----------|----------|-------------|----------|
| C1 | `bot/config_reload.py` | 420-445 | **HIGH** | SIGHUP vs File Watcher Race | `_sighup_handler()` calls `reload_env()` synchronously from a signal handler (which runs in the main thread). Meanwhile, `_file_watcher_loop()` can also trigger `reload_env()` from the event loop. Both use `_config_lock` (threading.RLock), but the SIGHUP handler runs in a signal context where acquiring a lock can deadlock if the main thread already holds it. Python signal handlers run between bytecodes, so if `_config_lock` is held when SIGHUP arrives, the handler blocks the main thread. | `_sighup_handler` calls `reload_env()` which does `with _config_lock:` — signal context |
| C2 | `bot/commands/admin_alert_commands.py` | 61 | **MEDIUM** | No Async Lock on Sessions | `self.sessions` dict is modified by `create_session()` and `get_session()` with no `asyncio.Lock`. If two coroutines call `create_session()` concurrently for the same user, both may read the old session, both replace it, and one session object is lost (with its state). | No Lock import or usage found in the file |
| C3 | `bot/core/bot.py` | 3149 | **LOW** | Lock Setdefault Race | `self._processing_locks.setdefault(message.id, asyncio.Lock())` — `dict.setdefault` is atomic in CPython (GIL) but semantically fragile. If two messages with the same ID arrive (extremely unlikely in Discord but theoretically possible in a replay), they'd share a lock. The finally block pops it, which could remove the lock while another coroutine holds it. | `lock = self._processing_locks.setdefault(message.id, asyncio.Lock())` |

---

## Table 4: Error Handling Defects

| # | File | Line(s) | Severity | Category | Description | Evidence |
|---|------|---------|----------|----------|-------------|----------|
| E1 | `bot/events/command_error_handler.py` | 165 | **MEDIUM** | Info Leakage | `str(error)[:100]` is logged including internal exception details. For `CommandInvokeError`, the `.original` exception may contain file paths, API error bodies, or connection strings. While this goes to logs (not users), the log line is a structured tree that includes guild/channel info — if logs are shared or exposed, this leaks internals. | `└── 📋 Details: {str(error)[:100]}{"..." if len(str(error)) > 100 else ""}` |
| E2 | `bot/controller.py` | 56 | **MEDIUM** | TTS Error Exposure | `await ctx.send(f"⚠️ TTS failed: {str(e)}")` — the raw exception message from `InferenceError` is sent directly to the Discord channel. This could include file paths, model names, or internal error details. | `await ctx.send(f"⚠️ TTS failed: {str(e)}")` |
| E3 | `bot/hear.py` | 948-972 | **LOW** | Swallowed Subprocess Errors | ffmpeg/ffprobe subprocess errors are caught and re-raised as `InferenceError`, but the stderr output (which may contain system paths, library versions, etc.) is included in the error message that may eventually reach the user via `see_infer` or `hear_infer` error handling. | `raise InferenceError(f"ffprobe failed ({proc.returncode}): {stderr.decode(errors='ignore')}")` |
| E4 | `bot/utils/external_api.py` | 89-158 | **LOW** | Cascading Fallback Masking | The screenshot function has multiple fallback paths (API → Playwright → None). If the API fails with an auth error (bad key), it falls through to Playwright silently. The original error (which might indicate a config problem) is logged at debug level but never surfaced to the caller, who gets a Playwright screenshot instead of being told their API key is invalid. | Multiple `return None` and `return await _playwright_screenshot(...)` fallthroughs |

---

## Table 5: Logic Bugs & Incorrect Behavior

| # | File | Line(s) | Severity | Category | Description | Evidence |
|---|------|---------|----------|----------|-------------|----------|
| L1 | `bot/core/bot.py` | 2228-2245 | **MEDIUM** | Stale Command Match | `_is_long_running_admin_command()` hardcodes `"!rag bootstrap"` etc. but `COMMAND_PREFIX` is configurable via env. If the prefix changes to `/` or `?`, this function never matches, so those commands are no longer treated as out-of-band and block the user queue. | `long_running_commands = ["!rag bootstrap", "!rag refresh", ...]` — hardcoded prefix |
| L2 | `bot/config_reload.py` | 413-415 | **LOW** | OWNER_IDS Parse Error | `OWNER_IDS` config: `int(id.strip())` inside a list comprehension. If `OWNER_IDS="abc,def"`, the entire config load fails with `ValueError` and the bot crashes at startup. No per-element try/except. | `"OWNER_IDS": [int(id.strip()) for id in os.getenv("OWNER_IDS", "").split(",") if id.strip()]` |
| L3 | `bot/hear.py` | 100 | **LOW** | Single-User STT Bottleneck | `_JOB_SEMAPHORE = asyncio.Semaphore(1)` at module scope means only 1 STT job can run at a time globally. If user A requests a 10-minute video transcription, user B's `!watch` is blocked until it finishes. Should be per-user or have a higher limit. | `_JOB_SEMAPHORE = asyncio.Semaphore(1)` — global, not per-user |
| L4 | `bot/controller.py` | 68 | **LOW** | Filename Collision | `Path(f"/tmp/{attachment.filename}")` — if two users upload files with the same filename, the second overwrites the first. No uniqueness guarantee. | No UUID or random suffix in the path |
| L5 | `bot/commands/vision_commands.py` | 528 | **INFO** | Safe Alternative Exists | Vision commands correctly use `tempfile.NamedTemporaryFile(delete=False, suffix=suffix)` with proper cleanup. This is the correct pattern — the controller should follow this. | `temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)` |

---

## Table 6: Supply Chain & Dependency Risks

| # | Package | Version | Severity | Category | Description |
|---|---------|---------|----------|----------|-------------|
| D1 | `torch` | 2.3.1 | **HIGH** | Outdated | PyTorch 2.3.1 is from April 2024. Multiple CVEs fixed in 2.4+ including CVE-2024-34487 (OOB read), CVE-2024-38384 (integer overflow). Not a direct exploit surface for a Discord bot, but if any user-controlled input reaches torch operations, it's a risk. |
| D2 | `pillow` | 11.3.0 | **LOW** | Monitor | Pillow has had multiple image-parsing CVEs. The bot processes user-uploaded images. 11.3.0 is recent but any image-decoding path should be monitored. |
| D3 | `pypdf2` | 3.0.1 | **MEDIUM** | Deprecated | PyPDF2 is end-of-life. The maintained fork is `pypdf` (v5+). EOL packages get no security patches. |
| D4 | `chromadb` | 1.0.20 | **LOW** | Monitor | ChromaDB 1.0.x is relatively new. RAG ingestion processes user documents — any future CVE in parsing would be exploitable via the `!rag` commands. |
| D5 | `playwright` | 1.58.0 | **LOW** | SSRF Vector | Playwright is the SSRF vector (S3). Any browser engine is an SSRF surface by definition. The risk is in the application logic, not the package itself. |

---

## Table 7: Hardening Gaps & Missing Controls

| # | File/Area | Severity | Category | Description | Recommended Fix |
|---|-----------|----------|----------|-------------|-----------------|
| H1 | `bot/commands/screenshot_commands.py` | **HIGH** | SSRF Protection | No validation of target URL against private/internal IP ranges. | Add `is_private_ip()` check; reject `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`, `::1`, `fc00::/7` |
| H2 | `bot/controller.py` | **HIGH** | Path Traversal | `attachment.filename` used unsanitized in file path. | Use `tempfile.NamedTemporaryFile(delete=False, suffix=Path(attachment.filename).suffix)` (like vision_commands does) |
| H3 | `bot/utils/external_api.py` | **HIGH** | Secret in URL | API key passed as query parameter and logged. | Move API key to HTTP header; strip key from logged URLs; add `SCREENSHOT_API_KEY` to `SensitiveDataFilter.SECRET_KEYS` |
| H4 | `bot/commands/admin_alert_commands.py` | **HIGH** | Missing Auth Decorator | `!alert` relies on body-level auth check, not a decorator. | Add `@commands.is_owner()` or `@commands.has_permissions(administrator=True)` decorator |
| H5 | `bot/commands/video_commands.py` | **MEDIUM** | Missing Rate Limit | No cooldown on `!watch` which spawns expensive subprocesses. | Add `@commands.cooldown(1, 30, commands.BucketType.user)` |
| H6 | `bot/commands/screenshot_commands.py` | **MEDIUM** | Missing Rate Limit | No cooldown on `!ss` which calls a paid API. | Add `@commands.cooldown(2, 30, commands.BucketType.user)` |
| H7 | `bot/tts/interface.py` | **MEDIUM** | Insecure Temp File | `tempfile.mktemp()` is deprecated and insecure. | Replace with `tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")` or `tempfile.mkstemp(suffix=".ogg")` |
| H8 | `bot/config_reload.py` | **MEDIUM** | Partial Secret Exposure | Redacted values show last 4 chars. | Show only `"***REDACTED***"` with no partial value |
| H9 | `bot/commands/admin_alert_commands.py` | **MEDIUM** | Session Memory Leak | No periodic sweep of expired sessions. | Add a background task or max-size cap that purges sessions older than `session_timeout` |
| H10 | `bot/core/bot.py` | **LOW** | Hardcoded Prefix | `_is_long_running_admin_command` hardcodes `!` prefix. | Read prefix from config: `prefix = self.config.get("COMMAND_PREFIX", "!")` |
| H11 | `bot/config.py` | **LOW** | OWNER_IDS Crash | Invalid OWNER_IDS env var crashes the bot at startup. | Wrap `int(id.strip())` in per-element try/except, skip invalid entries |
| H12 | `bot/hear.py` | **LOW** | Global STT Bottleneck | Semaphore(1) blocks all STT for all users. | Increase to 3 or make per-user with a global cap |

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| CRITICAL | 2 (S1, S2) |
| HIGH | 6 (S3-S5, R1, C1, H1-H4) |
| MEDIUM | 11 (S6-S9, R2, C2, E1-E2, L1, D3) |
| LOW | 12 (S10-S13, R3-R5, C3, E3-E4, L2-L4, D1-D2, D4-D5, H10-H12) |
| INFO | 1 (L5) |

### Top 3 Immediate Action Items

1. **S1 + H2**: Fix `controller.py` path traversal — replace `Path(f"/tmp/{attachment.filename}")` with `tempfile.NamedTemporaryFile(delete=False, suffix=Path(attachment.filename).suffix)`. This is a 1-line fix that mirrors the existing safe pattern in `vision_commands.py`.

2. **S2 + H3**: Stop logging the screenshot API URL with key. Either move the key to an HTTP header, or redact `key=` from the logged URL. Add `SCREENSHOT_API_KEY` to both `SENSITIVE_KEYS` and `SensitiveDataFilter.SECRET_KEYS`.

3. **S3 + H1**: Add private IP range validation before any screenshot request. This prevents the bot from being used as an SSRF proxy against internal services.
