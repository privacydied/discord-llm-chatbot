# Memory and Context Management
## Thread Tail Replies & Context (Threads/Forum)

- Always enabled in thread channels. No feature flag.
- Limits (env defaults):
  - `THREAD_CONTEXT_TAIL_COUNT=5` (number of previous messages to include)
- Behavior:
  - At send-time, the bot resolves the reply target to the newest message in the thread. If the newest is the bot and the previous is human, it replies to that human to avoid reply-to-self loops. If no human messages exist, it posts in the thread without a reply reference.
  - The router prepends a bounded “thread tail” context block built from up to K messages strictly before the reply target (ordered oldest→newest). Humans + the bot are included; other bots/system messages are excluded. Sanitization is applied.
  - If the triggering thread message contains no meaningful text (e.g., only an @mention), the router uses the resolved reply target’s text as the input content for this hop to avoid empty prompts.
- Telemetry (JSONL):
  - `subsys=mem.thread event=tail_ok` with `detail.k`, `detail.reply_target`, `detail.count`.
  - `subsys=mem.thread event=tail_fallback` with `detail.reason` on timeout/archived/permission errors.
- Fallbacks: On any failure, the bot falls back to the pre-existing behavior (reply to the triggering message, no tail context).

### Example .env additions

```env
# Threads: tail context size (always enabled)
THREAD_CONTEXT_TAIL_COUNT=5
```


This guide explains how the bot manages conversation context (ephemeral), user memory (persistent per user), and server memory (persistent per guild). It also documents related configuration keys and recommended practices.

- Applies Clean Architecture [CA], Robust Error Handling [REH], Input Validation [IV], and Constants Over Magic Values [CMV].
- Optimized for performance and resource usage [PA].

## Concepts

- **Conversation Context (Ephemeral):**
  - The rolling window of most recent messages included when generating an AI response.
  - Controlled by `MAX_CONTEXT_MESSAGES`.
  - Can be restricted to in-RAM only (no log files) with `IN_MEMORY_CONTEXT_ONLY`.

- **User Memory (Persistent):**
  - A small set of salient facts about a user (e.g., preferences, roles, recurring requests).
  - Size limited by `MAX_USER_MEMORY`.
  - Periodically saved per `MEMORY_SAVE_INTERVAL`.

- **Server Memory (Persistent):**
  - Shared knowledge captured at the guild/server level (e.g., channel conventions, team norms).
  - Size limited by `MAX_SERVER_MEMORY`.

> Note [KBT]: The exact storage engine/files may vary by deployment. If an environment key is not yet wired in your build, treat this document as the authoritative specification for intended behavior.

## Configuration Keys

- **MAX_CONTEXT_MESSAGES** (int)
  - Maximum number of most-recent messages kept in the model context.
  - Typical values: `20–60`. Default example: `30`.

- **IN_MEMORY_CONTEXT_ONLY** (bool)
  - If `true`, store conversation context only in memory; no disk logs are written.
  - Use for higher privacy or ephemeral environments.

- **MAX_USER_MEMORY** (int)
  - Maximum stored “memories” per user. Default example: `5`.

- **MAX_SERVER_MEMORY** (int)
  - Maximum stored server-level memories per guild. Default example: `100`.

- **MEMORY_SAVE_INTERVAL** (seconds)
  - Periodic interval to flush in-memory user/server profiles to persistent storage.
  - Default example: `30` seconds.

### Optional Paths and Limits

- **USER_PROFILE_DIR** (path)
  - Directory for user profiles (JSON/NDJSON/SQLite—implementation dependent).

- **SERVER_PROFILE_DIR** (path)
  - Directory for server/guild profiles.

- **DM_LOGS_DIR** (path)
  - Directory for DM conversation logs when not using `IN_MEMORY_CONTEXT_ONLY`.

- **TTS_PREFS_FILE** (path)
  - File storing TTS user preferences (voice, speed, language) if your build supports TTS.

- **MAX_CONVERSATION_LOG_SIZE** (int)
  - Maximum bytes/lines per conversation log file to cap disk usage.

- **DEBUG** (bool)
  - Enables extra diagnostics for memory/context components. Use cautiously in production.

## Behavior and Lifecycle

- **Context Window**
  - On each message, collect up to `MAX_CONTEXT_MESSAGES` most-recent exchanges.
  - Respect `IN_MEMORY_CONTEXT_ONLY`: if true, skip file writes for conversation logs.

- **User/Server Memory Updates**
  - On relevant events (new stable preferences, repeated patterns), update memory stores.
  - A background or scheduled task flushes changes at `MEMORY_SAVE_INTERVAL`.

- **Privacy Controls**
  - Set `IN_MEMORY_CONTEXT_ONLY=true` to disable writing conversation logs to disk.
  - Keep `MAX_*` values conservative to reduce data retention.

- **Resource Management** [RM]
  - Enforce `MAX_*` limits strictly to prevent unbounded memory growth.
  - Rotate/trim logs when `MAX_CONVERSATION_LOG_SIZE` is reached.

## Recommended Settings

- Local dev:
  - `MAX_CONTEXT_MESSAGES=30`
  - `IN_MEMORY_CONTEXT_ONLY=true`
  - `MAX_USER_MEMORY=5`
  - `MAX_SERVER_MEMORY=100`
  - `MEMORY_SAVE_INTERVAL=30`

- Production:
  - `MAX_CONTEXT_MESSAGES=40-60` (task-complexity dependent)
  - `IN_MEMORY_CONTEXT_ONLY=false` (unless strict privacy mandates)
  - `MAX_USER_MEMORY=5-10`
  - `MAX_SERVER_MEMORY=200-500` (depends on team size)
  - `MEMORY_SAVE_INTERVAL=30-120`

## Example .env

```env
# Context & Memory
MAX_CONTEXT_MESSAGES=30
IN_MEMORY_CONTEXT_ONLY=false
MAX_USER_MEMORY=5
MAX_SERVER_MEMORY=100
MEMORY_SAVE_INTERVAL=30

# Optional paths and limits
#USER_PROFILE_DIR=var/data/users
#SERVER_PROFILE_DIR=var/data/servers
#DM_LOGS_DIR=var/log/dm
#TTS_PREFS_FILE=var/data/tts_prefs.json
#MAX_CONVERSATION_LOG_SIZE=10000
#DEBUG=false
```

## Mention-aware Discord Threads & Reply Chains

- Always enabled. No feature flag required.
- Limits (env defaults):
  - `MEM_MAX_MSGS=40`
  - `MEM_MAX_CHARS=8000`
  - `MEM_MAX_AGE_MIN=240`
  - `MEM_FETCH_TIMEOUT_S=5`
  - `MEM_LOG_SUBSYS=mem.ctx`
- Behavior: When users @mention the bot, the router builds a bounded context block before normal memory:
  - Inside a Thread/Forum post: recent thread messages (oldest→newest), includes the thread starter and the triggering message.
  - Reply in a regular channel: linear reply chain around the root (up/down via references), ending at the triggering message.
  - Lone mention (no thread, no reply): unchanged.
- Guardrails: other bots are excluded (except this bot), age/char/message caps enforced; timeouts fall back to current behavior.
- Telemetry (JSONL):
  - `collect_ok` with `detail.case`, `detail.msgs`, `detail.chars`, `detail.ms`.
  - `collect_truncated` when caps hit.
  - `collect_fallback` on timeout/permission/fetch issues.
  - `merge_ok` when the new block is prepended before historical memory.

### Manual validation

- Mention inside a thread → context includes the thread’s recent messages in order (bounded by caps).
- Reply-mention in a regular channel → context includes only the reply chain, not unrelated chatter.
- Plain mention (no thread/reply) → unchanged behavior.
- Huge thread → verify truncation and clean answer.
- Interleaved chat → only reply-chain messages are included.

## Troubleshooting

- **Context feels too short**
  - Increase `MAX_CONTEXT_MESSAGES`. Be mindful of LLM token limits and latency [PA].

- **Too much retention**
  - Lower `MAX_*` values or enable `IN_MEMORY_CONTEXT_ONLY=true`.

- **High disk usage**
  - Set/adjust `MAX_CONVERSATION_LOG_SIZE` and prune old logs.

- **Unexpected persistence**
  - Ensure `IN_MEMORY_CONTEXT_ONLY` is correctly set and the process has no other log writers.

## Notes for Developers

- Validate env vars at startup [IV], log effective values using structured logs.
- Add unit tests around trimming logic and persistence boundaries [REH].
- Keep functions under 30 lines and avoid deep nesting [CSD].
- Prefer named constants over literals [CMV].
