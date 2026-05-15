# Discord-LLM-Chatbot Architecture Map for Dashboard Integration

## 1. Startup Flow

```
bot/main.py:run_bot()                       # Entry point (line 124)
  └─ asyncio.run(main_with_cleanup())       # line 128
      └─ main(bot_ref={})                   # line 145
          ├─ init_logging()                 # line 38
          ├─ parse_arguments()              # line 40
          ├─ check_venv_activation()        # line 56
          ├─ setup_config_reload()          # line 59 (config_reload.py)
          ├─ load_config()                  # line 61 (bot/config/__init__.py → _base.py)
          ├─ load_system_prompts()          # line 62 (bot/config/prompts.py)
          ├─ run_pre_flight_checks(config)  # line 63 (bot/core/startup.py)
          ├─ create_bot_intents()           # line 71 (discord.Intents.none() + guilds, messages, message_content)
          ├─ LLMBot(config=..., command_prefix=get_prefix, intents=..., help_command=None, max_messages=...)  # lines 75-81
          │   └─ bot_ref["bot"] = bot       # line 83 (exposes live bot instance)
          ├─ setup_signal_handlers(bot)     # line 86 (bot/shutdown.py: SIGINT, SIGTERM)
          ├─ await spawn_background_tasks(bot)  # line 91 (bot/tasks.py)
          │   └─ TaskManager(bot).start_all_tasks()
          │       ├─ profile_autosave       (discord.ext.tasks.loop, minutes=PROFILE_AUTOSAVE_INTERVAL)
          │       ├─ cleanup_old_logs       (discord.ext.tasks.loop, hours=CLEANUP_INTERVAL_HOURS)
          │       ├─ health_check           (discord.ext.tasks.loop, minutes=HEALTH_CHECK_INTERVAL)
          │       ├─ start_memory_service   (bot/memory/service.py)
          │       ├─ start_memory_distiller (bot/memory/archive_distiller.py)
          │       ├─ start_server_archive_service (bot/server_archive/service.py)
          │       └─ start_janitor          (bot/janitor.py)
          ├─ await start_file_watcher()     # line 92 (config_reload.py, watches .env and prompt files)
          └─ await bot.start(TOKEN)         # line 102 (blocks until disconnect)
              └─ Discord on_ready fires → LLMBot.on_ready() (line 785)
                  └─ sets self._is_ready event
```

### Key: LLMBot.setup_hook() (runs BEFORE on_ready, after Discord connects)
```
bot/core/bot.py:setup_hook()  (line 447)
  ├─ torch_compat shim (deferred)
  ├─ _install_public_output_safety_hooks()   # Patches discord.abc.Messageable.send, Message.reply/edit, etc.
  ├─ self.message_processor = MessageProcessor(self)
  ├─ PrometheusMetrics init (lines 476-493)
  │   ├─ PROMETHEUS_ENABLED env (default "true")
  │   ├─ PROMETHEUS_PORT env (default 8000)
  │   └─ start_http_server(port) via prometheus_client
  ├─ Define metric counters (gate, x.photo_to_vl, x.syndication, vision.route)
  ├─ Load system prompts
  ├─ Register config hot-reload callback (add_reload_callback)
  ├─ load_profiles()
  ├─ setup_background_tasks()                 # line 755 - SECOND layer of background tasks
  │   ├─ start_memory_service                 # curated memory asyncio.Task
  │   ├─ start_memory_distiller               # distiller asyncio.Task
  │   ├─ setup_memory_save_task               # discord.ext.tasks.Loop
  │   └─ start_server_archive_service         # server archive asyncio.Task
  ├─ setup_tts()
  ├─ setup_router()                           # Creates Router + VisionOrchestrator
  ├─ setup_rag()
  ├─ load_extensions()                        # Command cogs
  └─ setup_command_error_handler()
```

**Important**: Background tasks are registered in TWO places:
1. `main.py:91` → `spawn_background_tasks()` → `TaskManager.start_all_tasks()` (BEFORE bot.start)
2. `bot.py:setup_hook():755` → `self.setup_background_tasks()` (AFTER Discord connects, BEFORE on_ready)

This means tasks are duplicated (memory_service, memory_distiller, server_archive are started in BOTH places).


## 2. Bot Instance Lifecycle & Access

### How to access the live bot instance:
- **`bot_ref` dict** in `main()`: `bot_ref["bot"]` holds the LLMBot instance (line 83)
- **Global via `bot_ref`**: The only clean reference path from `main()` scope
- **Module-level singletons**: Various services store references to `self.bot`:
  - `bot/server_archive/service.py:_service` (ServerArchiveService singleton)
  - `bot/tasks.py:_task_manager` (TaskManager singleton)
  - `bot/shutdown.py:_shutdown_manager` (GracefulShutdown singleton)
  - `bot/tasks_registry.py:_registry` (BackgroundTaskRegistry singleton)
  - `bot/memory/service.py` and `bot/memory/archive_distiller.py` have their own singletons

### LLMBot key attributes for dashboard access:
```python
bot.config              # dict - live config snapshot
bot.user                # discord.ClientUser - bot's own user
bot.guilds              # list[discord.Guild] - all guilds
bot._is_ready           # asyncio.Event - signals ready state
bot.metrics             # PrometheusMetrics or NoopMetrics
bot.router              # Router instance (message dispatch)
bot.tts_manager         # TTSManager or None
bot.archive_service     # ServerArchiveService or None
bot.message_processor   # MessageProcessor
bot.user_profiles       # dict - user profile cache
bot.server_profiles     # dict - server profile cache
bot._background_tasks   # set[asyncio.Task] - tracked tasks
bot._active_long_running_tasks  # dict[str, asyncio.Task]
bot.console             # Rich Console
bot.memory_save_task    # discord.ext.tasks.Loop
```


## 3. Guilds/Channels/Users Access Pattern

```python
# From LLMBot instance:
bot.guilds                           # list[discord.Guild]
bot.get_guild(guild_id)              # discord.Guild | None
bot.get_channel(channel_id)          # discord.abc.GuildChannel | None
bot.get_user(user_id)                # discord.User | None

# Guild object:
guild.id, guild.name, guild.member_count, guild.owner
guild.channels                       # list of channels
guild.text_channels, guild.voice_channels

# Channel object:
channel.id, channel.name, channel.guild
channel.members                      # for voice channels
await channel.history(limit=N)       # message history

# ServerArchiveService also provides:
service.store.counts(guild_id=...)   # {guilds, channels, threads, users, messages, ...}
service.get_status(guild_id=...)     # full status dict
```


## 4. Metrics Exposure

### PrometheusMetrics (bot/metrics/prometheus_metrics.py)
- **HTTP Server**: `prometheus_client.start_http_server(port)` on port 8000 (default)
- **Env vars**:
  - `PROMETHEUS_ENABLED` (default "true" in bot.py setup_hook, but "false" in metrics/__init__.py)
  - `PROMETHEUS_PORT` (default 8000 in bot.py, 8001 in metrics/__init__.py)
  - `PROMETHEUS_HTTP_SERVER` (default "true")
  - `OBS_ENABLE_PROMETHEUS` (default "false" in metrics/__init__.py)
- **Defined counters**: gate.allowed, gate.blocked, x.photo_to_vl.*, x.syndication.*, vision.route.*
- **Interface**: `inc()`, `increment()`, `observe()`, `gauge()`, `timer()`, `define_counter()`, `define_histogram()`, `define_gauge()`
- **Scrape endpoint**: `http://0.0.0.0:8000/metrics` (standard Prometheus `/metrics` path)

### NoopMetrics / NullMetrics
- Used when Prometheus is disabled or unavailable
- All methods are no-ops

### Metric name constants (bot/metrics/__init__.py):
```python
METRIC_STARTUP_TOTAL_DURATION
METRIC_STARTUP_COMPONENT_DURATION
METRIC_STARTUP_PARALLEL_GROUPS
METRIC_COMPONENT_INIT_SUCCESS / FAILURE
METRIC_DEGRADED_MODE
METRIC_BACKGROUND_HEARTBEAT / LAST_HEARTBEAT / CONSECUTIVE_ERRORS / STALENESS
METRIC_PROCESS_RSS_BYTES
METRIC_EVENT_LOOP_LAG_SECONDS
METRIC_ERRORS_BY_MODULE
```


## 5. SQLite Patterns

### Two SQLite databases:

#### A. Server Archive (bot/server_archive/)
- **File**: `./data/server_archive.db` (configurable via SERVER_ARCHIVE_DB_PATH)
- **Tables**:
  - `archive_guilds`, `archive_channels`, `archive_threads`
  - `archive_users`
  - `archive_messages` (main message store)
  - `archive_attachments`, `archive_mentions`
  - `archive_sync_state` (sync tracking)
  - `archive_messages_fts` (FTS5 virtual table for full-text search)
  - `memory_distiller_state`, `memory_distiller_runs`
- **Pattern**: Thread-per-operation with `threading.RLock`, WAL mode, `PRAGMA synchronous=NORMAL`
- **Connection**: Short-lived (open, execute, close per operation)
- **Async bridge**: `await asyncio.to_thread(sync_func)`
- **Schema versioning**: `PRAGMA user_version`
- **Store class**: `ServerArchiveStore` (bot/server_archive/store.py)

#### B. Persistent Memory (bot/memory/persistent_store.py)
- **File**: Configurable (MEMORY_DB_PATH env)
- **Table**: `curated_memories`
  - Columns: memory_id (PK), user_id, guild_id, channel_id, thread_id, source_message_id, context_type, text, summary, importance, confidence, created_at, updated_at, last_accessed_at, expires_at, source, deleted_at, chroma_id, metadata_json
  - Indexes: idx_curated_memories_user_active, idx_curated_memories_guild_active, idx_curated_memories_context_type, idx_curated_memories_last_accessed
- **Pattern**: Same thread-per-operation with RLock, WAL mode
- **Store class**: `PersistentMemoryStore`
- **Also**: ChromaDB for semantic search (separate from SQLite)

### Shared SQLite pattern:
```python
def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.sqlite_path, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

# Each operation: open → lock → execute → commit → close
# All DB writes go through asyncio.to_thread() to avoid blocking event loop
```


## 6. Flask Server Status

**Flask is NOT used for the bot's dashboard or metrics.** It exists only for:
- `tts/service/app.py`: A standalone Kokoro TTS service on port 5000 (not started by the bot)
- Listed as a dependency in pyproject.toml / requirements.txt

The Prometheus metrics HTTP server uses `prometheus_client.start_http_server()` (a built-in lightweight WSGI server, NOT Flask).

**There is currently NO aiohttp web server in the bot.** The bot uses aiohttp for HTTP client operations only.


## 7. Safest Integration Point for aiohttp Dashboard Server

### Recommended approach: Run aiohttp alongside Discord bot in the same event loop

The cleanest integration point is in `bot/main.py`, between `spawn_background_tasks` and `bot.start()`:

```python
# In bot/main.py, around line 91-102:
async def main(bot_ref: Optional[Dict[str, LLMBot]] = None) -> NoReturn:
    # ... existing startup code ...
    
    await spawn_background_tasks(bot)
    await start_file_watcher()
    
    # === DASHBOARD INTEGRATION POINT ===
    # Create aiohttp web app and runner BEFORE bot.start()
    from bot.web.dashboard import create_dashboard_app, start_dashboard
    
    dashboard_port = int(os.getenv("DASHBOARD_PORT", "8080"))
    await start_dashboard(bot, port=dashboard_port)
    # ====================================
    
    # Then start Discord bot (blocks until disconnect)
    await bot.start(config["DISCORD_TOKEN"])
    
    # Dashboard cleanup happens in finally/bot.close()
```

### Why this is the safest point:
1. **Same event loop**: aiohttp runs in the same asyncio loop as the Discord bot
2. **Bot instance available**: The LLMBot is fully constructed and accessible
3. **Before bot.start()**: Dashboard is ready before Discord connection
4. **No port conflicts**: Prometheus uses 8000, dashboard uses 8080 (configurable)
5. **Clean shutdown**: aiohttp AppRunner can be closed in `bot.close()` or signal handlers

### Alternative: Add as a background task in setup_background_tasks()

```python
# In bot/core/bot.py:setup_background_tasks() (line 2343)
async def _start_dashboard(self):
    from bot.web.dashboard import create_dashboard_app
    from aiohttp import web
    
    app = create_dashboard_app(self)  # Pass bot reference
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(self.config.get("DASHBOARD_PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    self._dashboard_runner = runner  # Store for cleanup
    
    # Keep running until cancelled
    while True:
        await asyncio.sleep(3600)

# Add to setup_background_tasks():
dashboard_task = asyncio.create_task(self._start_dashboard(), name="dashboard")
self._track_background_task(dashboard_task)
```

### Dashboard should expose these data sources:

| Data Source | Access Pattern | Module |
|---|---|---|
| Bot status | `bot.is_ready()`, `bot.user`, `bot.guilds` | `bot.core.bot` |
| Guilds/channels | `bot.guilds`, `bot.get_guild()` | discord.py |
| Config | `bot.config` (dict) | `bot.config` |
| Prometheus metrics | `http://localhost:8000/metrics` (proxy or parse) | `bot.metrics.prometheus_metrics` |
| Server archive stats | `await get_server_archive_status()` | `bot.server_archive.service` |
| Memory stats | `await get_memory_service()`, `await get_memory_distiller_status()` | `bot.memory.service` |
| Background tasks | `get_registry().list_tasks()`, `get_registry().summary()` | `bot.tasks_registry` |
| Task status | `get_task_status()` | `bot.tasks` |
| Bot metrics counters | `bot.metrics` (PrometheusMetrics or NoopMetrics) | `bot.metrics` |
| Server archive query | `await search_archive(query, guild_id=...)` | `bot.server_archive.service` |
| Memory search | `await search_user_memories(user_id, query)` | `bot.memory.service` |
| Resource usage | psutil (health check task pattern) | `bot.tasks` |

### Module Dependencies for Dashboard:

```
bot/web/dashboard.py (NEW)
  ├── aiohttp.web (Application, AppRunner, TCPSite, Request, Response, json_response)
  ├── bot.core.bot (LLMBot reference)
  ├── bot.metrics (get_metrics, PrometheusMetrics)
  ├── bot.server_archive (get_server_archive_service, get_server_archive_status, search_archive)
  ├── bot.memory (get_memory_service, get_memory_distiller_status, search_user_memories)
  ├── bot.tasks_registry (get_registry)
  ├── bot.tasks (get_task_status)
  └── bot.config (load_config)
```

### Shutdown integration:
Add to `bot/core/bot.py:close()` (around line 2720):
```python
# Close dashboard aiohttp runner
if hasattr(self, "_dashboard_runner") and self._dashboard_runner:
    try:
        await self._dashboard_runner.cleanup()
        self.logger.info("Dashboard server stopped")
    except Exception as e:
        self.logger.warning(f"Error stopping dashboard: {e}")
```


## 8. Module Dependency Map

```
bot/main.py
├── bot/config/__init__.py → _base.py, prompts.py
├── bot/config_reload.py
├── bot/core/bot.py (LLMBot)
│   ├── bot/core/message_processor.py
│   ├── bot/core/fast_path_router.py
│   ├── bot/metrics/__init__.py → null_metrics.py, prometheus_metrics.py
│   ├── bot/memory/__init__.py → service.py, profiles.py, context_manager.py,
│   │   enhanced_context_manager.py, semantic_store.py, persistent_store.py,
│   │   retrieval.py, scoring.py, curator.py, ingestion_queue.py, mention_context.py
│   ├── bot/server_archive/__init__.py → service.py, store.py, models.py, sync.py
│   ├── bot/tts/interface.py
│   ├── bot/vision/gateway.py, budget_manager.py, artifact_cache.py
│   ├── bot/router.py
│   └── bot/commands/*.py (cogs)
├── bot/core/startup.py
├── bot/core/cli.py
├── bot/tasks.py → tasks_registry.py, janitor.py
├── bot/shutdown.py
└── bot/utils/logging.py

External HTTP clients (aiohttp):
├── bot/http_client.py (shared aiohttp.ClientSession)
├── bot/ollama.py
├── bot/openai_backend.py
├── bot/nvidia_backend.py
├── bot/web_extraction_service.py
├── bot/core/openrouter_client.py
└── bot/rag/chroma_backend.py

Flask (NOT in bot):
└── tts/service/app.py (standalone TTS service, port 5000)
```


## 9. Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DISCORD_TOKEN` | (required) | Discord bot token |
| `PROMETHEUS_ENABLED` | "true" (bot.py) / "false" (metrics/__init__.py) | Enable Prometheus metrics |
| `PROMETHEUS_PORT` | 8000 | Prometheus scrape port |
| `PROMETHEUS_HTTP_SERVER` | "true" | Start Prometheus HTTP server |
| `OBS_ENABLE_PROMETHEUS` | "false" | Alternative Prometheus toggle |
| `SERVER_ARCHIVE_ENABLED` | False | Enable server archive ingest |
| `SERVER_ARCHIVE_DB_PATH` | ./data/server_archive.db | Archive SQLite path |
| `PERSISTENT_MEMORY_ENABLE` | True | Enable curated memory |
| `MEMORY_DISTILLER_ENABLED` | False | Enable memory distiller |
| `DASHBOARD_PORT` | 8080 (proposed) | Dashboard HTTP port |


## 10. Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Port conflict with Prometheus (8000) | Medium | Use separate port (8080) |
| Blocking event loop with dashboard handlers | High | Use aiohttp (async), never sync DB calls in handlers |
| SQLite concurrent access | Medium | Current RLock + WAL pattern handles this; dashboard reads are safe |
| Duplicate background tasks | High | Note: memory_service/distiller started in both spawn_background_tasks AND setup_background_tasks |
| Memory leak from long-lived aiohttp sessions | Low | aiohttp handles lifecycle; cleanup in bot.close() |
| Config hot-reload race with dashboard | Low | Dashboard reads bot.config snapshot; reload swaps atomically |
