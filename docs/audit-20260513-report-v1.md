# PRODUCT AUDIT -- discord-llm-chatbot

## Date: 2026-05-13
## Version: v1

---

## 1. Executive Summary

A production-grade Discord chatbot (v0.2.0) that acts as an AI multi-modal assistant across text, image, video, audio, and web content. The bot sits in Discord guilds and DMs, responding to mentions, replies, commands, and URLs. It integrates with LLM providers via OpenRouter (primary), Ollama, and NVIDIA NIM, with fallback ladders for both text and vision models. Key capabilities include:

- **Text chat** with LLM backends (OpenRouter, Ollama, NVIDIA NIM) via OpenAI-compatible API
- **Vision-language (VL)** analysis of images, screenshots, and embedded media
- **Image/video generation** via Together.ai and Novita.ai
- **Text-to-Speech (TTS)** using Kokoro ONNX (IPA/English G2P pipeline)
- **Speech-to-Text (STT)** using faster-whisper with multimodal API fallback
- **Online web search** (DuckDuckGo with custom provider support)
- **RAG (Retrieval-Augmented Generation)** with ChromaDB vector store and hybrid search
- **Persistent memory** (SQLite + ChromaDB) with automatic curation and user/server scoping
- **Server archiving** (SQLite-backed message history with search)
- **Twitter/X syndication** (API-first with web fallback, thread unrolling)
- **Screenshot capture** (external API + Playwright fallback)
- **Prometheus metrics** endpoint for operational monitoring
- **Hot-reload** of .env configuration via file watching and SIGHUP

The bot is deployed via Docker Compose on a Synology NAS (UID 1026), exposed on port 8010 for Prometheus metrics, with Discord gateway connectivity as its primary external interface.

---

## 2. Architecture Overview

**Deployment model:** Self-hosted Docker container on Synology NAS. Playwright runs as a separate Docker container on port 3006 (remote browser server).

**Tech stack:**
- Python 3.11+ (async-first, discord.py 2.6.3)
- OpenAI-compatible client (aiohttp) for LLM calls
- ChromaDB for vector embeddings (RAG + memory semantic store)
- SQLite for persistent memory + server archive
- ONNX Runtime + Kokoro for TTS
- faster-whisper for STT
- Playwright for browser automation (web extraction, screenshots)
- Flask (internal health/metrics server on port 8000)
- Prometheus for observability
- Together.ai / Novita.ai for image/video generation

**Directory structure:**
```
bot/
  commands/    # 17 command cog files (memory, TTS, RAG, vision, search, etc.)
  config/      # Environment config loader (config.py, config_reload.py) + media_config.py
  core/        # LLMBot class (3100 lines), startup, permissions, CLI, output, phases
  events/      # Command error handler (global)
  infra/       # Voice memo sender
  memory/      # Short-term context + long-term curated memory (SQLite+ChromaDB)
  metrics/     # Prometheus metrics + null fallback
  persistence/ # Empty (persistence lives in memory/server_archive modules)
  rag/         # RAG system (ChromaDB, chunking, parsers, hybrid search, lazy loading)
  router_components/  # Decomposed router: compose, gating, input_harvest, prompt_access, runtime, x_routing
  routing/     # Route handler protocol (screenshot, unknown)
  search/      # Search provider system (DDG + custom providers)
  server_archive/     # SQLite message archive (sync, ingest queue, search)
  stt_module/  # STT multimodal fallback, failure classifier
  stt_pipeline/       # STT orchestration (ffmpeg runtime, youtube path, transcribe flow)
  syndication/ # Twitter/X syndication (handler, extract, url_utils)
  threads/     # X thread unrolling
  tts/         # TTS manager, engines (kokoro-onnx, kokoro v8, stub), IPA vocab
  types/       # Shared types (Command, ParsedCommand, ResponseMessage, InputModality)
  utils/       # Cross-cutting utilities (logging, playwright, file, env, caching)
  vision/      # Vision generation (gateway, orchestrator, budget, safety, providers)
  vl/          # VL postprocessing
  voice/       # Discord voice publisher (TTS audio -> voice channel)
  exceptions.py        # Central exception hierarchy
  config.py            # ~1026-line configuration loader
  config_reload.py     # Hot .env reload with file watcher
  shutdown.py          # Graceful shutdown (SIGTERM/SIGINT)
  main.py              # Bootstrap: parse args, load config, create bot, start
  tasks.py             # Background tasks (autosave, distillation, etc.)
  modality.py          # InputModality enum
  janitor.py           # Background cleanup daemon
  ollama.py            # Ollama client wrapper
  router.py            # Main router (dispatches messages to backends)
```

**Key architectural patterns:**
- **Protocol-based routing** -- `RouteHandler` protocol with `can_handle()` / `handle()` methods, composed via `compose.py` into an evaluation chain
- **Lazy initialization** -- TTS, RAG, and memory services are singletons initialized on first use, not at startup
- **Circuit breakers** -- Search, X API, and other external services use configurable circuit breaker patterns (failure window, open timeout, half-open probability)
- **Tiered web timeouts** -- Three tiers (A: fast, B: medium, C: slow) for different URL extraction strategies
- **Multi-modal processing** -- Each input can carry attachments, URLs, screenshots, voice recordings; the router classifies modality and dispatches accordingly

---

## 3. Startup & Bootstrap

**Entry point:** `run.py` -> `bot.main.run_bot()` -> `asyncio.run(main_with_cleanup())`

1. `init_logging()` -- Sets up dual-sink logging (Rich console + JSONL file). If enforcer fails to find both handlers, startup aborts.
2. `parse_arguments()` -- CLI arg parsing (--version, --debug, --config-check)
3. `setup_config_reload()` -- Registers SIGHUP handler and file watcher for .env hot-reload. Watches cwd/.env, root/.env, and yoroi.env variants.
4. `load_config()` -- Loads all 150+ environment variables from .env with defaults, type conversion, and boolean parsing. Caches for 300s.
5. `load_system_prompts()` -- Reads text and VL prompt files from paths in env vars.
6. `run_pre_flight_checks()` -- Validates at least one AI provider is configured, checks network reachability.
7. `create_bot_intents()` -- Creates discord.py Intents with members, messages, message_content, reactions, guilds.
8. `LLMBot.__init__()` -- Constructs the bot with config, intents, prefix, no help command.
9. `setup_signal_handlers(bot)` -- Registers SIGTERM/SIGINT for graceful shutdown.
10. `await spawn_background_tasks(bot)` -- Initializes background tasks (autosave, memory distiller, janitor, etc.)
11. `await start_file_watcher()` -- Starts async task watching .env for changes.
12. `await bot.start(config["DISCORD_TOKEN"])` -- Connects to Discord (retries up to 3 times with exponential backoff).

**LLMBot.setup_hook()** (called on Discord connection):
- Loads metrics (Prometheus or NoopMetrics)
- Creates MessageProcessor instance
- Loads system prompts into bot state
- Registers config hot-reload callbacks
- Starts background tasks
- Initializes TTS subsystem (lazy)
- Initializes router (lazy)
- Initializes RAG system (lazy unless eager loading enabled)
- Loads all command cogs via `setup_commands()`

**Signal handlers:** SIGTERM and SIGINT trigger `GracefulShutdown.shutdown_with_timeout(30s)`, which:
1. Closes bot Discord connection
2. Saves all user/server profiles to disk
3. Stops all background tasks
4. Forces cleanup of remaining resources on timeout

---

## 4. Core Flows -- Message Routing

### Message Entry Path

1. Discord fires `on_message` -> LLMBot receives
2. `process_commands()` checks prefix -- non-prefixed messages skip command processing
3. MessageProcessor extracts context (content, attachments, embeds, references)
4. Router evaluates the message through a chain of `RouteHandler` instances

### Router Architecture (Protocol-Based)

The router was decomposed from a monolithic class into composable components in `bot/router_components/`:

- **compose.py** -- Chains route handlers into a pipeline. Each handler declares `can_handle()` and `handle()`. First match wins.
- **gating.py** -- Feature gates (TTS enabled, vision enabled, RAG enabled, etc.) that gate entire routes
- **input_harvest.py** -- Extracts attachments, URLs, embeds, and references from Discord messages
- **prompt_access.py** -- Retrieves text/VL prompts with user context injection
- **runtime.py** -- Runtime state management and async execution context
- **x_routing.py** -- Massive module (1100+ exported names) handling all Twitter/X URL detection, syndication, API calls, thread unrolling, media extraction, and fallback paths

### Route Handler Registry

1. **X/Twitter URL handler** -- Detects x.com, twitter.com URLs. Routes to:
   - X API first (if enabled) with full tweet hydration (author, media, replies, polls, places)
   - Syndication fallback (web scrape with Playwright)
   - Thread unrolling (up to 30 tweets, 6000 chars)
   - Photo routing to VL analysis (configurable)
   - STT probe on X URLs (configurable, default enabled)

2. **General URL handler** -- Detects any other URLs. Routes to:
   - Tier A: Fast HTML extraction (trafilatura, requests, 2000ms deadline)
   - Tier B: Playwright rendering (8000ms deadline)
   - Tier C: Screenshot + VL analysis (8000ms deadline)
   - Domain-specific fast-fail for SPA hosts (medium.com, heavy.com)

3. **Image/URL handler** -- Direct image URLs route to VL analysis
4. **Screenshot handler** -- Captures webpage screenshot via external API, then runs VL
5. **Unknown handler** -- Catch-all for plain text chat

### Chat Flow (Plain Text)

1. Router classifies input modality (text-only, text+image, text+audio, etc.)
2. Checks silence gate (BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO -- default True)
3. Builds conversation context from ContextManager (last N messages)
4. Fetches relevant memories from CuratedMemoryService (semantic + keyword search)
5. Fetches RAG documents if enabled (hybrid search with confidence threshold)
6. Constructs prompt with system prompt + conversation history + context + memories + RAG
7. Calls LLM backend (OpenRouter, Ollama, or NVIDIA NIM) with fallback ladder
8. Parses response for commands (TTS toggle, /img, /video, etc.)
9. Sends response text to Discord
10. Optionally generates TTS audio and sends as voice message or attachment

### Multimodal Flow

For messages with images, voice recordings, video URLs, or document attachments:

1. InputModality classification (image, audio, video, document, text, url)
2. Per-item processing with budget enforcement (MULTIMODAL_PER_ITEM_BUDGET=45s)
3. Sequential processing with early termination on timeout
4. Results aggregated into context for LLM consumption

---

## 5. Command Inventory

### Standard Prefix Commands (!)

| Command | File | Description |
|---------|------|-------------|
| `!status` | operator_commands.py | Show operator/health status |
| `!feature` | operator_commands.py | Toggle server features |
| `!reload` | config_commands.py | Hot-reload .env configuration |
| `!tts` | tts_cmds.py | Toggle TTS for current user |
| `!tts-all` | tts_cmds.py | Admin: global TTS on/off |
| `!speak` | tts_cmds.py | Single TTS response, then revert to text |
| `!say` | tts_cmds.py | Say a specific message in TTS |
| `!alert` | admin_alert_commands.py | Admin DM alert broadcasting with TTS |
| `!index` | rag_commands.py | Index text/URL/attachment into RAG |
| `!rag-status` | rag_commands.py | Show RAG system status |
| `!rag-search` | rag_commands.py | Search RAG knowledge base |
| `!rag-clear` | rag_commands.py | Clear RAG collection |
| `!rag-invalidate` | rag_commands.py | Invalidate RAG version cache |
| `!search` | search_commands.py | Online web search (inline) |
| `!screenshot` | screenshot_commands.py | Capture webpage screenshot + VL analysis |
| `!janitor` | janitor_commands.py | Cache/media janitor controls (run, status, configure) |
| `!archive-setup` | archive_commands.py | Set up server archive |
| `!archive-search` | archive_commands.py | Search server archive |
| `!context-privacy` | context_commands.py | Control conversation context privacy |
| `!memory-add` | memory_cmds.py | Add a curated memory |
| `!memory-list` | memory_cmds.py | List memories (scoped) |
| `!memory-delete` | memory_cmds.py | Delete a specific memory |
| `!memory-wipe` | memory_cmds.py | Wipe memories (scoped) |
| `!memory-search` | memory_cmds.py | Semantic memory search |
| `!memory-review` | memory_cmds.py | Review curated memories with approve/reject |
| `!memory-forget` | memory_cmds.py | Remove memories for a user |
| `!memory-export` | memory_cmds.py | Export memories to file |
| `!server-memory-add` | memory_extended_cmds.py | Add server-scoped memory |
| `!server-memory-del` | memory_extended_cmds.py | Delete server-scoped memory |
| `!server-memory-wipe` | memory_extended_cmds.py | Wipe server memories |
| `!server-memory-search` | memory_extended_cmds.py | Search server memories |

### Slash Commands (/)

| Command | File | Description |
|---------|------|-------------|
| `/image` | vision_commands.py | Generate images from text prompts |
| `/imgedit` | vision_commands.py | Image editing (edit prompt + reference image) |
| `/video` | vision_commands.py | Generate videos from text prompts |
| `/vidref` | vision_commands.py | Video with reference image |
| Image upgrade | image_upgrade_commands.py | Upscale/enhance images |
| IMG commands | img_commands.py | Image generation (prefix-based) |

---

## 6. Platform/Service Inventory

### LLM Providers (Text)

**Primary: OpenRouter**
- Transport: HTTP POST to `https://openrouter.ai/api/v1/chat/completions`
- Auth: Bearer token (OPENAI_API_KEY reused via OPENAI_API_BASE override)
- Default model: `deepseek/deepseek-chat-v3-0324:free`
- Fallback ladder: `deepseek-r1-0528:free` -> `deepseek-chat-v3-0324:free` -> `glm-4.5-air:free`
- Timeouts: 20s, 25s, 30s per model in ladder
- Client: async OpenAI-compatible client with retry and timeout handling

**Secondary: Ollama**
- Transport: HTTP POST to `http://localhost:11434`
- Default model: `llama3`
- Used when TEXT_BACKEND=ollama

**Tertiary: NVIDIA NIM**
- Transport: HTTP POST to `https://integrate.api.nvidia.com/v1`
- Auth: NVIDIA_NIM_API_KEY
- Default model: `meta/llama3-70b-instruct`

### Vision-Language Models

- Transport: OpenAI-compatible API via OpenRouter
- Default model: `moonshotai/kimi-vl-a3b-thinking:free`
- Fallback ladder: kimi-vl-a3b -> mistral-small-3.2-24b -> mistral-small-3.2-24b (duplicate in default)
- Separate timeout ladder: 12s, 15s, 18s

### Image/Video Generation (Vision System)

**Together.ai**
- Transport: HTTP REST API
- Auth: VISION_API_KEY
- Models: FLUX Pro, SDXL variants
- Budget: $0.25/job, $5.00/day

**Novita.ai**
- Transport: HTTP REST API
- Auth: Same VISION_API_KEY
- Models: Qwen Image, SDXL variants

### Search

- **DuckDuckGo**: HTML scraping endpoint (`html.duckduckgo.com/html/`), no API key needed
- **Custom provider**: Configurable HTTP endpoint with API key, headers, and JSONPath result extraction
- Circuit breaker: 5 failures in window, 15s open, 25% half-open retry probability
- Connection pool: 10 max connections

### X/Twitter API

- Transport: HTTP REST API (api.twitter.com)
- Auth: Bearer token (X_API_BEARER_TOKEN) or OAuth2 app mode
- Hydration fields: tweet, media, user, poll, place, expansions
- Fallback: syndication (web scraping) on 5xx/429, strict on 401/403/404/410
- Timeout: 8000ms total, 5 retry attempts
- Circuit breaker: same pattern as search

### Screenshot API

- Primary: ScreenshotMachine (external API)
- Fallback: Playwright remote server (ws://localhost:3006)
- Configurable device, dimensions, format, cookies

### Playwright

- Remote server at `ws://localhost:3006` (Docker playwright:v1.59.1-noble)
- Used for: web extraction (Tier B/C), X syndication fallback, screenshot fallback
- No local browser fallback exists

### TTS Engine (Kokoro ONNX)

- Model: kokoro-v1.0.onnx (downloaded from GitHub releases)
- Voices: voices-v1.0.bin (multiple voice options)
- G2P: Misaki (English IPA) with espeak-ng fallback
- Output: WAV (primary) or OGG (via ffmpeg transcode)
- Phoneme validation: ensures first token matches IPA mapping
- Gibberish detection: retries on phoneme mismatch
- Timeout: 25s cold, 25s warm (configurable)

### STT Engine (faster-whisper)

- Model: medium-int8 (configurable via WHISPER_MODEL_SIZE)
- Multimodal fallback: OpenRouter API with whisper-large-v3, then Llama multimodal
- FFmpeg: audio decoding and format conversion
- YouTube: transcript extraction as fallback path
- Confidence threshold: 0.0 (accept all), 0.5 for multimodal fallback

### RAG (ChromaDB)

- PersistentClient at `./chroma_db`
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (local, default)
- OpenAI embeddings: text-embedding-3-small (alternative)
- Hybrid search: 0.7 vector weight + 0.3 keyword weight
- Confidence threshold: 0.7, fallback on low confidence
- Chunking: 512 tokens, 50 overlap, minimum 100
- Lazy loading: vector index loads on first search (non-blocking by default)
- Background indexing queue: 256 max, 2 workers, batch size 32

### Memory Persistence

- **SQLite**: `./data/memory.db` (WAL mode, RLock thread safety, soft deletes, TTL)
- **ChromaDB**: semantic similarity search in same chroma_db (0.85 dedup threshold)
- **Profiles**: JSON files in `user_profiles/` and `server_profiles/` (atomic writes)
- **Encrypted context**: Fernet-encrypted enhanced_context.json for multi-user tracking

### Server Archive

- SQLite: `./data/server_archive.db`
- Async ingestion queue (1000 max, batch size 100)
- Search: full-text search with configurable limit (default 10)
- Admin-only by default

### Prometheus Metrics

- HTTP server on port 8000 (mapped to 8010 in docker-compose)
- Metrics: request counts, latency histograms, error rates, TTS/STT/Vision stats
- NullMetrics fallback when disabled

---

## 7. Shared Utilities

| Module | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `bot/utils/env.py` | Safe env var parsing | env key, default | Typed value |
| `bot/utils/logging.py` | Dual-sink logging setup | -- | Logger instances |
| `bot/utils/logging_helper.py` | Structured log formatting | message, extra | Formatted output |
| `bot/utils/playwright_helpers.py` | Headless browser utils | URL, options | Page content/screenshot |
| `bot/utils/file_utils.py` | File operations | path, content | bool status |
| `bot/utils/external_api.py` | HTTP request helpers | URL, headers, timeout | Response data |
| `bot/utils/attachment_text.py` | Extract text from attachments | Discord attachment | Parsed text |
| `bot/utils/bounded_lru.py` | Bounded LRU cache | max_size | Cached results |
| `bot/utils/mention_utils.py` | Mention parsing/sanitization | Discord message content | Cleaned text |
| `bot/utils/torch_compat.py` | Torch compatibility | -- | Device detection |
| `bot/exceptions.py` | Exception hierarchy | -- | Custom exception classes |

**Exception Hierarchy:**
```
BotBaseException
├── ConfigurationError
├── BotError
│   ├── BackendError (timeout, rate limit, unavailable, all exhausted, API)
│   ├── InferenceError → VisionError
│   ├── MemoryError
│   ├── PersistenceError (corruption, atomic write)
│   ├── RAGIndexError
│   ├── PermissionDeniedError
│   ├── UrlSafetyError
│   └── CommandError
├── DispatchEmptyError
├── DispatchTypeError
├── TTSAudioError
└── FileProcessingError
```

---

## 8. Persistence & Schema

### SQLite -- Curated Memory Store (bot/memory/persistent_store.py)

**Tables/Schema:**
- `memories` -- id (PK), user_id, guild_id, channel_id, thread_id, text, category (enum), importance (float), created_at, expires_at, soft_delete (bool), source (explicit|inferred), metadata (JSON)
- Indexes on (user_id, guild_id), (category), (created_at DESC), (soft_delete, expires_at)
- WAL mode enabled
- RLock for thread safety across asyncio threads
- Dedup: normalized text match + semantic similarity (0.85 threshold)

### SQLite -- Server Archive (bot/server_archive/store.py)

**Tables/Schema:**
- `messages` -- id (PK), guild_id, channel_id, message_id, author_id, author_name, content, timestamp, attachments (JSON), is_bot (bool)
- Indexes on (guild_id, channel_id), (message_id), (timestamp DESC)

### ChromaDB -- RAG Vector Store

- Collection: "rag_documents" (default)
- Schema: document (text chunk), embedding (384-dim), metadata (source, user_id, guild_id, timestamp, document_id)
- L2 distance metric, converted to similarity: 1/(1+distance)

### ChromaDB -- Memory Semantic Store

- Collection: "curated_memories"
- Same embedding interface as RAG (shared SentenceTransformer cache)
- Metadata sanitized to primitive types only

### File-Based Persistence

- **User profiles**: `user_profiles/<user_id>.json` -- in-memory cache + atomic file writes, Fernet encryption for enhanced context
- **Server profiles**: `server_profiles/<guild_id>.json` -- same pattern
- **RAG version tracking**: `runtime/rag_versions.json` -- SHA-256 hashes of knowledge base files
- **Vision jobs**: `vision_data/jobs/*.json` -- job state persistence
- **Vision ledger**: `vision_data/ledger.jsonl` -- cost tracking
- **Context files**: `context.json`, `enhanced_context.json` -- conversation state
- **TTS cache**: `cache/tts/` -- 7-day purge cycle

---

## 9. Background Jobs & Scheduled Tasks

| Job | Trigger | Frequency | Description |
|-----|---------|-----------|-------------|
| Context autosave | Timer | Every MEMORY_SAVE_INTERVAL (30s) | Persist conversation context to disk |
| Memory distiller | Timer | Every 900s (15 min) default | Mine server archive history for durable memories using keyword hint matching |
| Janitor cleanup | Timer | Configurable | Purge expired TTS cache, old screenshots, vision artifacts past TTL |
| RAG background indexing | Queue-triggered | Asynchronous | Process new documents uploaded to kb/ |
| Config file watcher | File system | Continuous | Monitor .env for changes, trigger hot-reload |
| Server archive sync | Event-driven | On message receipt | Queue messages for archival indexing |
| Memory ingestion queue | Queue-triggered | Asynchronous | Process curated memories for SQLite + ChromaDB storage |
| Prometheus metrics | HTTP | On request | Serve /metrics endpoint for scraping |

---

## 10. Middleware & Cross-Cutting Concerns

### Feature Gating (bot/router_components/gating.py)

All routes pass through feature gates before processing:
- TTS_ENABLED: gates TTS routes
- VISION_ENABLED: gates image/video generation
- ENABLE_RAG: gates RAG retrieval
- SERVER_ARCHIVE_ENABLED: gates archive access
- PERSISTENT_MEMORY_ENABLE: gates memory operations
- X_API_ENABLED: gates X API routes

### Circuit Breakers

Three implementations share the same pattern (failure window, open timeout, half-open retry):
- Search (SEARCH_BREAKER_*)
- X API (X_API_BREAKER_*)
- General HTTP (implied in external_api.py)

### Timeouts

All external I/O has bounded timeouts:
- LLM text: 20-30s per fallback model
- VL model: 12-18s per fallback model
- Search: 5-8s per tier
- X API: 8s total
- Playwright: 15s page timeout
- STT: 300s total deadline (audio can be long)
- TTS: 25s synthesis timeout
- Vision generation: 30s provider timeout
- Web extraction: Tier A=2s, Tier B=8s, Tier C=8s
- OCR: 240s global, 20s per batch

### Input Validation

- URL safety checking (bot/types/ and router)
- Content sanitization before LLM prompts
- Prompt injection defense: system prompts separate user content, URL content sanitized
- Mention sanitization via mention_utils (strips @-mentions from user content)

### Caching

- Config cache: 300s TTL, invalidated on hot-reload
- SentenceTransformer model cache: global singleton with per-model asyncio locks
- RAG versions: SHA-256 file hashes in JSON
- TTS audio: file-based cache with 7-day purge
- STT transcripts: 600s TTL default, 7 days for STT_CACHE_TTL_S
- Twitter/X: 86400s (24h) cache, 900s negative cache
- In-memory bounded LRU (bot/utils/bounded_lru.py)
- Readability cache: 14400s (4h) TTL for web content
- Chat completion request coalescing (single-flight)

### Logging

Dual-sink strategy (Section 2.5 of AGENTS.md):
- **Rich console sink**: timestamped, colored, with icons and local-time milliseconds
- **JSONL file sink**: `logs/bot.jsonl` -- structured with ts, level, name, subsys, guild_id, user_id, msg_id, event, detail fields
- Enforcer checks both handlers exist at startup; aborts if misconfigured
- Structured logging via `extra={"subsys": "...", "event": "..."}` pattern

---

## 11. Type System & Contracts

### InputModality (bot/modality.py)

Enum classifying message content types for routing decisions:
- TEXT, IMAGE, AUDIO, VIDEO, DOCUMENT, URL, VOICE, SCREENSHOT

### Command/ParsedCommand (bot/types/__init__.py)

- `Command` enum: CHAT, PING, HELP, STATUS, FEATURE, INDEX, SEARCH, TTS, TTS_ALL, SPEAK, SAY, MEMORY_*, RAG_*, ALERT, IMG, IGNORE
- `ParsedCommand` dataclass: (command, cleaned_content)

### ResponseMessage (bot/types/__init__.py)

- `ResponseMessage` dataclass: (text: str, audio_path: Path)
- `OutputModality` enum: TEXT, TTS

### Router Components Internal Contracts

- `RouteHandler` protocol: `can_handle(ctx) -> bool`, `handle(ctx) -> ResponseMessage`
- `RouterContext`: carries message, channel, guild, user, config, modality, attachments, URLs, conversation history, memories, RAG results
- X routing: TweetData, SyndicatedContent, ThreadData (internal dataclasses)

### Vision Types

- VisionJob: id, user_id, type (image|video|edit), prompt, status, provider, cost, artifact_path
- VisionBudgetEntry: timestamp, user_id, cost, job_type

### Memory Types

- MemoryRecord: record_id, text, category, importance, scope (user|server|channel|thread), source, created_at, expires_at
- Categories: user_preference, recurring_instruction, project_fact, server_fact, conversation_decision, correction, relationship_note, inside_joke, temporary_context

---

## 12. Configuration & Environment

~150+ environment variables control every aspect of the bot. Key groups:

**Required:**
- `DISCORD_TOKEN`: Bot token (REQUIRED)
- `PROMPT_FILE`: Path to system text prompt (REQUIRED, default: prompts/prompt-yoroi-super-chill.txt)
- `VL_PROMPT_FILE`: Path to VL system prompt (REQUIRED, default: prompts/vl-prompt.txt)

**Text backend:**
- `TEXT_BACKEND`: openai|ollama|nvidia (default: openai)
- `OPENAI_API_KEY`: API key for OpenRouter (REQUIRED when text_backend=openai)
- `OPENAI_API_BASE`: API endpoint (default: https://openrouter.ai/api/v1)
- `OPENAI_TEXT_MODEL`: Model name (default: deepseek/deepseek-chat-v3-0324:free)
- `TEXT_FALLBACK_MODELS`: Comma-separated fallback ladder
- `TEXT_FALLBACK_TIMEOUTS`: Per-model timeout seconds

**Vision:**
- `VISION_ENABLED`: Master toggle (default: true)
- `VISION_API_KEY`: Provider credential
- `VISION_ALLOWED_PROVIDERS`: together,novita
- `VISION_DEFAULT_PROVIDER`: together

**VL (Vision-Language):**
- `VL_MODEL`: Comma-separated model ladder (default: kimi-vl-a3b-thinking:free)

**TTS:**
- `TTS_BACKEND`: kokoro-onnx|kokoro|stub (default: kokoro-onnx)
- `TTS_VOICE`: Voice name (default: af)
- `TTS_TIMEOUT_S`: Synthesis timeout (default: 25.0)

**STT:**
- `STT_ENABLE`: Global toggle (default: true)
- `STT_MODE`: single|cascade|parallel|hybrid (default: single)
- `STT_ACTIVE_PROVIDERS`: local_whisper
- `WHISPER_MODEL_SIZE`: faster-whisper model (default: medium-int8)

**RAG:**
- `ENABLE_RAG`: Enable RAG (default: true)
- `RAG_DB_PATH`: ChromaDB path (default: ./chroma_db)
- `RAG_KB_PATH`: Knowledge base dir (default: kb)
- `RAG_EAGER_VECTOR_LOAD`: Load at startup (default: true)

**Memory:**
- `PERSISTENT_MEMORY_ENABLE`: Enable memory (default: true)
- `PERSISTENT_MEMORY_SQLITE_PATH`: SQLite path (default: ./data/memory.db)
- `PERSISTENT_MEMORY_TOP_K`: Results to fetch (default: 6)

**Server Archive:**
- `SERVER_ARCHIVE_ENABLED`: Enable archive (default: false in .env.example, true in practice)
- `SERVER_ARCHIVE_DB_PATH`: Archive SQLite (default: ./data/server_archive.db)

**Search:**
- `SEARCH_PROVIDER`: ddg|custom (default: ddg)
- `SEARCH_MAX_RESULTS`: Max results (default: 5)

**X API:**
- `X_API_ENABLED`: Enable X API (default: false)
- `X_API_BEARER_TOKEN`: Bearer token

**Networking:**
- `HTTP2_ENABLE`: Enable HTTP/2 (default: true)
- `HTTP_MAX_CONNECTIONS`: Connection limit (default: 64)
- `HTTP_CONNECT_TIMEOUT_MS`: 1500ms
- `HTTP_TOTAL_DEADLINE_MS`: 6000ms

**Playwright:**
- `PW_SERVER_URL`: Remote server endpoint (default: ws://localhost:3006)

**Bot Behavior:**
- `BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO`: Silence gate (default: true)
- `REQUIRE_MENTION_IN_GUILDS`: Require mention (default: true)
- `COMMAND_PREFIX`: Prefix char (default: !)
- `TEMPERATURE`: LLM temperature (default: 0.7)
- `TIMEOUT`: Response timeout (default: 120.0)

---

## 13. CI/CD Pipeline

**Single GitHub Actions workflow:** `.github/workflows/ci.yml`

**Triggers:**
- Push: main, master
- Pull request: main, master
- Concurrency: cancel-in-progress per branch

**Jobs (sequential, ubuntu-latest, Python 3.11):**

1. `uv sync --dev` -- Install dependencies via uv package manager
2. `uv run ruff check .` -- Lint
3. Import checks: `import bot`, `from bot.core.bot import LLMBot`, `from bot.router import Router`
4. `uv run pytest -q --tb=short` -- Run test suite
5. Config validation smoke: Loads config with fake tokens, asserts COMMAND_PREFIX present

**Characteristics:**
- Fast pipeline (single job, no matrix)
- No integration tests against live services
- No security scanning (bandit/semgrep not in CI)
- Pytest markers defined (integration) but CI runs everything
- No docker/compose validation

---

## 14. Security Posture

### Authentication
- Discord: Bot token in .env, never committed
- API keys: OpenAI, NVIDIA, Vision, X, Whisper, ScreenshotMachine -- all in .env with [REDACTED] in examples
- Admin commands: Owner IDs, alert admin user IDs restrict sensitive operations

### Input Security
- URL safety checking at entry points
- Prompt injection defense: system prompts are separate from user content
- @-mention sanitization strips user mentions from content
- Content size limits: MAX_FILE_SIZE (2MB), MAX_ATTACHMENT_SIZE_MB (25MB)
- Vision safety filter (configs/vision_policy.json)

### Secret Handling
- .env file never committed (gitignored)
- Logging strips secrets via enforcer
- Fernet encryption for enhanced context storage
- Token scrub in log messages

### Network Security
- Docker with no-new-privileges:true
- Non-root user (UID 1026) in container
- Bounded timeouts on all HTTP calls
- Circuit breakers on flaky services
- Connection pooling with max limits

### Known gaps
- No HTTPS enforcement for webhook endpoints
- No rate limiting on command invocation (relies on Discord rate limits)
- .env file readable by container user (no chmod 600 enforcement)
- No CSRF protection (not applicable for bot, but Flask metrics endpoint has no auth)

---

## 15. Known Patterns & Conventions

**File organization:**
- `bot/` contains all application code
- `tests/` mirrors `bot/` structure where practical
- `utils/` for tools and scripts
- `docs/` for documentation
- `configs/` for JSON configuration files
- `prompts/` for system prompt files

**Naming conventions:**
- snake_case for functions, variables, files
- PascalCase for classes
- Constants in UPPER_SNAKE_CASE
- Exception hierarchy follows single-chain inheritance, no diamond MRO
- Module-level `__all__` for public API surface

**Async patterns:**
- async/await throughout the event loop
- `asyncio.to_thread()` for blocking operations (file I/O, ML inference)
- Thread pools for CPU-bound work (PDF parsing, embeddings)
- Bounded async queues for backpressure

**Error handling:**
- Structured logging with subsys/event metadata
- Typed exceptions per boundary (bot/exceptions.py)
- Retry with exponential backoff for transient I/O
- Graceful fallback chains at every external integration point
- Fail-open on compatibility shims, fail-closed on configuration errors

**Testing patterns:**
- pytest with asyncio_mode=auto
- Mock/fake tokens for CI
- Unit tests for individual components
- Integration tests marked with @pytest.mark.integration
- Conftest.py provides shared fixtures

---

## 16. Dependency Map

**Core framework:**
- discord.py 2.6.3 -- Discord bot framework
- aiohttp 3.12.15 -- Async HTTP client
- openai 1.107.0 -- OpenAI-compatible API client (used for OpenRouter, NVIDIA NIM)
- httpx[http2] 0.28.1 -- HTTP/2 client for specific integrations

**ML/AI:**
- torch 2.3.1 -- PyTorch (TTS, embeddings)
- onnxruntime 1.22.1 -- ONNX inference (Kokoro TTS)
- sentence-transformers 5.1.0 -- Embedding models (RAG, memory)
- faster-whisper 1.2.0 -- STT engine
- chromadb 1.0.20 -- Vector database
- kokoro-onnx >= 0.4.9 -- Kokoro TTS engine
- misaki 0.9.4 -- G2P for TTS
- phonemizer 3.3.0 -- Phoneme extraction
- g2p-en 2.1.0 -- English G2P
- cmudict 1.1.1 -- Pronunciation dictionary

**Web/content:**
- playwright 1.58.0 -- Headless browser automation
- trafilatura 2.0.0 -- Web content extraction
- beautifulsoup4 4.13.5 -- HTML parsing
- requests 2.32.5 -- Sync HTTP (fallbacks)
- yt-dlp 2024.10.22 -- YouTube video/audio download
- fake-useragent 2.2.0 -- User agent rotation

**Document processing:**
- pdf2image 1.17.0 -- PDF to images
- pymupdf 1.26.3 -- PDF processing
- pypdf2 3.0.1 -- PDF parsing
- pytesseract 0.3.13 -- OCR
- python-docx 1.1.2 -- DOCX parsing
- ebooklib 0.19 -- EPUB/MOBI parsing
- reportlab 4.4.3 -- PDF generation

**Infrastructure:**
- flask 3.1.2 -- Metrics HTTP server
- prometheus-client 0.22.1 -- Metrics export
- rich 14.1.0 -- Console formatting
- python-dotenv 1.1.1 -- .env loading
- cryptography 45.0.7 -- Fernet encryption
- cachetools 6.2.0 -- LRU caching
- psutil 7.0.0 -- System monitoring

**Audio:**
- pydub 0.25.1 -- Audio manipulation
- soundfile 0.13.1 -- Audio file I/O
- ffmpeg-python >= 0.2.0 -- FFmpeg bindings
- ffprobe >= 0.5 -- Media probing
- scipy 1.16.1 -- Audio signal processing

**Testing:**
- pytest 8.4.1
- pytest-asyncio 0.23.8
- mypy >= 1.19.1 -- Type checking

---

## 17. Edge Cases & Operational Notes

### Synology NAS deployment
- Docker user UID 1026 matches host user (pry) on Synology
- Playwright on separate Docker container (port 3006)
- Volume mounts for persistent data (context, chroma_db, profiles, logs, cache)
- Resource limits: 8GB max, 2GB reserved
- Logs rotate: 100MB max per file, 5 files
- Health check: curl /metrics every 30s

### NFS mount considerations
- Repository lives on NFS mount (/mnt/nasirjones/)
- SQLite WAL mode helps with NFS reliability
- Chromium browser in isolated Docker container (not on NFS)

### Large codebase concerns
- `bot/core/bot.py`: 3100 lines -- exceeds 300 SLOC guideline by 10x
- `bot/router_components/x_routing.py`: Massive, 1100+ exports
- `bot/commands/admin_alert_commands.py`: 1408 lines
- `bot/memory/curator.py`: 990 lines
- `bot/config.py`: 1026 lines (should be decomposed)
- `bot/tts/kokoro_direct.py`: 1071 lines
- These are maintainability risks per AGENTS.md §7 (CSD gate at 300 SLOC)

### Memory management
- ChromaDB can grow unbounded without periodic compaction
- Vision artifacts have 7-day TTL but enforcement relies on janitor
- SQLite WAL mode requires periodic checkpointing for disk space
- Context files can grow large with long conversations (MAX_CONTEXT_MESSAGES=10 limits this)

### Known failure modes
- Playwright server death causes all web extraction to fail (no local fallback)
- ChromaDB startup can be slow with large collections (lazy loading mitigates)
- Kokoro ONNX model cold start exceeds TTS timeout on first synthesis
- SentenceTransformer model download blocks startup on first run
- DDG HTML endpoint may block scraping (IP-based rate limiting)
- Vision budget enforcement relies on provider response parsing (may be inaccurate)

### Cache directories
- `cache/media_probes/` -- URL probe results
- `cache/screenshots/` -- Screenshot captures
- `cache/stt_pcm/` -- STT audio files
- `cache/stt_transcripts/` -- STT text output
- `cache/tts/` -- TTS audio cache (7-day purge)
- `cache/video_audio/` -- Extracted audio from video
- `cache/youtube_transcripts/` -- YouTube transcript cache
- `.kokoro_espeak_tmp/` -- Temporary Kokoro/espeak files

---

## 18. Testing Posture

**Framework:** pytest with `asyncio_mode = auto`

**Test count:** ~160+ test files across multiple directories:
- `tests/core/` -- Bot class, router, command parser, inline search, multimodal sequence
- `tests/commands/` -- Archive, memory, operator, RAG, screenshot, TTS commands
- `tests/memory/` -- Archive distiller, curated memory, memory service, mention context, persistence
- `tests/router/` -- Image URL detection, router components, X API routing, vision triggers
- `tests/tts/` -- Kokoro engine, text preprocessing, output, registry
- `tests/vision/` -- Money/pricing, router integration, error handling
- `tests/stt_pipeline/` -- FFmpeg runtime, lifecycle, transcribe flow, stitch, URL ingest
- `tests/server_archive/` -- Service and store tests
- `tests/syndication/` -- Extract policy, handler VL cap
- `tests/backend/` -- OpenAI client retry, vision ladder fallback
- `tests/scripts/` -- Smoke tests for bot, env, TTS
- `tests/voice/` -- Voice publisher
- `tests/decision/` -- Decision helpers

**Strengths:**
- Comprehensive coverage of router, TTS, STT pipeline, and memory systems
- Regression tests for specific production errors (gibberish detection, tokenizer alias, etc.)
- Integration-level testing for media ingestion and syndication
- Good coverage of failure paths (retry, fallback, timeout)

**Gaps:**
- No explicit `integration` marker used in CI (CI runs all tests, no --ignore-marker)
- No tests for RAG chunking/parsers (despite complex logic)
- No tests for config hot-reload flow (partial: test_config_reload.py exists)
- No tests for Prometheus metrics behavior
- No E2E tests (real Discord gateway integration impossible in CI)
- Playwright-dependent tests skipped in CI (no browser available)

**Notable:** Test suite includes regression tests for many historical bugs (vl exhaustion, multimodal fixes, Kokoro IPA, stt chunk ordering, twitter modality, etc.), indicating a history of production incidents being caught and fixed.

---

## 19. Top Priorities / Recommendations

### Critical

1. **Decompose oversized modules** -- `bot/core/bot.py` (3100 LOC), `bot/config.py` (1026 LOC), `bot/router_components/x_routing.py` (1100+ exports), `bot/commands/admin_alert_commands.py` (1408 LOC), `bot/tts/kokoro_direct.py` (1071 LOC), `bot/memory/curator.py` (990 LOC). These violate the 300 SLOC gate and make refactoring risky.

2. **Remove build/ and MagicMock/ directories from repo** -- `build/` (build artifacts) and `MagicMock/` (pytest artifact dirs with thousands of nested directories) should be in .gitignore and cleaned up. These directories cause find/ls commands to hang and waste disk space.

3. **Add CI security scanning** -- Add `bandit -q -r bot` to CI pipeline. No static security analysis currently runs in CI despite the AGENTS.md requirement.

### High

4. **Vision model fallback ladder has duplicate entry** -- `VISION_FALLBACK_MODELS` defaults to `mistral-small-3.2-24b-instruct:free` twice. The second entry provides no additional resilience.

5. **Docker Compose health check uses unmapped port** -- Health check curls `localhost:8000` inside container, which is correct. Docker maps to 8010 externally. This is actually correct, but worth noting the external port is 8010 not 8000.

6. **No rate limiting on admin commands** -- The !alert command has a session timeout but no rate limiting. Admin commands should have per-user cooldowns.

7. **ChromaDB WAL/performance** -- With multiple collections (RAG, memory, semantic store), ChromaDB can become a bottleneck. Consider connection pooling or async batching.

### Medium

8. **Add integration test markers to CI** -- Configure CI to skip or separately run @pytest.mark.integration tests. Currently all tests run together with no differentiation.

9. **Document the router component contract** -- The decomposed router components (compose, gating, input_harvest, prompt_access, runtime) are well-designed but undocumented. Add ARCHITECTURE.md section explaining the protocol.

10. **TTS engine V8 path is incomplete** -- `bot/tts/engines/kokoro_v8.py` exists but the manager routes to kokoro-onnx by default. V8 engine should have clear feature flag or deprecation notice.

11. **STT multimodal fallback disabled by default** -- `STT_MULTIMODAL_FALLBACK_ENABLED=false` means STT failures have no recovery path besides the multimodal fallback which is disabled. Consider enabling or documenting the gap.

### Low

12. **Config 5-minute cache TTL** -- `load_config()` caches for 300s. Hot-reload invalidates the cache, but any code path that bypasses config_reload won't see fresh values. Document this contract.

13. **Remove deprecated env var warnings** -- Deprecation warnings for TEXT_MODEL and OPENAI_MODEL printed to stderr. Consider emitting these via logger.warning instead of print().

14. **Consolidate persistence** -- Server archive, memory store, and profiles all use SQLite/JSON independently. A unified persistence abstraction would simplify connection management and WAL mode handling.
