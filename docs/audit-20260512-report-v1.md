# PRODUCT AUDIT — Discord LLM Chatbot

## Date: 2026-05-12
## Version: v1
## Codebase: 244 Python files, 101,642 lines, 383 classes

---

## 1. Executive Summary

This is a production-grade Discord bot with comprehensive AI capabilities. It connects to Discord and acts as an intelligent conversational agent that can process text, images, audio, video, PDFs, and URLs. It features multi-backend AI support (OpenRouter, OpenAI, Ollama), a full RAG (Retrieval-Augmented Generation) system with ChromaDB, curated memory with semantic search, text-to-speech (TTS) with Kokoro engine, speech-to-text (STT) with Whisper, a content moderation pipeline for images, URL safety checking, and extensive observability (Prometheus metrics, SLO monitoring, resource monitoring, structured JSONL logging, Rich console).

The bot uses a message routing architecture where incoming Discord messages are classified and dispatched through pipelines that can invoke search (DuckDuckGo), web extraction, vision models (Novita, Together), text generation, and media processing. It operates in both DM and guild (server) contexts with per-user message queues, streaming status cards, and hot-reloadable configuration.

Key capabilities:
- **Text AI**: LLM-powered conversations via OpenRouter, OpenAI, or Ollama
- **Vision/Image**: Image analysis via vision-language models, screenshot capture, image generation (T2I)
- **Search & Web**: DuckDuckGo search, web extraction (httpx + Playwright fallback), YouTube transcripts
- **TTS**: Kokoro TTS engine for audio responses (including Discord native voice messages)
- **STT**: Whisper-based transcription of audio attachments, video URLs
- **Memory**: Curated memory system with semantic search, profiles, archival, and ingestion queue
- **RAG**: ChromaDB-backed document retrieval with hybrid search, auto-indexing
- **Syndication**: X/Twitter thread unrolling and content extraction
- **Server Archive**: SQLite-backed Discord message archiving and search

---

## 2. Architecture Overview

### Deployment Model
- Self-hosted on a Linux environment (Synology NAS observed)
- Runs as an async Python process: `uv run python -m bot.main`
- Uses Discord.py (discord-ext-commands) as the Discord client library
- Optional Docker container for Playwright (remote browser, port 3006)
- Prometheus metrics server (default port 8000)

### Tech Stack
- **Python 3.11+** (async-first, `asyncio`)
- **discord.py** (Discord API client with commands framework)
- **aiohttp** (HTTP client with connection pooling)
- **httpx** (alternate HTTP client in web extraction service)
- **ChromaDB** (RAG vector database)
- **Sentence-Transformers** (local embeddings for memory + RAG)
- **SQLite** (persistent store for memory, archive, atomic JSON)
- **Kokoro TTS** (local text-to-speech engine via ONNX)
- **Playwright** (remote browser for screenshots and web extraction)
- **Prometheus** (metrics export)

### Data Flow

```
User sends Discord message
  → LLMBot.on_message (gateway)
    → _dispatch_lock guard (dedup)
      → _get_user_queue / _ensure_user_processor (per-user serialization)
        → _process_single_message
          → context_manager.append (history)
          → enhanced_context_manager.append_message (multi-user context)
          → enqueue_inferred_memory (fire-and-forget memory curation)
          → router.dispatch_message()
            → Router class (149 methods) → BotAction
            → If delegated_to_cog: process_commands() → command handler
            → If has_payload: _execute_action()
              → TTS processing (if requested)
              → Voice publish (native Discord voice message)
              → Content sanitization
              → Reply/send with retry
```

### Directory Structure

| Directory | Purpose | Size |
|---|---|---|
| `bot/` | Core bot logic (orchestration) | ~30 files, root level |
| `bot/core/` | Bot class, startup, routing, monitoring | 13 files |
| `bot/commands/` | Discord command cogs (15 modules) | 15 files |
| `bot/memory/` | Curated memory system | 10 files |
| `bot/rag/` | RAG/document retrieval system | 8 files |
| `bot/tts/` | Text-to-speech (Kokoro engine) | 18 files |
| `bot/tts/engines/` | TTS engine implementations | 3 files |
| `bot/vision/` | Vision/image processing pipeline | 14 files |
| `bot/vision/providers/` | Vision model providers | 3 files |
| `bot/search/` | Search providers (DuckDuckGo) | 3 files |
| `bot/search/providers/` | Search backends | 1 file |
| `bot/stt_module/` | STT failure handling | 2 files |
| `bot/stt_pipeline/` | STT processing pipeline | 8 files |
| `bot/threads/` | X/Twitter thread unrolling | 1 file |
| `bot/voice/` | Discord voice message publishing | 1 file |
| `bot/utils/` | Utility modules | 6 files |
| `bot/util/` | Logging utility | 1 file |
| `bot/routing/` | Route handlers | 2 files (base, screenshot, unknown) |
| `bot/router_components/` | Router sub-components | 5 files |
| `bot/server_archive/` | Discord message archiving | 5 files |
| `bot/syndication/` | X/Twitter content syndication | 3 files |
| `bot/infra/` | Infrastructure helpers | 1 file |
| `bot/vl/` | Vision-language postprocessing | 1 file |
| `bot/metrics/` | Prometheus/noop metrics | 2 files |
| `bot/types/` | Extra type definitions | 1 file |

### Key Architectural Patterns

1. **Message Router Pattern**: All incoming messages pass through a `Router` class that classifies intent and produces a `BotAction`. Commands can be delegated to Discord.py's command processor.

2. **Per-User Serialization**: Messages from the same user are queued and processed serially via `asyncio.Queue` per user to prevent lockout.

3. **Circuit Breaker**: Both the OpenRouter client and HTTP client use circuit breaker pattern to handle provider failures gracefully.

4. **Hot-Reload**: Configuration can be reloaded at runtime via file watcher; TTS and vision components are hot-reinitialized.

5. **Fast-Path Routing**: Simple DM messages can skip heavy context loading, RAG, and modality detection for lower latency.

---

## 3. Startup & Bootstrap

### Entry Point
- `bot/__main__.py` → `runpy.run_module("bot.main")`
- `bot/main.py` → `run_bot()` → `asyncio.run(main_with_cleanup())` → `main()`

### Startup Sequence (bot/main.py)

1. **Logging init**: `init_logging()` sets up Rich console + JSONL file logging
2. **CLI parsing**: `parse_arguments()` supports `--version`, `--config-check`, `--debug`
3. **Venv check**: `check_venv_activation()` validates virtual environment
4. **Config reload system**: `setup_config_reload()` + `start_file_watcher()` for hot-reload
5. **Config loading**: `load_config()` + `load_system_prompts()`, then `run_pre_flight_checks()`
6. **Bot instantiation**: `LLMBot(config, command_prefix=get_prefix, intents=intents)`
7. **Signal handlers**: `setup_signal_handlers(bot)` for graceful shutdown
8. **Background tasks**: `spawn_background_tasks(bot)` before connecting
9. **File watcher**: `await start_file_watcher()` for config hot-reload
10. **Discord connect**: `bot.start(token)` with 3-retry loop

### Bot setup_hook() sequence (bot/core/bot.py)

1. **Metrics init**: Prometheus or Noop based on `PROMETHEUS_ENABLED`
2. **Gate counter registration**: Pre-register all Prometheus counters for SSOT gate, X syndication, vision routing
3. **System prompts**: `load_system_prompts()`
4. **Config reload callback**: Register `_on_config_reload()` for hot-reload (TTS, vision, HTTP client)
5. **Profiles**: `load_profiles()` (user/server profiles)
6. **Background tasks**: `setup_background_tasks()`
7. **TTS**: `setup_tts()` → creates TTSManager
8. **Router**: `setup_router()` → creates Router instance
9. **RAG**: `setup_rag()` → initializes ChromaDB backend
10. **Extensions**: `load_extensions()` → loads command cogs
11. **Error handler**: `setup_command_error_handler()`
12. **Public output safety hooks**: Monkey-patches `Messageable.send`, `Message.reply`, `Message.edit`, etc.

### Pre-Flight Checks (bot/core/startup.py)

1. Discord token validation (SHA-256 hash logged, token not exposed)
2. Intent verification (message_content, guilds, messages, voice_states)
3. Discord.py version logging
4. Playwright browser check (local binary or remote PW_SERVER_URL validation)

---

## 4. Complete Route/Command Map

The bot does NOT expose HTTP routes — it's a Discord bot. Commands use Discord.py's `@app_commands.command` and `@commands.command` decorators.

### Registered Command Cogs

| Cog | File | Commands/Approx |
|---|---|---|
| **AdminAlertCommands** | `commands/admin_alert_commands.py` | Alert session management, destination config, notification routing |
| **ArchiveCommands** | `commands/archive_commands.py` | Archive search, sync, status, guild/channel management |
| **ConfigCommands** | `commands/config_commands.py` | View/edit runtime config, reload config |
| **ContextCommands** | `commands/context_commands.py` | Context management, history inspection |
| **ImageUpgradeCommands** | `commands/image_upgrade_commands.py` | Image quality upgrade operations |
| **ImgCommands** | `commands/img_commands.py` | Image generation commands (T2I) |
| **JanitorCommands** | `commands/janitor_commands.py` | File cleanup, directory management |
| **MemoryCommands** | `commands/memory_cmds.py` | Most of the memory system commands (add, delete, list, profile, etc.) |
| **ExtendedMemoryCommands** | `commands/memory_extended_cmds.py` | Additional memory operations |
| **OperatorCommands** | `commands/operator_commands.py` | Operator-level diagnostics, metrics, task management |
| **RAGCommands** | `commands/rag_commands.py` | RAG bootstrap, scan, wipe, search, document management |
| **ScreenshotCommands** | `commands/screenshot_commands.py` | Capture webpage screenshots |
| **SearchCommands** | `commands/search_commands.py` | Web search commands |
| **TestCommands** | `commands/test_cmds.py` | Testing/diagnostics commands |
| **TTSCommands** | `commands/tts_cmds.py` | TTS status, engine selection, voice settings |
| **VideoCommands** | `commands/video_commands.py` | Video processing/transcription commands |
| **VisionCommands** | `commands/vision_commands.py` | Vision pipeline management, model selection |

### Message Processing Routes (non-command)

The `Router` class (bot/router.py, 149 methods) handles all non-command messages:

- **Text → LLM**: Plain DM/server text routed to OpenRouter/OpenAI/Ollama backend
- **URLs → Web extraction**: URLs processed via `WebExtractionService` (httpx → Playwright fallback)
- **Images → Vision**: Image attachments sent to vision providers (Novita, Together)
- **Audio → STT → LLM**: Audio attachments transcribed, then processed as text
- **YouTube URLs → Transcript**: YouTube transcripts extracted and injected into context
- **X/Twitter URLs → Syndication**: Tweet content extracted via syndication endpoints
- **PDFs → RAG/Extraction**: PDFs parsed, text extracted, summarized
- **Videos → STT**: Video files processed for audio tracks, transcribed

---

## 5. Platform/Service Inventory

### AI/LLM Providers
- **OpenRouter**: Primary LLM API (via `OptimizedOpenRouterClient` with circuit breaker, connection pooling, retry with backoff, model fallback)
- **OpenAI**: Direct OpenAI API backend (`bot/openai_backend.py`)
- **Ollama**: Local model serving (`bot/ollama.py`, `OllamaClient`)

### Vision Providers
- **Novita**: Vision model provider (`vision/providers/novita_adapter.py`)
- **Together**: Vision model provider (`vision/providers/together_adapter.py`)
- Unified adapter supports plugin pattern: Together, Novita, OpenRouter, Nvidia

### Search
- **DuckDuckGo HTML**: `DDGSearchProvider` — uses `html.duckduckgo.com/html/` endpoint
  - Transport: HTTP request via shared http_client
  - Parsing: BeautifulSoup/lxml for HTML results
  - Categories: General, news, images support
  - Rate limiting: Bounded via http_client's per-host limits

### TTS
- **Kokoro**: Primary TTS engine, runs locally via ONNX
  - Two engine variants: `KokoroONNXEngine` and `KokoroV8Engine`
  - Direct mode: `KokoroDirect` / `KokoroDirect_fixed` (31 methods in fixed version)
  - G2P (grapheme-to-phoneme): `eng_g2p_local.py`, IPA vocab loader
  - Fallback stub engine for when Kokoro is unavailable
  - Thread-pooled synthesis (doesn't block event loop)

### STT
- **Whisper**: Local Speech-to-Text
  - Multiple model specs configurable
  - RAM-guard system to prevent OOM (STTRAMGuard)
  - PCM stream handling: `FfmpegPCMStream`, `CachedPCMStream`
  - Failure classifier for diagnosing STT issues
  - Multimodal fallback provider

### Web Extraction
- **Tier A**: httpx (fast path, `WebExtractionService`)
- **Tier B**: Playwright (remote browser, `PW_SERVER_URL`)
- **Fallback**: curl-based extraction on Synology NAS (memory note)

### Media Processing
- **MediaIngestionManager**: Handles audio/video file validation, extraction
  - File type detection via `AttachmentClassifier` + `MediaCapabilityDetector`
  - FFmpeg for audio/video processing
- **PDFProcessor**: PDF text extraction, OCR support
- **VideoIngestionManager**: 30 methods, handles download, audio extraction
- **YouTubeTranscriptService**: Extracts YouTube video transcripts

### Discord
- **Discord.py**: Primary transport (discord.py library)
- **VoiceMessagePublisher**: Posts native Discord voice messages (opus format)
- **VoiceMemoSender**: Infrastructure helper for voice delivery

### Metrics/Observability
- **Prometheus**: HTTP server on port 8000, comprehensive counter/gauge/histogram metrics
- **SLOMonitor**: Tracks latency, error rate, uptime against targets
- **ResourceMonitor**: Event loop monitoring, memory/CPU tracking
- **BackgroundTaskMonitor**: Heartbeat monitoring for long-running tasks
- **Structured JSONL logging**: Dual-sink (Rich console + JSONL file)

---

## 6. Shared Utilities

### Core Cross-Cutting Utilities

| Module | Purpose | Key Functions |
|---|---|---|
| `bot/retry_utils.py` | Retry with backoff | `RetryConfig`, `EnhancedRetryManager` with circuit breakers per provider |
| `bot/enhanced_retry.py` | Multi-provider retry ladder | `ProviderStatus`, `CircuitBreakerState`, provider ordering from env |
| `bot/http_client.py` | Shared HTTP client (aiohttp) | `SharedHttpClient` with connection pooling, circuit breaker, per-host limits, single-flight patterns |
| `bot/request_coalescing.py` | Request dedup | `RequestCoalescer` — prevents duplicate concurrent requests for same input |
| `bot/single_flight_cache.py` | Single-flight + LRU cache | `SingleFlightCache` with `LRUCache`, metrics, TTL |
| `bot/concurrency_manager.py` | Task pool management | `BoundedExecutionPool`, `CancellationTree`, `PoolType` |
| `bot/retry_utils.py` | Retry config and execution | Retry config for different provider types |
| `bot/budget_manager.py` | Token/budget tracking | `BudgetManager`, `BudgetExecution`, soft/hard deadline enforcement |
| `bot/action.py` | Bot action data class | `BotAction` — content, embeds, files, audio_path, metadata |
| `bot/evidence.py` | Evidence collection | `EvidenceBundle` for debug/audit evidence |
| `bot/public_output.py` | Content sanitization | `sanitize_public_text`, `sanitize_embed_for_public` — prevents leaking internal info |
| `bot/exceptions.py` | Exception hierarchy | ~20 exception types: `ConfigurationError`, `BackendError`, `BackendTimeoutError`, `BackendRateLimitError`, `AllProvidersExhaustedError`, `InferenceError`, `MemoryError`, `PersistenceError`, `CorruptionError`, `RAGIndexError`, `PermissionDeniedError`, `UrlSafetyError`, `CommandError` |
| `bot/atomic_json.py` | Atomic JSON file writes | Concurrency-safe JSON read/write with file locking and fallback |
| `bot/url_safety.py` | URL validation | DNS resolution, scheme validation, redirect following, safety gate (183+ lines) |
| `bot/url_classifier.py` | URL type detection | Classifies URLs by service (YouTube, X/Twitter, etc.) |
| `bot/attachment_classifier.py` | Attachment classification | `AttachmentBucket`, `ClassifiedAttachment` |
| `bot/tokenizer_registry.py` | Tokenizer management | `TokenizerRegistry` with 14 methods for multiple tokenizer backends |
| `bot/modality.py` | Input modality detection | `InputModality`, `InputItem`, `ImageRef` |
| `bot/types.py` | Core types | `Command`, `ParsedCommand`, `ResponseMessage`, `OutputModality` |
| `bot/controller.py` | Multimodal pipeline controller | `hybrid_pipeline` orchestrating text→speak+see+hear |
| `bot/brain.py` | LLM text generation | `brain_infer` — sends text to LLM backend |
| `bot/speak.py` | TTS generation | `speak_infer` — generates audio from text |
| `bot/see.py` | Vision analysis | `see_infer` — analyzes images with vision models |
| `bot/hear.py` | STT transcription | `hear_infer` — transcribes audio to text |
| `bot/context.py` | Message context | Context building for LLM prompts |
| `bot/contextual_brain.py` | Context-aware LLM | `contextual_brain_infer` with context injection |
| `bot/decision_helpers.py` | Scope/permission checks | `ScopeResult`, decision-making helpers |
| `bot/command_parser.py` | Command parsing | `command_parser` for extracting commands from text |
| `bot/janitor.py` | File cleanup | `Janitor` — directory policy enforcement, file cleanup |
| `bot/tasks.py` | Background task manager | `TaskManager` — startup background tasks |
| `bot/shutdown.py` | Graceful shutdown | `GracefulShutdown` — signal handling, cleanup |
| `bot/edit_coalescer.py` | Message edit coalescing | `EditCoalescer` — prevents rapid message edits |
| `bot/result_aggregator.py` | Result aggregation | `ResultAggregator` — combines multi-source results |
| `bot/multimodal_retry.py` | Multimodal retry | Retry logic for vision/audio failures |
| `bot/server_features.py` | Server feature flags | Feature toggle management |
| `bot/config_reload.py` | Config hot-reload | File watcher + atomic config swap |
| `bot/config.py` | Configuration | `load_config()`, `load_system_prompts()`, ~187 lines of defaults/env vars |
| `bot/env_validator.py` | Environment validation | Validates required env vars at startup |
| `bot/env_utils.py` | Env utilities | Helper functions for environment detection |
| `bot/nvidia_backend.py` | Nvidia-specific backend | GPU detection, CUDA setup |

### Vision System Utilities

| Module | Purpose |
|---|---|
| `bot/vision/money.py` | `Money` class (26 methods) — financial amount handling for budget tracking |
| `bot/vision/pricing_loader.py` | `PricingTable` (11 methods) — model pricing data |
| `bot/vision/safety_filter.py` | `VisionSafetyFilter` (11 methods) — content filtering for vision |
| `bot/vision/provider_usage_parser.py` | `ProviderUsageParser` (11 methods) — parses usage/token data from providers |
| `bot/vision/job_store.py` | `VisionJobStore` (11 methods) — job persistence |
| `bot/vision/job_watcher.py` | `JobWatcherRegistry` (9 methods) — monitors active vision jobs |
| `bot/vision/artifact_cache.py` | `VisionArtifactCache` (18 methods) — caches vision analysis results |
| `bot/vision/intent_router.py` | `VisionIntentRouter` (15 methods) — classifies vision intent |

### Logging/Utils

| Module | Purpose |
|---|---|
| `bot/utils/logging.py` | `JsonlFormatter`, `SensitiveDataFilter`, `LevelIconFilter` |
| `bot/utils/mention_utils.py` | Mention formatting utilities |
| `bot/utils/file_utils.py` | File handling helpers |
| `bot/utils/bounded_lru.py` | Bounded LRU cache |
| `bot/utils/external_api.py` | External API helpers |
| `bot/utils/attachment_text.py` | Text extraction from attachments |
| `bot/utils/playwright_helpers.py` | Playwright utility functions |
| `bot/utils/torch_compat.py` | PyTorch compatibility shims |
| `bot/utils/env.py` | Environment variable helpers |
| `bot/util/logging.py` | Additional logging utilities |
| `bot/utils/logging_helper.py` | Logging helper functions |

---

## 7. Database Schema

### Persistent Memory Store (SQLite)
**File**: `bot/memory/persistent_store.py` (`PersistentMemoryStore`, 31 methods)

- `memory_records` table: `id`, `user_id`, `text`, `category`, `created_at`, `source_message_id`, `metadata`, `profile_id`
- `memory_profiles` table: Profile definitions for curated memory
- Full-text search support via SQLite FTS
- CRUD operations: create, read, update, delete, list memories
- Ownership verification: `delete_memory` checks `record.user_id`
- Prefix matching for partial-ID lookups

### Semantic Store (SQLite)
**File**: `bot/memory/semantic_store.py` (`CuratedMemorySemanticStore`, 9 methods)

- Embedding-backed semantic search
- Uses SentenceTransformer for embeddings
- Vector similarity search over memory records

### Server Archive (SQLite)
**File**: `bot/server_archive/store.py` (`ServerArchiveStore`, 48 methods)

Models (`bot/server_archive/models.py`):
- `ArchiveGuild`: Guild metadata
- `ArchiveChannel`: Channel metadata
- `ArchiveThread`: Thread metadata
- `ArchiveUser`: User metadata
- `ArchiveAttachment`: Attachment records
- `ArchiveMessage`: Message records (content, author, timestamp, channel)
- `ArchiveMention`: Mention records
- `ArchiveMessageBundle`: Bundled message groups
- `ArchiveSyncState`: Sync state tracking
- `ArchiveSearchResult`: Search result DTOs

Service (`bot/server_archive/service.py`): `ServerArchiveService` (16 methods)
- Ingestion queue (`bot/server_archive/ingestion_queue.py`): `ArchiveIngestionQueue` (6 methods)

### RAG/ChromaDB
**File**: `bot/rag/chroma_backend.py` (`ChromaRAGBackend`, 12 methods)

- ChromaDB persistent collection for document storage
- Embedding via `EmbeddingInterface` (SentenceTransformer or OpenAI)
- Vector search + full-text hybrid search
- Indexing queue (`bot/rag/indexing_queue.py`): `IndexingQueue` (10 methods)
- Bootstrap (`bot/rag/bootstrap.py`): `RAGBootstrap` (9 methods)

### Config (JSON files)
- `context.json`: Context manager state (per conversation)
- `enhanced_context.json`: Enhanced context manager state (per user/message)
- `atomic_json.py`: Atomic write helper for all JSON-based storage

---

## 8. Background Jobs & Scheduled Tasks

### Managed by `TaskManager` (bot/tasks.py, 7 methods)

Background tasks spawned via `spawn_background_tasks(bot)`:

1. **Memory Save Task** (`memory_save_task`): Periodic persistence of memory state
2. **Memory Distiller** (`start_memory_distiller`): Background task for distilling archived memories
3. **Memory Ingestion Queue Worker**: Processes queued memory curation requests
4. **RAG Indexing Queue Worker**: Processes queued document indexing
5. **Server Archive Sync Worker**: Syncs Discord messages to archive store
6. **Resource Monitor**: Periodic CPU/memory/loop monitoring
7. **SLO Monitor**: Ongoing SLO compliance tracking
8. **Janitor**: File cleanup and directory policy enforcement

### BackgroundTaskMonitor
**File**: `bot/core/background_task_monitor.py` (`BackgroundTaskMonitor`, 14 methods)

- Heartbeat-based monitoring of long-running tasks
- Restart policies for failed tasks
- Task state machine: PENDING → RUNNING → SUCCESS/FAILED
- Per-component configurable timeouts and retry counts

---

## 9. Middleware & Cross-Cutting Concerns

### Message Processing Pipeline

**Pre-processing (LLMBot level)**:
1. **Dedup guard**: `_processed_messages` OrderedDict prevents double-processing
2. **Per-user serialization**: `_user_queues` ensure one message at a time per user
3. **Typing indicator**: `_optional_typing()` context manager (with rate-limit suppression)

**Security/Safety**:
1. **Public output sanitization** (bot/public_output.py): Monkey-patches all Discord send/edit boundaries. Sanitizes content, embeds before they reach Discord. Prevents info leaks.
2. **URL safety** (bot/url_safety.py): Validates URL scheme (http/https only), DNS resolution, redirect following, safety gate.
3. **Permission checks** (bot/core/permissions.py): Owner ID verification, role-based access
4. **Attachment classification**: Identifies file types and capabilities before processing

### Cross-Cutting:

| Concern | Implementation |
|---|---|
| **Logging** | Dual-sink: Rich console (human-readable) + JSONL file (machine-parseable). Structured with `event`, `subsys` labels. `LoggingEnforcer` ensures compliance. |
| **Metrics** | Prometheus counters/gauges/histograms defined in `Metrics` base class. `define_counter()` pattern for counter registration. Null/Noop fallback when disabled. |
| **Error Handling** | Hierarchical exception classes. Per-module error handling. `CommandErrorHandler` for Discord command errors. |
| **Retry** | Multi-layer: `retry_utils` (basic), `enhanced_retry` (provider ladder with circuit breakers), Discord-specific retry in LLMBot |
| **Circuit Breaker** | OpenRouter client: per-model circuit breakers (CLOSED/OPEN/HALF_OPEN). HTTP client: circuit breaker with configurable thresholds. |
| **Request Coalescing** | `RequestCoalescer` — prevents duplicate concurrent work for identical inputs |
| **Rate Limiting** | Discord native rate limit handling. HTTP client per-host limits. Router decision budget enforcement. |
| **Streaming Status** | `STREAMING_ENABLE` → `start_streaming_status()`, background embed updater, plan inference based on content/attachments |
| **Hot-Reload** | File watcher on config files. Atomic config swap. Scoped component re-init (TTS, vision, HTTP client). |
| **Graceful Shutdown** | `GracefulShutdown` — signal handlers (SIGINT/SIGTERM), task cancellation, resource cleanup, session flushing. |
| **Phase Timing** | `PipelineTracker` tracks time spent in each phase (router, LLM call, search, vision, etc.) |
| **Session Cache** | `SessionCache` (15 methods) — per-user/server session state caching with UserProfile and ServerContext |

---

## 10. Type System & Contracts

### Core Data Models

**BotAction (bot/action.py)**: The unified output type for all processing
- `content`: str (text response)
- `embeds`: List[discord.Embed]
- `files`: List (file attachments)
- `audio_path`: Optional[str] (TTS output path)
- `meta`: Dict (metadata flags like `requires_tts`, `delegated_to_cog`)

**Router Response Flow**:
```
message → Router.dispatch_message() → Optional[BotAction]
  → If BotAction.meta["delegated_to_cog"]: fallthrough to process_commands()
  → If BotAction.has_payload: _execute_action()
```

**Vision Types (bot/vision/types.py)**:
- `VisionTask`, `VisionProvider`, `VisionJobState`, `VisionErrorType`, `VisionError`
- `VisionRequest`, `VisionResponse`, `VisionJob`, `ProviderRequest`, `ProviderResponse`

### Memory Types

**MemoryRecord (bot/memory/persistent_store.py)**:
- `id`: str (UUID)
- `user_id`: str
- `text`: str
- `category`: str
- `created_at`: float
- `source_message_id`: str
- `metadata`: dict
- `profile_id`: str

### RAG Types

**VectorDocument (bot/rag/vector_schema.py)**:
- `id`, `content`, `metadata`, `embedding`
- `chunk_index`, `source_path`, `doc_type`

**HybridSearchResult**:
- `content`, `score`, `metadata`, `source`

### Search Types

**SearchResult (bot/search/types.py)**:
- `title`, `url`, `snippet`, `score`
- `category`, `safe_search`

---

## 11. Configuration & Environment

### Required Environment Variables
- `DISCORD_TOKEN` (required)
- `BOT_PREFIX` (default: `!`, comma-separated for multiple)
- `OWNER_IDS` (list of user IDs, for admin commands)

### LLM Provider Configuration
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL`

### TTS Configuration
- `TTS_ENGINE` (kokoro, stub)
- `TTS_*` various engine-specific settings

### Vision Configuration
- `VISION_*` provider settings (Novita, Together API keys)
- `VL_*` vision-language settings

### Search/Web
- `DDG_API_ENDPOINT` (default: `https://html.duckduckgo.com/html/`)

### Infrastructure
- `PW_SERVER_URL` (Playwright remote server URL)
- `PROMETHEUS_ENABLED`, `PROMETHEUS_PORT`, `PROMETHEUS_HTTP_SERVER`
- `STREAMING_ENABLE`, `STREAMING_EMBED_STYLE`, `STREAMING_TICK_MS`
- `HTTP_*`, `PROXY_*`, `*_TIMEOUT`, `RETRY_*`
- Various context/token limits: `HISTORY_WINDOW`, `MAX_CONTEXT_TOKENS`

### Memory
- `CONTEXT_FILE_PATH`, `ENHANCED_CONTEXT_FILE_PATH`
- Memory profile configuration files

---

## 12. Security Posture

### Authentication
- Discord token in env, never logged (SHA-256 hash logged)
- Owner IDs checked for admin commands via `self.owner_ids` set

### Input Validation
- URL scheme restricted to http/https only
- Attachment file extension allowlist for temp file creation
- Content sanitization before Discord output (`sanitize_public_text`)

### File Safety
- Temp files use `tempfile.NamedTemporaryFile` (not attacker-controlled paths)
- Suffix validation for uploaded files
- Safe unlink on cleanup

### Data Protection
- Public output safety hooks monkey-patch ALL Discord send/edit boundaries
- Embeds sanitized before sending
- No secrets in logs (SensitiveDataFilter)

### Rate Limiting
- Discord native rate limit handling with Retry-After header respect
- HTTP client per-host rate limits
- Router decision budget enforcement

### Permission System
- Role-based access for operator commands
- Permission denied errors with clear messaging

---

## 13. Known Patterns & Conventions

### Coding Style
- **Async-first**: All I/O is async. No blocking calls in event loop.
- **Snake_case**: Functions, files, variables
- **Small functions**: Target <=30 lines, nesting <=3
- **Dual-sink logging**: `get_logger(__name__)` logs to both Rich console and JSONL file
- **Metric namespace**: Counters registered via `self.metrics.define_counter()`
- **Subsys labels**: JSONL logs use `extra={"subsys": "..."}` for categorization
- **Event labels**: JSONL logs use `extra={"event": "..."}` for structured events

### Error Handling
- Hierarchical exception types defined in `bot/exceptions.py`
- Never `except: pass` — catches specific exceptions (0 bare excepts found)
- Fallback patterns: graceful degradation with warnings logged
- No `eval()` or `exec()` usage (0 found by AST scan)
- No mutable default arguments (0 found by AST scan)
- No syntax errors (244/244 files parse clean)

### File Organization
- `bot/` root: core orchestration types (action, brain, see, speak, hear, router)
- `bot/core/`: bot class, startup, monitoring, infrastructure
- `bot/commands/`: Discord command cogs (one per domain)
- Feature subpackages: `memory/`, `rag/`, `tts/`, `vision/`, `stt_pipeline/`, `search/`, `server_archive/`, `syndication/`

### Testing
- pytest with `asyncio_mode = auto`
- ~1690 tests passing, 361 skipped, 0 failed
- Test files mirror `bot/` layout under `tests/`

---

## 14. Dependency Map

### Core Dependencies
| Package | Purpose |
|---|---|
| `discord.py` | Discord API client, voice support |
| `aiohttp` | HTTP client (connection pooling, circuit breaker) |
| `httpx` | HTTP client (web extraction service) |
| `rich` | Console formatting, progress indicators |
| `pydantic` | Data validation (likely via some imports) |

### AI/ML Dependencies
| Package | Purpose |
|---|---|
| `openai` | OpenAI API client |
| `sentence-transformers` | Local embeddings for RAG and memory |
| `chromadb` | Vector database for RAG |
| `torch` (optional) | PyTorch for SentenceTransformer |
| `kokoro` (local) | TTS engine via ONNX |
| `whisper` | Speech-to-text |

### Media Processing
| Package | Purpose |
|---|---|
| `playwright` | Remote browser (screenshots, web extraction) |
| `pillow` | Image processing |
| `pymupdf` / `PyPDF2` | PDF processing |
| `ffmpeg` (system) | Audio/video processing |

### Utilities
| Package | Purpose |
|---|---|
| `prometheus-client` | Metrics export |
| `beautifulsoup4` / `lxml` | HTML parsing |
| Various standard library modules | asyncio, pathlib, logging, sqlite3, etc. |

---

## 15. Edge Cases & Operational Notes

### NAS-Specific Workarounds
- **Playwright Chromium missing**: System `libatk-1.0.so.0` absent → Playwright containers. Use `sudo docker restart playwright` to fix.
- **Node fetch() blocked by Cloudflare**: System `curl` bypasses Cloudflare; Node scripts use `fetchHtmlWithCurlFallback()`.
- **entreee /opt in PATH**: Bad RPATH for native modules. Start gearo with clean PATH.
- **GLIBC 2.36**: nautilus_trader Rust extensions require GLIBC 2.39 — cannot run on this NAS.

### Known Stale/Duplicate Files
- `bot/core/bot-old.py`: Legacy LLMBot class (retained for reference)
- `bot/tts/manager.py` vs `bot/tts/manager_fixed.py`: Fixed version of TTS manager
- `bot/tts/kokoro_direct.py` vs `bot/tts/kokoro_direct_fixed.py`: Fixed version, `kokoro_direct_fixed` is the active one (31 methods vs 6)
- `bot/tts/kokoro_adapter.py` and `bot/tts/kokoro_bootstrap.py`: May be legacy/unused (read_file returned error)
- `bot/stt_pipeline/stitch.py` (23 lines), `bot/stt_pipeline/result_payload.py` (39 lines), `bot/stt_pipeline/logging.py` (32 lines): Very small — possibly thin compatibility shims

### Operational Quirks
- **Per-user queues with 300s timeout**: If a user goes silent for 5 minutes, their processor task is cleaned up
- **Typing suppression**: After a 429 rate limit on typing(), suppresses for 300s
- **Streaming status cards**: Background asyncio task updates embed every 750ms (configurable). Auto-stops after max steps or on cancellation
- **Content overflow handling**: Messages >2000 chars are split into chunks and overflow is attached as `full_response.txt`
- **Config hot-reload**: TTS manager is destroyed and recreated (thread executor shutdown). Vision orchestrator config is rebound in-place. HTTP client cleanup is fire-and-forget
- **Boot idempotency**: `_boot_completed` flag prevents duplicate setup_hook() execution
- **Session cache**: Uses UserProfile and ServerContext for caching user/guild-specific state

### Risk Areas
1. **Router class size (149 methods)**: The Router is the largest single class. Any routing changes require deep understanding of interdependent methods.
2. **Hot-reload thread safety**: `_on_config_reload` runs in a file watcher callback thread; all component re-init is wrapped in try/except but some dict mutations are not thread-safe
3. **Monkey-patching**: Public output safety hooks monkey-patch Discord classes' send/edit methods — this is fragile to discord.py upgrades
4. **Global state**: `_client_instance` in `bot/core/openrouter_client.py` is a global singleton. `enhanced_retry.get_retry_manager()` is also a global singleton.
5. **Fire-and-forget tasks**: `asyncio.create_task(cleanup_http_client())` during hot-reload with no reference retention — could be garbage collected
6. **`bot/stt_pipeline/`**: Very thin files (23-95 lines each) suggest a refactoring in progress or incomplete migration
7. **Duplicate TTS modules**: Three TTS manager implementations (`manager.py`, `manager_fixed.py`, `bot/tts/interface.py` TTSManager) — may cause confusion about which is active

---

## 16. Notable Code Quality Metrics

### AST Structural Scan Results
- **Files**: 244 Python files
- **Lines**: 101,642 total
- **Classes**: 383 defined
- **Bare `except:` clauses**: 0 (excellent)
- **Mutable default arguments**: 0 (excellent)
- **`eval()`/`exec()` usage**: 0 (excellent)
- **Syntax errors**: 0 (all files parse cleanly)

### Largest Classes
1. **Router** (bot/router.py): 149 methods — the central message routing engine
2. **LLMBot** (bot/core/bot.py): 38 methods — the main Discord bot class (3296 lines)
3. **PersistentMemoryStore** (bot/memory/persistent_store.py): 31 methods
4. **VideoIngestionManager** (bot/video_ingest.py): 30 methods
5. **CuratedMemoryCurator** (bot/memory/curator.py): 32 methods
6. **UnifiedVisionAdapter** (bot/vision/unified_adapter.py): 29 methods
7. **KokoroDirect** (bot/tts/kokoro_direct_fixed.py): 31 methods
8. **ServerArchiveStore** (bot/server_archive/store.py): 48 methods
9. **ObservabilityManager** (bot/core/observability_integration.py): 24 methods

### Smallest Modules (Potential Refactoring Candidates)
- `bot/stt_pipeline/stitch.py` (23 lines)
- `bot/stt_pipeline/logging.py` (32 lines)
- `bot/stt_pipeline/result_payload.py` (39 lines)
- `bot/stt_pipeline/spec_select.py` (27 lines)
