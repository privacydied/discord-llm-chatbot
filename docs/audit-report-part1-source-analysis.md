COMPREHENSIVE PRODUCT AUDIT REPORT - PART 1: SOURCE CODE ANALYSIS
================================================================================
Project: /mnt/nasirjones/py/discord-llm-chatbot
Date: 2026-05-13
Scope: bot/core/, bot/config/, bot/commands/, bot/routing/, bot/router_components/, bot/events/

1) LLMBot CLASS ARCHITECTURE (bot/core/bot.py - 3100 lines, 133KB)
================================================================================

INHERITANCE:
  - LLMBot extends discord.ext.commands.Bot
  - Provides command routing (via prefix system) and cog management through
    the discord.py framework

__init__(self, *args, config: dict | None = None, **kwargs):
  - Defaults command_prefix from COMMAND_PREFIX env or "!"
  - Defaults intents to discord.Intents.none()
  - Stores config dict (empty if None)
  - self.owner_ids = {int(x) for x in config.get("OWNER_IDS", [])}
  - self.metrics = NullMetrics() (replaced in setup_hook with PrometheusMetrics)
  - self.user_profiles, self.server_profiles = {}, {}
  - self.tts_manager: Optional[TTSManager] = None
  - self.archive_service = None
  - self.router: Optional[Router] = None
  - self._background_tasks: set[asyncio.Task]
  - self._boot_completed = False (idempotency guard)
  - ContextManager: filepath=CONTEXT_FILE_PATH (default "context.json"),
    max_messages=MAX_CONTEXT_MESSAGES (default 10)
  - EnhancedContextManager: filepath=ENHANCED_CONTEXT_FILE_PATH,
    history_window=HISTORY_WINDOW (default 10),
    max_token_limit=MAX_CONTEXT_TOKENS (default 4000)
  - _typing_suppressed_until: Dict[int, float] (rate-limit avoidance)
  - _active_long_running_tasks, _task_metadata, _task_lock
  - self.console = Console() (Rich output)

setup_hook(self) -> None (async, 462-845):
  Idempotent via _boot_completed flag. Sequence:
  1. Stores running event loop in self._event_loop
  2. Installs public output safety hooks (_install_public_output_safety_hooks)
     - Patches Messageable.send, Message.reply, Message.edit,
       InteractionResponse.send_message/edit_message, Interaction.edit_original_response,
       Webhook.send to auto-sanitize content via sanitize_public_message_payload
  3. Instantiates MessageProcessor(self)
  4. Initializes metrics:
     - Reads PROMETHEUS_ENABLED (default true), PROMETHEUS_PORT (8000),
       PROMETHEUS_HTTP_SERVER (true)
     - If enabled: PrometheusMetrics, otherwise NoopMetrics
     - Registers counters: gate.allowed, gate.blocked, x.photo_to_vl.*,
       x.syndication.*, vision.route.*
  5. Loads system prompts via load_system_prompts()
  6. Registers config hot-reload callback (_apply_config_reload):
     - Swaps live config, refreshes system prompts
     - Rebinds router.config, TTS manager, Vision orchestrator
     - Restarts HTTP client on HTTP_/PROXY_/TIMEOUT_/RETRY_ changes
  7. await self.load_profiles()
  8. self.setup_background_tasks()
  9. await self.setup_tts()
  10. await self.setup_router()
  11. await self.setup_rag()
  12. await self.load_extensions() - loads all command cogs via commands/setup_commands()
  13. setup_command_error_handler(self) - installs global command error handler
  On failure: resets _boot_completed = False and re-raises

load_extensions / COG LOADING:
  Delegated to bot/commands/__init__.py:setup_commands(bot). This function:
  1. Dynamically imports 10 command modules via __import__:
     test_cmds, memory_cmds, memory_extended_cmds, tts_cmds, config_commands,
     janitor_commands, video_commands, rag_commands, img_commands, archive_commands
  2. For each imported module, checks for setup() function and calls await module.setup(bot)
  3. Each module.setup() calls bot.add_cog(CogClass(bot))
  Additional cogs loaded directly (not via setup_commands):
     - ScreenshotCommands: loaded via bot/commands/screenshot_commands.py (not in module_imports!)
     - ImageUpgradeCommands: loaded via bot/commands/image_upgrade_commands.py (not in module_imports!)
     - VisionCommands: loaded via bot/commands/vision_commands.py (not in module_imports!)
     - SearchCommands: loaded via bot/commands/search_commands.py (not in module_imports!)
     - OperatorCommands: loaded via bot/commands/operator_commands.py (not in module_imports!)
     - ContextCommands: loaded via bot/commands/context_commands.py (not in module_imports!)
     - AdminAlertCommands: loaded via bot/commands/admin_alert_commands.py (not in module_imports!)
  NOTE: Only the 10 cogs in setup_commands module_imports dict are auto-loaded via
  load_extensions. The other 7 cogs must be loaded elsewhere (likely in startup.py or main.py).

process_commands(self, message) -> Optional[Any]:
  Overridden from commands.Bot. Short-circuits non-command messages:
  - Only processes messages starting with a registered prefix
  - Returns None for non-commands (avoids CommandNotFound errors)
  - Falls through to super().process_commands() for valid commands
  - Catches and suppresses commands.errors.CommandNotFound

on_ready(self):
  Logs bot identity, sets _is_ready event. All setup is done in setup_hook.

_process_single_message(self, message):
  - Appends to both context managers
  - Best-effort enqueue_inferred_memory for long-term memory
  - If router exists, calls await self.router.dispatch_message(message)
  - Handles BotAction:
    - action.meta.get("delegated_to_cog"): calls await self.process_commands(message)
    - action.has_payload: calls await self._execute_action(message, action, ...)
  - If no action and message looks like a command: falls back to process_commands
  - Supports streaming status cards (STREAMING_ENABLE config)

_execute_action(self, message, action, target_message, dispatch_meta):
  - Handles TTS processing (requires_tts meta flag)
  - Native Discord voice message publishing (VOICE_ENABLE_NATIVE)
  - Sends/edits messages to target channel/thread with embeds, content, files
  - Tracks in enhanced context manager
  - Comprehensive logging with dispatch metadata

KEY HELPER METHODS:
  _track_background_task(task): Registers and logs exception for background tasks
  _optional_typing(channel): Enter typing() with rate-limit suppression (5min on 429, 1min otherwise)
  _call_with_discord_retry(operation, func, attempts=3): Retries transient Discord HTTP errors
  _discord_retry_delay(error, attempt): Computes backoff using Retry-After header
  _is_retryable_discord_http_error(error): 429 or 5xx or specific error strings
  _infer_streaming_plan(message): Returns step labels based on content/attachments
  _start_streaming_status(message): Creates embed + background updater task
  _stop_streaming_status(stream_ctx): Cancels task, finalizes embed
  _streaming_updater(msg, style, tick_ms, max_steps, plan): Updates embed in loop
  _build_stream_embed(label, style, step, max_steps, done): Creates status embed

CROSS-REFERENCES:
  - Messages routed via Router.dispatch_message() (bot/core/fast_path_router.py)
  - TTS via self.tts_manager (bot/tts/)
  - Memory via bot.memory.* functions
  - Archives via bot.server_archive.*

2) CONFIGURATION KEYS AND DEFAULTS
================================================================================

A) bot/config/media_config.py - MediaIngestionConfig dataclass:
  probe_cache_ttl_seconds: 300 (MEDIA_PROBE_CACHE_TTL)
  probe_timeout_seconds: 10 (MEDIA_PROBE_TIMEOUT)
  probe_cache_dir: "cache/media_probes" (MEDIA_PROBE_CACHE_DIR)
  max_concurrent_downloads: 2 (MEDIA_MAX_CONCURRENT)
  download_timeout_seconds: 60 (MEDIA_DOWNLOAD_TIMEOUT)
  speedup_factor: 1.5 (MEDIA_SPEEDUP_FACTOR)
  retry_max_attempts: 3 (MEDIA_RETRY_MAX_ATTEMPTS)
  retry_base_delay_seconds: 2.0 (MEDIA_RETRY_BASE_DELAY)
  audio_cache_dir: "cache/video_audio" (VIDEO_CACHE_DIR)
  cache_expiry_days: 7 (VIDEO_CACHE_EXPIRY_DAYS)
  whitelisted_domains: youtube.com, youtu.be, tiktok.com, m.tiktok.com, vm.tiktok.com, twitter.com, x.com
  max_title_length: 200 (MEDIA_MAX_TITLE_LENGTH)
  max_uploader_length: 100 (MEDIA_MAX_UPLOADER_LENGTH)
  max_url_length: 500 (MEDIA_MAX_URL_LENGTH)
  enable_media_ingestion: true (ENABLE_MEDIA_INGESTION)
  enable_twitter_video_detection: true (ENABLE_TWITTER_VIDEO_DETECTION)
  enable_contextual_brain: true (USE_ENHANCED_CONTEXT)

B) bot/core/config_validation.py - ConfigValidator constraints:
  Required:
    DISCORD_TOKEN: Discord bot token (REQUIRED)
  Optional:
    DISCORD_INTENTS: default|all|messages|guild_messages
    OPENAI_API_KEY: for GPT models
    ANTHROPIC_API_KEY: for Claude models
    OBS_ENABLE_PROMETHEUS: true|false
    OBS_PARALLEL_STARTUP: true|false
    OBS_ENABLE_HEALTHCHECKS: true|false
    OBS_ENABLE_RESOURCE_METRICS: true|false
    PROMETHEUS_PORT: 1024-65535 (int)
    LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL
    LOG_JSONL_PATH: file path
    RAG_EAGER_VECTOR_LOAD: true|false
  Cross-field constraints:
    - At least one AI provider: OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_HOST
    - If PROMETHEUS enabled + HTTP server enabled, PROMETHEUS_PORT must be set

C) Keys referenced from bot/core/bot.py __init__:
  COMMAND_PREFIX: "!"
  OWNER_IDS: []
  CONTEXT_FILE_PATH: "context.json"
  MAX_CONTEXT_MESSAGES: 10
  ENHANCED_CONTEXT_FILE_PATH: "enhanced_context.json"
  MAX_CONTEXT_TOKENS: 4000
  HISTORY_WINDOW: 10

D) Keys observed throughout the codebase (from commands, routing, etc.):
  PROMETHEUS_ENABLED: "true"
  PROMETHEUS_PORT: 8000
  PROMETHEUS_HTTP_SERVER: "true"
  STREAMING_ENABLE: False (bot.py)
  STREAMING_EMBED_STYLE: "compact"
  STREAMING_TICK_MS: 750
  STREAMING_MAX_STEPS: 8
  VISION_ENABLED: False
  VISION_EPHEMERAL_RESPONSES: True
  VOICE_ENABLE_NATIVE: False
  ALERT_ENABLE: "false"
  ALERT_SESSION_TIMEOUT_S: 1800
  ALERT_ADMIN_USER_IDS: ""
  SEARCH_PROVIDER: "ddg"
  SEARCH_MAX_RESULTS: 5
  SEARCH_SAFE: "moderate"
  DDG_TIMEOUT_MS: 5000
  CUSTOM_SEARCH_TIMEOUT_MS: 8000
  IMG_ATTACHMENT_MAX_BYTES: 262144
  IMG_ATTACHMENT_ENABLE: "true"
  TEXT_BACKEND: (varies, shown in config-status)
  OPENAI_TEXT_MODEL: (varies)
  OLLAMA_MODEL: (varies)
  TTS_BACKEND: (varies)
  STT_ENGINE: (varies)
  MAX_USER_MEMORY: (varies)
  MAX_SERVER_MEMORY: (varies)
  MAX_MEMORIES: (from config)
  MAX_SERVER_MEMORIES: (from config)
  MEDIA_PROBE_CACHE_TTL: 300
  VISION_TAGS_ENABLE: True

E) HealthMonitor config_validation.py:
  - Tracks ComponentHealth: name, status, last_init_timestamp, last_error,
    check_count, consecutive_failures
  - SystemHealth: status, components, prometheus_enabled, degraded_mode,
    degraded_reasons, uptime_seconds, rss_mb, event_loop_lag_ms
  - Liveness: psutil process check + event loop responsiveness (<100ms)
  - Readiness: all components READY, memory usage <90%
  - Health statuses: READY, DEGRADED, NOT_READY

3) EVERY DISCORD COMMAND AND BEHAVIOR
================================================================================

A) test_cmds.py - TestCommands cog:
  !ping - Sends "Pong! Pong!" - simple connectivity test

B) memory_cmds.py - MemoryCommands cog:
  !memory-add <content> - Add curated memory (user_preference type)
  !memory-show [limit] - Show recent durable memories (default 5, via embed)
  !memory-del <id|search> - Delete memory by ID or fuzzy search
  !memory-wipe - Wipe all durable memories (no confirmation!)
  !memory-search <query> - Search durable memories
  !memory-distill-once [guild-only, admin] - Run distillation in background
  !memory-distill-status [guild-only, admin] - Show distiller metrics
  !memory-distill-dryrun <on|off> [guild-only, admin] - Toggle dry-run mode
  !memory group (invoke_without_command):
    !memory add <content> - Add memory (10s cooldown per user, 2000 char limit,
      sanitizes <script>, javascript:, data:, vbscript:)
    !memory list [limit] - List recent memories (max 20)
    !memory clear - Clear all memories (30s guild cooldown, requires "yes" confirmation)
  !server-memory group [guild-only, admin]:
    !server-memory add <content> - Add server memory (2000 char limit, sanitization)
    !server-memory list - List server memories (max 25 fields)
    !server-memory clear - Clear all server memories (60s cooldown, confirmation)

C) memory_extended_cmds.py - ExtendedMemoryCommands cog:
  !memory-status [alias: mem-status] - Owner/admin only: show service status,
    queue depth, vector store, SQLite store
  !memory-review [limit] [alias: mem-review] - Review curated memories with
    confidence scores, summary, created date
  !memory-forget <id> [alias: mem-forget] - Forget specific memory by exact/prefix match
  !memory-enable - Enable memory service
  !memory-disable - Disable memory service
  !memory-export - Export memories as JSON file (CSV-like, attached)

D) tts_cmds.py - TTSCommands cog:
  !tts [text] - Base group. If text provided: run speak(ctx, text). Else: show help
  !tts on - Enable TTS for user
  !tts off - Disable TTS for user
  !tts all <on|off> [guild-only, admin] - Global TTS toggle
  !speak <text> [, pcm16=bool] - Speak text directly OR set one-time TTS flag
  !say <text> [, timeout_s, cold, timeout_cold_s, timeout_warm_s] - Direct TTS
    synthesis without AI response. Falls back to user's previous message if text empty.
    Tries native Discord voice message first, then falls back to file attach.

E) config_commands.py - ConfigCommands cog:
  !reload-config [aliases: reload_config, config_reload] [admin] - Reload .env config
  !config-status [aliases: config_status, config_info] [admin] - Show config version
    and key settings embed
  !config-help [alias: config_help] - Show config commands help

F) janitor_commands.py - JanitorCommands cog:
  !clean [aliases: cleanup, janitor] [admin] - Manual cache/log cleanup
  !clean-status [aliases: cleanup-status, janitor-status] [admin] - Show janitor
    configuration: schedule (60min), hold-off (30min), retention, directory policies
  !clean-help [aliases: cleanup-help, janitor-help] - Help for janitor commands

G) video_commands.py - VideoCommands cog:
  !watch <url> [--speed N] [--force] [aliases: transcribe, listen] [60s/user cooldown]
    - Transcribe YouTube/TikTok video audio via hear_infer_from_url
    - Returns title, duration, speedup, cache status, transcription preview
    - Speed range: 0.5x-3.0x
  !video-help [alias: watch-help] - Help for video commands
  !video-cache [alias: watch-cache] [admin] - Show video cache statistics

H) rag_commands.py - RAGCommands cog:
  !rag group [admin check via bot.core.permissions.is_admin_user]:
    !rag status - Show RAG environment, collection stats, health check
    !rag test - Run RAG system tests (env, hybrid search, search fn, collection)
    !rag reload - Reload text index
    !rag clear - Remove documents
    !rag wipe - WIPE entire database
    !rag invalidate - Invalidate collection
    !rag search <query> - Search for documents
    !rag index <path> - Index documents from directory

I) img_commands.py - ImgCommands cog:
  !img <prompt> [alias: image] - Image generation via vision system
    - Accepts inline prompt OR text attachment (.txt/.json/.yaml/.yml, <=256KB)
    - JSON prompts can include: prompt, negative_prompt, width, height, steps,
      guidance_scale, seed, model
    - Delegates to router._handle_vision_generation()
    - Shows help embed if prompt empty/attachment disabled

J) archive_commands.py - ArchiveCommands cog [guild-only]:
  !archive-status [admin] - Show server archive status (OFF/PAUSED/ON, sync state,
    message counts, queue depth, dropped count)
  !archive-search <query> [admin] - Search archived messages (limit from service config)
  !archive-sync [admin] - Start guild-level archive sync
  !archive-sync-channel [channel] [admin] - Sync specific channel/thread
  !archive-pause [admin] - Pause archive background sync
  !archive-resume [admin] - Resume archive background sync

K) screenshot_commands.py - ScreenshotCommands cog:
  !ss <url> [alias: screenshot] - Take screenshot and analyze via VL
    - Streaming progress updates (6 stages: validate, prepare, capture, save, analyze, done)
    - Respects STREAMING_ENABLE config
    - Privacy note: explicitly command-gated, no background screenshots

L) search_commands.py - SearchCommands cog:
  !search <query> - Web search via pluggable provider (SEARCH_PROVIDER config, default "ddg")
    - Configurable: max_results, safesearch (moderate/strict/off), locale, timeout

M) vision_commands.py - VisionCommands cog (SLASH COMMANDS):
  /image <prompt> [size, steps, guidance, negative, seed, count, provider, model]
    - Text-to-image generation via VisionOrchestrator
    - Sizes: square(1024x1024), portrait(768x1024), landscape(1024x768), 4k(2048x2048)
    - Providers: together, novita, openrouter, auto
    - Monitors job progress in background
  /imgedit <image attachment> <prompt> [strength, steps, guidance, negative, seed, provider, model]
    - Image editing/variations
    - Image-to-image, strength 0.1-1.0
    - Valid attachments: jpg/jpeg/png/webp, max 25MB
  /video <prompt> [duration, fps, resolution, style, seed, provider, model]
    - Text-to-video generation
    - Resolutions: 720p, 1080p
  /vidref <image attachment> <prompt> [duration, fps, mode, seed, provider, model]
    - Image-to-video animation
    - Modes: image2video, start_end

N) screenshot_commands.py - ScreenshotCommands (continued, 225 lines total)

O) context_commands.py - ContextCommands cog:
  !context_reset [aliases: reset_context, clear_context] - Reset conversation context
  !context_stats [alias: ctx_stats] - Show context manager statistics
  !privacy_optout [aliases: opt_out, no_context] - Opt out of context tracking
  !privacy_optin [aliases: opt_in, enable_context] - Opt into context tracking
  !context_help [alias: ctx_help] - Show context management help

P) operator_commands.py - OperatorCommands cog:
  !help [aliases: capabilities, capability] - Show capability card embed
  !status [admin] - Lightweight operator health summary: uptime, RSS, backend,
    vision status, degraded mode, memory service, RAG, STT, TTS, Ollama,
    Playwright, queue/backpressure, feature toggles
  !feature <name> <on|off> [aliases: toggle-feature, toggle_feature] [admin] - Toggle
    per-server feature flags: stt, tts, vision, image, web, x, rag

Q) admin_alert_commands.py - AdminAlertCommands cog (1408 lines):
  Extensive DM-only alert broadcasting system with emoji-driven composer:
  - Session-based composer (COMPOSING, READY, POSTING, COMPLETED, CANCELLED, EXPIRED)
  - Steps: select_channels, compose_content, preview_alert, confirm_send
  - Destination discovery (max 10 guilds, 3 channels per guild, permission-checked)
  - Reaction-queue processing with 250ms spacing
  - Session timeout: configured ALERT_SESSION_TIMEOUT_S (default 1800s)
  - Authorization: ALERT_ADMIN_USER_IDS or OWNER_IDS fallback
  - Command definitions visible in cog (line 500+): creates sessions, handles reactions

4) MESSAGE ROUTING SYSTEM
================================================================================

A) bot/routing/ Directory:
  Base types (bot/routing/base.py):
    - RouteContext: dataclass carrying message, author_id, source_type, payload,
      model_override, progress_cb, item
    - RouteResult: dataclass with text field, text_only() factory
    - RouteHandler (Protocol): can_handle(ctx) -> bool, async handle(ctx) -> RouteResult

  ScreenshotHandler (bot/routing/screenshot_handler.py):
    - can_handle: True when ctx.source_type == "url"
    - handle: Captures screenshot via external_screenshot(url), analyzes via see_infer,
      supports progress_cb with stages: validate(1), prepare(2), capture(3),
      saved(4), analyze(5), done(6)
    - Module singleton: screenshot_handler
    - Compatibility function: handle_screenshot_url(item, progress_cb)

  UnknownHandler (bot/routing/unknown_handler.py):
    - can_handle: Always True (final fallback)
    - handle: Logs warning, returns RouteResult with "Unsupported input type" message
    - Constructor takes optional logger

B) bot/router_components/ Directory:
  This is a massive extraction layer (1117 lines in __init__.py exports alone) that
  decomposes the Router into independently-testable components:

  compose.py (288 lines):
    - format_x_tweet_with_transcription(): Assembles EvidenceBundle with caption + STT
      transcript, adds STT grounding instructions
    - format_x_tweet_result(): Formats X API tweet response into text
    - has_visual_facts_section(): Detects visual-facts evidence blocks in text
    - build_visual_analysis_anchor_prompt(): Builds visual-analysis anchoring instruction
    - compose_x_tweet_with_visual_facts(): Composes text-flow with caption + VL facts

  gating.py (51 lines):
    - mentions_bot(message, bot_user_id): Checks if message mentions bot
    - is_reply_to_bot(message, bot_user_id): Checks if reply is to bot's message
    - strip_leading_bot_mention(content, bot_user_id): Strips <@id> prefix

  input_harvest.py (200 lines):
    - is_text_attachment(attachment): Checks for .txt or text/* mime type
    - all_attachments_are_text(attachments): All attachments are text files
    - has_meaningful_text(text): Relaxed chat signal check (alphanumeric, emoji, short tokens)
    - has_explicit_media_intent(text): Detects media intent keywords
    - is_direct_image_url(url): Checks for .jpg/.png/.webp extensions
    - extract_urls_loose(text): Permissive URL extraction
    - extract_urls_strict(text): Stricter URL extraction
    - strip_urls(text): Remove URLs from text
    - strip_discord_mentions_and_urls(text): Remove mentions and URLs
    - existing_url_payloads(items): Collect URL payloads for deduplication
    - append_unique_url_items(items, urls, item_ctor, ...): Append URL items with dedup
    - append_embed_related_urls(found_urls, embeds): Extract URLs from embed objects

  prompt_access.py:
    - get_system_prompt() - retrieves configured system prompt

  runtime.py:
    - RouterRuntimeCompat, load_router_runtime_compat - compatibility shim for
      runtime configuration access

  x_routing.py (massive, 1100+ export names):
    Comprehensive Twitter/X syndication and URL extraction:
    - URL extraction: compile_url_extract_regex, status_url_extract_regex,
      x_url_extract_regex, iter_url_matches, extract_x_status_urls_from_text
    - Syndication: build_syndication_fetch_headers, build_syndication_fetch_params,
      build_syndication_accept_language*, build_syndication_oembed*
    - Content extraction: extract_x_article_text, extract_syndication_text,
      extract_syndication_article_text, extract_note_tweet_text
    - Cache: syndication_cache_ttl_s, build_syndication_cache_entry,
      classify_syndication_cache_hit
    - STT error handling: classify_stt_error_reason, is_stt_hard_error,
      build_stt_fail_log_payload
    - URL collection: collect_x_candidate_urls, collect_embed_candidate_urls,
      collect_attachment_candidate_urls, filter_non_empty_urls
    - Tweet extraction: extract_x_api_primary_tweet, extract_fxtwitter_tweet_node,
      extract_sparse_media_resolution
    - Article processing: iter_article_blocks, extract_article_content,
      normalize_article_block_text, build_article_text_parts
    - Media probing: resolve_and_probe_twitter_images, probed_image_urls_or_empty

C) Fast Path Router (bot/core/fast_path_router.py):
  The main Router class referenced throughout. It:
  - Receives messages via dispatch_message(message) -> BotAction
  - Uses router_components for input harvesting, gating, composition
  - Routes to handlers based on InputItem source_type
  - Integrates with modality system (InputItem with source_type, payload, order_index)

5) EVENT HANDLERS (bot/events/)
================================================================================

A) bot/events/command_error_handler.py:
  setup_command_error_handler(bot) -> CommandErrorHandler:
    - Installs global error handler via bot.tree.on_error (for slash commands)
      and bot.event handlers
    - Handles commands.CommandNotFound, commands.MissingPermissions,
      commands.CommandOnCooldown, commands.BadArgument, commands.CheckFailure
    - Provides user-friendly error messages for each error type
    - Logs detailed error info with subsys="command_error" event classification
    - Rate-limits error messages to prevent spam
    - Handles interaction errors for slash commands
    - Provides breadcrumb logging with event labels

  Key error type handling:
    - CommandNotFound: silently ignored (handled by process_commands override)
    - MissingPermissions: "You don't have permission to use this command."
    - CommandOnCooldown: "Please wait X seconds before using this command again."
    - BadArgument: "Invalid argument provided. Use !help for usage."
    - CheckFailure: generic check failure message

B) bot/events/__init__.py:
  Empty export file. Imports command_error_handler.

================================================================================
CROSS-REFERENCE MAP
================================================================================

bot/core/bot.py (LLMBot)
  |--> Commands: loaded via setup_commands() + direct add_cog() calls
  |--> Router: setup_router() instantiates Router, set as self.router
  |--> Message pipeline: _process_single_message -> router.dispatch_message -> BotAction
  |--> Config: self.config dict, hot-reload via add_reload_callback
  |--> TTS: self.tts_manager (bot/tts/)
  |--> Memory: bot.memory.* functions for enqueue/save/load
  |--> Archives: bot.server_archive.* 
  |--> Metrics: NullMetrics -> PrometheusMetrics
  |--> Events: setup_command_error_handler() -> global error handling

bot/commands/ (17 files, 15+ cogs)
  |--> Each cog is a discord.ext.commands.Cog subclass
  |--> Module-level async setup(bot) -> await bot.add_cog(Cog(bot))
  |--> Loaded during setup_hook() phase

bot/routing/ (3 handler files + base types)
  |--> RouteHandler protocol: can_handle() + handle()
  |--> ScreenshotHandler: URL -> external_screenshot -> see_infer VL analysis
  |--> UnknownHandler: catch-all fallback

bot/router_components/ (6 extraction modules)
  |--> compose.py: EvidenceBundle composition for X tweets
  |--> gating.py: Bot mention/reply detection
  |--> input_harvest.py: URL extraction, text analysis, attachment handling
  |--> prompt_access.py: System prompt retrieval
  |--> runtime.py: Runtime config compatibility
  |--> x_routing.py: Massive X/Twitter syndication + URL extraction library

bot/events/ (1 file)
  |--> command_error_handler.py: global error handler for prefix + slash commands

================================================================================
END OF REPORT
================================================================================
