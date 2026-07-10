# Architecture Decision Record (ADR)

## Overview

This document describes the architecture of the Discord bot, focusing on
the core components, data flows, and important design decisions.

## High-Level Flow

Discord events (messages, interactions) flow through the following layers:

1. **Gate** (`bot/core/startup_orchestrator.py`): Filters and validates incoming events (feature flags, degraded mode).
2. **Queue** (`bot/router.py`): Manages backpressure and task prioritization via `ConcurrencyManager`.
3. **Router** (`bot/router.py`): Routes events to appropriate handlers (commands, vision, text, media, URLs).
4. **Backend** (`bot/openai_backend.py`, `bot/vision/`, `bot/hear.py`, `bot/tts/`): Executes logic (LLM, TTS, STT, vision).
5. **Response**: Sends response back to Discord with output sanitization (`bot/public_output.py`).

## Component Ownership

| Component | Owner | Responsibilities |
|-----------|-------|------------------|
| `bot/core/bot.py` | Core bot | Discord event loop, command registration, bot lifecycle |
| `bot/core/startup_orchestrator.py` | Startup | Feature flags, degraded mode tracking, component health |
| `bot/router.py` | Router | Event routing to handlers (URL, media, vision, text) |
| `bot/commands/` | Command cogs | Admin, archive, config, memory, TTS, RAG, context operators |
| `bot/openai_backend.py` | Text backend | OpenRouter/OpenAI-compatible text generation, fallback ladder |
| `bot/hear.py` | STT | Speech-to-text with Whisper, chunk ordering, RAM guard |
| `bot/tts/` | TTS | Text-to-speech with KokoroDirect, phoneme/grapheme pipeline |
| `bot/vision/` | Vision | Image/video analysis via Together.ai, Novita.ai, OpenRouter |
| `bot/memory/` | Memory | User/server memory, context management, persistence |
| `bot/rag/` | RAG | ChromaDB hybrid search, indexing, bootstrap |
| `bot/syndication/` | Syndication | X/Twitter extraction, captioning, media selection |
| `bot/atomic_json.py` | Persistence | Atomic JSON writes for memory/context/profile files |
| `bot/url_safety.py` | Security | SSRF validation, prompt injection wrapping |
| `bot/core/permissions.py` | Security | Centralized admin permission checks |

## Explicit "Must Not Own" Boundaries

- **`bot/router.py`** must not depend on specific command implementations. Use command routing only.
- **`bot/commands/`** must not directly instantiate backend services; use dependency injection via bot attributes.
- **`bot/openai_backend.py`** must not depend on Discord library; use abstractions.
- **`bot/memory/`** must not depend on specific backend implementations.
- **`bot/vision/`** must not depend on Discord library; use `VisionGateway`/`UnifiedVisionAdapter`.
- **`bot/url_safety.py`** must not block on DNS resolution; use `run_in_executor`.

## Data Flow

### Message Processing
1. Discord event → `on_message`/`on_command` in `bot/core/bot.py`
2. Router applies gate (feature flags) → queue (backpressure) → handler dispatch
3. Handler may use backend services (LLM, TTS, STT, vision, web extraction)
4. Response sent back to Discord with output sanitization

### Memory/Context/RAG Flow
1. User message → context extraction → memory lookup (`CuratedMemoryService`)
2. RAG query → `hybrid_search` (ChromaDB) → vector embeddings → LLM prompt
3. Context stored in memory → used for future responses
4. Atomic writes via `bot/atomic_json.py` for profile/context persistence

### Provider Fallback Ladders
- **Text (via `bot/openai_backend.py`)**: OpenRouter model ladder (configurable via `VISION_FALLBACK_MODELS`/`TEXT_FALLBACK_` env vars) → retry with backoff
- **Vision (via `bot/vision/unified_adapter.py`)**: Together.ai → Novita.ai → OpenRouter → fallback providers
- **STT (via `bot/hear.py`)**: Whisper models (configurable by duration) → RAM guard protection
- **TTS (via `bot/tts/`)**: KokoroDirect with phoneme/grapheme routing via `TokeniserRegistry`

## Degraded Mode Behavior

When components fail, the bot should:
- Continue operating with available components (graceful degradation)
- Disable unavailable features (vision, TTS, STT, RAG) per-server via feature flags
- Report degraded status via `/status` command (includes `is_degraded_mode()` + `get_degraded_reasons()`)
- Retry with fallback providers when possible
- Track degraded reasons via `bot/core/startup_orchestrator.py` component state

## Admin Permission Model

Centralized permission checks in `bot/core/permissions.py`:
- `is_admin_user(user, bot)` — single async check for bot owners + configured admin IDs + guild admin
- `admin_only_prefix()` — decorator for prefix commands (normal reply on denial, NEVER ephemeral)
- `admin_only_slash()` — decorator for slash commands (ephemeral denial on interaction)
- Owner resolution: `bot.owner_ids`, config `OWNER_IDS`, `ALERT_ADMIN_USER_IDS`
- DM context: only allows configured owners/admins
- Guild context: `guild_permissions.administrator`

## URL/Prompt Safety Model

### SSRF Protection (`bot/url_safety.py`)
- **Pre-fetch**: `validate_url()` checks scheme (http/https), hostname, raw IP against forbidden ranges
- **DNS resolution**: `validate_url_with_dns()` resolves hostname off event loop, checks all IPs
- **Post-redirect**: `validate_redirect_response()` validates final URL after httpx redirects
- **Blocked ranges**: RFC1918 private, loopback (127.0.0.0/8, ::1), link-local (169.254.0.0/16), reserved, cloud metadata (169.254.169.254, 169.254.170.2), IPv6 loopback/link-local

### Prompt Injection Wrapping
- `wrap_untrusted_content()` wraps all externally fetched content as untrusted
- Header instructs model not to follow instructions inside fetched content
- Applied to URL summaries, PDF ingestion, RAG content, web extraction results

### Output Sanitization
- Bot responses go through `bot/public_output.py` (`extract_public_reply_text`, `has_reasoning_leakage`, `sanitize_public_text`)
- Streaming chunks sanitized before sending to Discord

## How to Add a Backend

1. Create new module in appropriate location (`bot/` for backends)
2. Implement required interface (e.g., text backend follows OpenAI-compatible format)
3. Register in `bot/config.py` configuration
4. Add fallback entry to provider ladder configuration
5. Use environment variables for credentials and settings

## How to Add a Command Cog

1. Create new file in `bot/commands/`
2. Define cog class inheriting from `commands.Cog`
3. Implement command methods with `@commands.command()` (prefix) or `@app_commands.command()` (slash)
4. Use `@admin_only_prefix()`/`@admin_only_slash()` from `bot/core/permissions.py` for admin commands
5. Register in `bot/commands/__init__.py` or via bot.add_cog()

## How to Add a Media Route

1. Create handler in `bot/routing/`
2. Implement `can_handle(ctx)` → `bool` and `async handle(ctx)` → `RouteResult`
3. Register with router dispatch logic
4. Add feature flag if needed
5. Add tests in `tests/test_router_*`

## Important Env Vars/Config Controls

| Variable | Purpose |
|----------|---------|
| `DISCORD_BOT_TOKEN` | Discord bot authentication token |
| `OWNER_IDS` | Comma-separated Discord user IDs of bot owners |
| `ALERT_ADMIN_USER_IDS` | Admin user IDs for alert commands |
| `TEXT_BACKEND` | Text backend selection (`openrouter`, `openai`) |
| `OPENAI_TEXT_MODEL` | Default text model (e.g., `anthropic/claude-sonnet-4-20250514`) |
| `VISION_API_KEY` | Vision provider API key (Together/Novita) |
| `VISION_ALLOWED_PROVIDERS` | Comma-separated: `together,novita,openrouter` |
| `VISION_FALLBACK_MODELS` | Fallback model ladder for vision |
| `STT_ENGINE` | STT engine selection (`whisper`, `faster_whisper`) |
| `TTS_BACKEND` | TTS backend selection (`kokoro`, `edge`, `elevenlabs`) |
| `RAG_ENABLED` | Enable/disable RAG system |
| `CHROMA_URL` | ChromaDB connection URL |
| `PW_SERVER_URL` | Playwright remote server URL (`ws://host:3006`) |
| `SCREENSHOT_API_KEY` | Screenshot service API key |
| `WEBEX_ENABLE_TIER_B` | Enable web extraction tier B (Playwright) |
| `KOKORO_FORCE_IPA` | Force IPA phonemes for TTS (`0` or `1`) |
| `KOKORO_GRAPHEME_FALLBACK` | Allow grapheme fallback for TTS (`0` or `1`) |
| `MAX_USER_MEMORY` | Max user memory entries |
| `MAX_SERVER_MEMORY` | Max server memory entries |
| `TIMEOUT` | Default request timeout |
| `VL_REQUEST_TIMEOUT` | Vision request timeout |
| `TEXT_FALLBACK_TIMEOUTS` | Text fallback timeout ladder |
| `TEXT_FALLBACK_MAX_ATTEMPTS` | Max fallback attempts |

## Router Component Contract

### Message Entry Path

Messages flow through a strict pipeline with defined boundaries at each layer:

```
discord.on_message (bot/core/bot.py)
  └─> MessageProcessor.on_message()          # dedup, archive, bot-self filter
        └─> MessageProcessor.enqueue()        # per-user asyncio.Queue
              └─> MessageProcessor._process_user_messages()  # drain queue serially
                    └─> LLMBot._process_single_message()     # context, memory injection
                          └─> Router.dispatch_message()      # gate → route → respond
                                └─> BotAction or ResponseMessage returned
```

Key invariants:
- **MessageProcessor** owns per-user queues and deduplication. Do not bypass it.
- **LLMBot._process_single_message** is the single entry point to the router. It injects conversation context and triggers memory ingestion before calling `Router.dispatch_message()`.
- **Router.dispatch_message** enforces the *1 IN > 1 OUT* rule: exactly zero or one response is generated per message.

### Route Handler Protocol

Route handlers in `bot/routing/` implement the `RouteHandler` protocol (`bot/routing/base.py`):

```python
class RouteHandler(Protocol):
    async def can_handle(self, ctx: RouteContext) -> bool: ...
    async def handle(self, ctx: RouteContext) -> RouteResult: ...
```

Rules:

- **`can_handle(ctx)`** — synchronous in behavior, **cheap, no network, no blocking I/O**. Called for *every* message. Check source types, URL patterns, or attachment types here. Return `True` only when this handler is the correct match.
- **`handle(ctx)`** — async, **may perform bounded I/O** (HTTP calls, file downloads, vision inference). Must respect configured timeouts (see `TIMEOUT`, `VL_REQUEST_TIMEOUT`, and per-provider ladders). Always returns a `RouteResult` (or `ResponseMessage` in the legacy flow); never returns `None` or raises uncaught exceptions.

### First-Match-Wins Routing

The router iterates handlers in **priority order** and stops at the first match where `can_handle` returns `True`. There is no fallback chaining or multi-handler dispatch for a single modality. Ordering matters — more specific handlers must register before broader ones.

### Feature Gates

Handlers must check feature toggles before doing work. The router provides `_feature_gate_response()` which inspects server-level flags for STT, TTS, vision, RAG, and other optional subsystems. When a feature is disabled for the current guild, handlers should return an informative error response rather than silently dropping or attempting the operation.

Use `bot/server_features.is_server_feature_enabled()` or the router's gate helpers — do not read config flags directly in handler code.

### Mention-Gating Invariant (CRITICAL)

This is a hard behavioral contract:

| Context | Behavior |
|---------|----------|
| **Guild channels** (text, threads, voice) | Reply **ONLY** when the bot is mentioned (`<@bot_id>` / `<@!bot_id>`) OR explicitly invoked via the configured command prefix (`!`). Unmentioned text in guild channels is ignored by the router. |
| **DM channels** | No mention or prefix required. The bot responds to every message from the user. |

The gating helpers live in `bot/router_components/gating.py`:
- `mentions_bot(message, bot_user_id)` — checks Discord `message.mentions` for the bot's user ID.
- `is_reply_to_bot(message, bot_user_id)` — checks if the message is a reply to a bot-authored message.
- `strip_leading_bot_mention(content, bot_user_id)` — removes the `<@id>` / `<@!id>` prefix before passing text to downstream handlers.

The `BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO` config flag (default `True`) controls whether the mention gate is enforced in guild contexts.

### How to Add a Handler

1. **Create the handler class** in `bot/routing/`. Implement `can_handle()` and `handle()` matching the `RouteHandler` protocol.
2. **Export it** from `bot/routing/__init__.py`.
3. **Register in the routing table** inside `Router.dispatch_message()` in `bot/router.py`, in the correct priority order (specific before general).
4. **Add tests** in `tests/test_router_*` or `tests/core/test_router.py`.
5. **Add a feature flag** in `bot/server_features.py` if the handler depends on an optional subsystem.

### Where NOT to Put New Logic

- **Do not** add routing logic to `bot/core/bot.py`. The bot class delegates to `MessageProcessor` and `Router` only.
- **Do not** create parallel router systems or bypass the `Router.dispatch_message()` pipeline.
- **Do not** import Discord library types directly into `bot/routing/` handlers — use `RouteContext` which abstracts the Discord message.
- **Do not** add business logic to `bot/core/message_processor.py` — it owns queuing and dedup only.

### X/Twitter URL Routing Special Case

X/Twitter (twitter.com, x.com, fxtwitter.com, vxtwitter.com) URL routing has its own dedicated pipeline within the router, separate from the generic URL/web extraction path. It includes:

- **Syndication cache** (`_syn_cache` + `_syn_locks`) to avoid redundant API calls for the same tweet.
- **X API client** (`bot/x_api_client.py`) for direct tweet data retrieval.
- **STT pipeline for X video**: X video content goes through Whisper transcribe with its own timeout budget (`X_STT_MIN_TIMEOUT_S` = 120s through `X_STT_MAX_TIMEOUT_S` = 900s, RTF-based calculation).
- **Frontend URL canonicalization**: fx/vx/vx variants are normalized to canonical x.com/twitcher.com status URLs before processing.
- **Dedicated helper module**: `bot/router_components/` contains 40+ X-specific functions for URL extraction, oEmbed fallbacks, media resolution, tweet text formatting, and syndication payload building.

Do not route X/Twitter URLs through the generic `process_url` or `web_extractor` paths — they have a specialized pipeline with caching, transcription, and media selection semantics.

### Conversational Image Editing (mention/reply img2img)

An addressed message (mention/DM/reply) that includes an image and reads as an
edit *instruction* ("give him a beard", "remove the background") is routed to
img2img instead of VL analysis or `/imgedit`. See `docs/CONVERSATIONAL_IMAGE_EDIT.md`
for the full design; summary:

- Decision point: `Router._maybe_route_conversational_edit()`, called from
  `_process_multimodal_message_internal()` **before** the existing
  reply-image → VL-perception branch, gated on `(is_dm or mentioned_me or
  is_reply) and combined_count >= 1` (same image-presence signal the VL
  branch already computes) so it never runs on unaddressed traffic.
- Intent heuristic: `bot.router_components.conversational_edit.classify_edit_intent()`
  (keyword-based v1; analysis/question phrasing always wins ties).
- Image sourcing: `resolve_edit_source_image()` — current message attachment/embed,
  then the replied-to message's attachment/embed, then a bare image URL in the
  triggering text; enforces `MAX_ATTACHMENT_SIZE_MB` via `download_robust_image()`.
- Execution: reuses `VisionOrchestrator.submit_job()` unchanged — same safety
  filter and budget ledger as `/imgedit`, no new provider/pool. Result files are
  attached to `BotAction.files`, which `LLMBot._execute_action()` now forwards
  into the standard reply-with-reference send path (previously a dead field).
- Feature gate: `!feature image_editing on|off` (`bot/server_features.py`), plus
  `VISION_CONVERSATIONAL_EDIT_ENABLED` (global kill switch, `.env.example`).

