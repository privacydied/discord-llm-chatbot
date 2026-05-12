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

