# Architecture Decision Record (ADR)

## Overview

This document describes the architecture of the Discord bot, focusing on
the core components, data flows, and important design decisions.

## High-Level Flow

Discord events (messages, interactions) flow through the following layers:

1. **Gate**: Filters and validates incoming events (spam detection, permissions).
2. **Queue**: Manages backpressure and task prioritization.
3. **Router**: Routes events to appropriate handlers (commands, vision, etc.).
4. **Backend**: Executes the actual logic (LLM calls, TTS, STT, vision, etc.).
5. **Response**: Sends the response back to Discord.

## Component Ownership

| Component | Owner | Responsibilities |
|-----------|-------|------------------|
| bot/core/bot.py | Core bot initialization | Discord event loop, command registration |
| bot/core/router.py | Event routing | Gate, queue, router, handler dispatch |
| bot/commands/ | Command cogs | Command handlers, permissions |
| bot/backend/ | Backend services | LLM, TTS, STT, vision, web extraction |
| bot/memory/ | Memory management | Persistence, RAG, context |
| bot/vision/ | Vision processing | Image/video analysis |
| bot/stt/ | Speech-to-text | Audio transcription |
| bot/tts/ | Text-to-speech | Audio generation |
| bot/queue/ | Task queue | Backpressure management |
| bot/gateway/ | External service clients | HTTP clients, API integrations |

## Explicit "Must Not Own" Boundaries

- **bot/core/router.py** must not depend on specific command implementations.
- **bot/commands/** must not directly instantiate backend services; use dependency injection.
- **bot/backend/** must not depend on Discord library directly; use abstractions.
- **bot/memory/** must not depend on specific backend implementations.
- **bot/vision/** must not depend on Discord library directly.

## Data Flow

### Message Processing
1. Discord event → `bot/core/bot.py` → `Router`
2. Router applies gate → queue → handler
3. Handler may use backend services (LLM, TTS, STT, vision)
4. Response sent back to Discord

### Memory/Context/RAG Flow
1. User message → context extraction → memory lookup
2. RAG query → ChromaDB → vector embeddings → LLM
3. Context stored in memory → used for future responses

### Provider Fallback Ladders
- **Text**: OpenRouter → fallback to specific models
- **Vision**: Together.ai → Novita.ai → fallback to other providers
- **STT**: Whisper → fallback to other providers
- **TTS**: Provider-specific fallback ladders

## Degraded Mode Behavior

When components fail, the bot should:
- Continue operating with available components
- Gracefully degrade functionality (e.g., disable vision if vision service unavailable)
- Report degraded status via `/status` command
- Retry with fallback providers when possible

## Admin Permission Model

Centralized permission checks in `bot/core/permissions.py`:
- `admin_required` decorator for commands
- Supports bot owners, guild admins, and configured admin role
- Slash commands: ephemeral denial
- Prefix commands: normal reply (no ephemeral)

## URL/PROMPT Safety Model

- All user-provided URLs are validated against private IP ranges (SSRF protection)
- External content fetched via `safe_fetch` (bot/url_safety.py)
- Fetched content is wrapped as untrusted external content in prompts
- Output sanitization applied to all responses

## How to Add a Backend

1. Create a new module in `bot/backend/` (e.g., `bot/backend/my_backend.py`)
2. Implement the backend interface (e.g., `TextBackend`, `TTSEngine`)
3. Register in `bot/backend/__init__.py`
4. Add to config and enable via environment variables

## How to Add a Command Cog

1. Create a new cog file in `bot/commands/`
2. Define a cog class inheriting from `commands.Cog`
3. Implement command methods with `@commands.command` or `@app_commands.command`
4. Use `@admin_required` for admin-only commands
5. Register in `bot/commands/__init__.py`

## How to Add a Media Route

1. Create a new handler in `bot/routing/` (after router extraction)
2. Implement `can_handle(ctx)` and `handle(ctx)` methods
3. Register with the router

## Important Env Vars/Config Controls

- `OWNER_IDS`: Bot owner Discord IDs
- `ADMIN_ROLE_ID`: Configurable admin role ID
- `TEXT_BACKEND`, `VISION_BACKEND`, `STT_ENGINE`, `TTS_BACKEND`: Backend configurations
- `RAG_ENABLED`: Enable/disable RAG
- `MAX_USER_MEMORY`, `MAX_SERVER_MEMORY`: Memory limits
- `TIMEOUT`, `VL_REQUEST_TIMEOUT`, `VL_NOTES_TIMEOUT_S`: Timeouts
- `TEXT_FALLBACK_TIMEOUTS`, `TEXT_FALLBACK_MAX_ATTEMPTS`: Text fallback ladders
- `VISION_FALLBACK_MODELS`: Vision fallback ladder
- `SCREENSHOT_API_KEY`, `SCREENSHOT_API_URL`: Screenshot API configuration
- `CHROMA_URL`: ChromaDB connection URL (integration tests only)
- `WEBEX_ENABLE_TIER_B`: Web extraction tier B enablement

