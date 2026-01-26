# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Run the bot
uv run python -m bot.main

# Run tests
uv run -m pytest -q

# Run a single test file
uv run -m pytest tests/test_router.py -v

# Run a specific test
uv run -m pytest tests/test_router.py::test_function_name -v

# Lint and format
uv run ruff check .
uv run ruff format .

# Security scan
uv run bandit -q -r bot

# First-time setup: install Playwright for web screenshots
uv run playwright install chromium
```

**Always use `uv run`** for Python execution to ensure locked interpreter/env reproducibility.

## Non-Negotiables

1. **Repo hygiene**: Ad-hoc scripts → `utils/`. Tests → `tests/`. **Never** drop new files in repo root.
2. **Logging**: Must use **Dual Sink Strategy** (Rich console + JSONL file). Startup enforcer aborts if both handlers aren't active.
3. **Tracebacks**: When errors are pasted, respond with **root cause, minimal repro, and a ready patch**.
4. **Autonomy**: Deliver patches without permission gates. Pause only for destructive/ambiguous changes (data loss, API breaks).

## Code Quality Gates (CSD)

Fail CI unless justified:
- Function > **30 lines** (excl. docstring) or nesting depth > **3**
- File > **300 SLOC** (hard cap at **1600 lines**)
- Class > **5 public methods**

Refactor strategies: extract method/object, split adapters, strategy pattern, pipeline composition.

## Error Handling (REH)

- **Timeouts**: External calls default ≤ 10s; no unbounded waits
- **Retries**: Exponential backoff with jitter for transient I/O; cap retries; log attempts
- **Typed exceptions** per boundary (e.g., `DiscordGatewayError`, `StorageError`)
- Top-level guard: visible crash banner on unrecoverable errors; non-zero exit

## RAT Tags for Commits/PRs

Use bracketed tags to indicate rules applied:
- **CA** – Clean Architecture
- **REH** – Robust Error Handling
- **CSD** – Code Smell Detection
- **IV** – Input Validation
- **RM** – Resource Management
- **CMV** – Constants over Magic Values
- **SFT** – Security-First Thinking
- **PA** – Performance Awareness

Example: `fix(router): handle timeout in X fetch [REH][PA]`

## Architecture Overview

### Entrypoint Flow
`run.py` → `bot.main:run_bot()` → `LLMBot.start()` → `setup_hook()` loads commands/cogs

### Core Components

**bot/core/bot.py - LLMBot class**
- Extends `discord.py commands.Bot`
- Manages context via `ContextManager` and `EnhancedContextManager`
- Tracks per-user message queues for sequential processing
- Commands autoload in `setup_hook()`

**bot/router.py - Router**
- Central message dispatcher for sequential multimodal processing
- Collects all input items (attachments, URLs, embeds) via `collect_input_items()` and processes sequentially
- Delegates to specialized handlers: `_handle_image`, `_handle_video_url`, etc.
- All handler results flow through `_flow_process_text()` for unified text processing

**bot/commands/** - Cog modules
- `rag_commands.py` - RAG/knowledge base operations
- `vision_commands.py` - Slash commands for `/image`, `/video`, `/imgedit`, `/vidref`
- `tts_cmds.py` - Text-to-speech: `!tts`, `!say`, `!speak`
- `admin_alert_commands.py` - Admin/alerting functionality

### Subsystems

**Vision (bot/vision/)**
- `orchestrator.py` / `orchestrator_v2.py` - Job orchestration for image/video generation
- `unified_adapter.py` - Provider abstraction (Together, Novita)
- `budget_manager.py` - Per-user/per-server spend tracking

**RAG (bot/rag/)**
- `hybrid_search.py` - Vector + keyword search over local KB
- `chroma_backend.py` - ChromaDB integration
- `embedding_interface.py` - Embedding model abstraction

**Memory (bot/memory/)**
- `context_manager.py` - Basic conversation context
- `enhanced_context_manager.py` - Multi-user conversation tracking with token limits
- `profiles.py` - User/server profile persistence
- `thread_tail.py` - Discord thread context collection

**TTS (bot/tts/)**
- `interface.py` - Main TTS manager
- `kokoro_direct_fixed.py` - Kokoro ONNX TTS engine
- `eng_g2p_local.py` - English grapheme-to-phoneme

**AI Backends**
- `bot/openai_backend.py` - OpenAI/OpenRouter integration
- `bot/ollama.py` - Local Ollama support
- `bot/ai_backend.py` - Backend abstraction layer

### Key Data Flows

1. **Message → Response**: Discord message → `LLMBot.on_message` → `Router.handle_message()` → modality detection → specialized handlers → `_flow_process_text()` → LLM call → Discord reply

2. **Multimodal Processing**: `collect_input_items()` extracts all media → sequential processing with per-item handlers → results aggregated → final LLM synthesis

3. **X/Twitter URLs**: Detected via `_detect_x_twitter_media()` → thread unroll via `unroll_author_thread()` → route to STT/VL/text extraction based on media type

## Configuration

- Primary config via `.env` (see `.env.example` for all options)
- `bot/config.py` - Config loading and validation
- `bot/config_reload.py` - Dynamic hot-reload support
- Prompt files loaded from `prompts/` directory

## Logging Specification

**Dual Sink Strategy (mandatory)**:

1. **Pretty Console (RichHandler)**: Rich tracebacks with locals on DEBUG, timestamps with ms precision
2. **Structured JSONL** (`logs/bot.jsonl`): Keys: `ts, level, name, subsys, guild_id, user_id, msg_id, event, detail, message`

Never log secrets/PII; scrub tokens; hash IDs when required.

## Operational Discipline

- **Understand before coding**: List files, map entrypoints, identify env vars/config before edits
- **Follow existing patterns**: Match style, structure, and conventions already in the codebase
- **Surface edge cases immediately**: Lock down inputs/outputs/constraints before changes
- **No magic values**: Extract to constants or Enums
- **Async I/O**: No blocking inside event loops; prefer async for network/file concurrency

## Testing

- pytest with `asyncio_mode = auto`
- Markers: `unit`, `integration`, `slow`
- Tests live in `tests/` mirroring `bot/` structure
- Coverage target: ≥ 85%
