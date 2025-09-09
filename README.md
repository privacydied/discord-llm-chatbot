# discord-llm-bot

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview
- Production‑oriented Discord bot that blends chat, search, RAG, and multimodal (vision, TTS/STT) features with robust logging, observability, and reliability.
- Built on discord.py 2.x with message‑commands and selective slash commands for vision workflows.
- Backends: OpenAI/OpenRouter or local Ollama for text; optional RAG over ChromaDB; OCR/STT for media.
- Emphasizes clean startup, dual‑sink logging (Rich + JSONL), graceful shutdown, retries, and Prometheus metrics.

## Features
- Discord commands:
  - Text: general chat, admin/config, memory, search, screenshots.
  - RAG: hybrid vector+keyword search over a local KB with lazy/eager loading.
  - Media: video transcription, OCR fallbacks, and X/Twitter extractors.
  - Vision: slash commands for image/video generation (Together/Novita).
  - TTS: Kokoro‑ONNX speech synthesis and voice message publishing.
- Backends and routing:
  - Text: OpenAI/OpenRouter (default), or Ollama.
  - RAG: ChromaDB (hybrid search + background indexing).
  - Vision: providers (Together, Novita) with budgets, retries, and artifacts.
  - STT: faster‑whisper/whispercpp orchestration with timeouts and caching.
- Ops and reliability:
  - Dual logging sinks: Rich console + structured JSONL at `logs/bot.jsonl`.
  - Logging enforcer, sensitive data scrubbing, graceful shutdown.
  - Prometheus metrics (optional HTTP server).
  - Extensive test suite (pytest + pytest‑asyncio).

## Architecture

```mermaid
flowchart LR
  A[Discord Gateway] --> B[LLMBot (discord.py)]
  B --> C[Router]
  C -->|Text| D1[OpenAI/OpenRouter]
  C -->|Text (local)| D2[Ollama]
  C -->|RAG| E[Hybrid Search (ChromaDB)]
  C -->|Vision| F[Vision Orchestrator\nTogether/Novita]
  C -->|Media| G1[STT Orchestrator] --> G2[faster-whisper/whispercpp]
  C -->|PDF/OCR| H1[PyMuPDF] --> H2[Tesseract OCR (optional)]
  B --> I[Commands/Cogs]
  B --> J[Prometheus Metrics]
  B --> K[Logging: Rich + JSONL]
  subgraph Files/Storage
    L1[kb/]
    L2[chroma_db/]
    L3[vision_data/, logs/, user_profiles/, server_profiles/]
  end
```

- Entrypoint: `run.py` → `bot.main:run_bot()` → async `main()` → `LLMBot.start()`.
- Intents: message_content, guilds, members, voice_states checked at startup.
- Commands loaded in `LLMBot.setup_hook()` from `bot/commands/*`; vision slash commands via `discord.app_commands`.
- Config loads from `.env` and validates required keys; prompt files loaded from disk.

## Getting Started

### Prerequisites
- Python 3.11+
- System tools (feature‑based):
  - Tesseract OCR (tesseract, language packs like `tesseract-data-eng`)
  - Playwright Chromium (auto‑install attempted; may need `uv run playwright install chromium`)
  - Poppler utils for some OCR flows (`pdftoppm`)
- Discord Developer Portal:
  - A bot application with Bot + `applications.commands` scopes
  - Message Content Intent enabled

### Installation

#### UV (preferred)
```bash
# Create and activate an isolated venv
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip sync requirements.txt

# Optionally, install the package in editable mode
uv pip install -e .

# Ensure Playwright browser (Chromium) for screenshot/vision flows
uv run playwright install chromium
```

#### pip + venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Configuration
Copy `.env.example` to `.env` and fill in secrets and paths.
```bash
cp .env.example .env
```
At minimum: `DISCORD_TOKEN`, `PROMPT_FILE`, `VL_PROMPT_FILE`, and `OPENAI_API_KEY` (if using OpenAI/OpenRouter).

### Environment Variables (Core)

| Name | Required | Default | Description | Example |
| ---- | -------- | ------- | ----------- | ------- |
| DISCORD_TOKEN | Yes | — | Discord bot token | A1B2... |
| TEXT_BACKEND | No | openai | Text backend: openai or ollama | openai |
| OPENAI_API_KEY | Maybe | — | API key for OpenAI/OpenRouter | sk-or-... |
| OPENAI_API_BASE | No | https://openrouter.ai/api/v1 | API base for OpenRouter | https://openrouter.ai/api/v1 |
| OPENAI_TEXT_MODEL | No | — | Text model ID (OpenRouter/OpenAI) | deepseek/... |
| OLLAMA_BASE_URL | No | http://localhost:11434 | Ollama base URL | http://localhost:11434 |
| OLLAMA_MODEL | No | llama3 | Default local model | qwen3 |
| PROMPT_FILE | Yes | — | System prompt file path | prompts/prompt-yoroi-super-chill.txt |
| VL_PROMPT_FILE | Yes | — | Vision system prompt file path | prompts/vl-prompt.txt |
| BOT_PREFIX | No | ! | Message command prefix (comma‑separated allowed) | !,? |
| LOG_LEVEL | No | INFO | Logging level | DEBUG |
| LOG_JSONL_PATH | No | logs/bot.jsonl | JSONL log path | logs/bot.jsonl |
| PROMETHEUS_ENABLED | No | true | Enable Prometheus metrics | true |
| PROMETHEUS_PORT | No | 8000 | Metrics port | 8000 |
| STREAMING_ENABLE | No | false | Streaming status embeds | true |

More environment variables (RAG, budgets, retries, streaming, etc.) are documented in `.env.example` and `bot/config.py`.

## Quickstart

### UV
```bash
# 1) venv + deps
uv venv --python 3.11
source .venv/bin/activate
uv pip sync requirements.txt

# 2) configure env
cp .env.example .env

# 3) run
uv run python -m bot.main
# or: python run.py
```

### pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
cp .env.example .env
python -m bot.main
```

### Invite the bot
- Scopes: `bot`, `applications.commands`
- Permissions (minimal): Send Messages, Attach Files, Embed Links, Read Message History
- Privileged intents: enable “Message Content Intent” in the Developer Portal

## Usage

### Local run
```bash
uv run python -m bot.main
```
On first start, pre‑flight checks validate token, intents, and Playwright availability.
If Playwright isn’t present, auto‑install is attempted; otherwise run:
```bash
uv run playwright install chromium
```

### Slash commands (Vision)
- `/image`, `/imgedit`, `/video`, `/vidref` (see `bot/commands/vision_commands.py`)
- Set `VISION_ENABLED=true` and provide `VISION_API_KEY` for allowed providers
- Slash command propagation can take time globally; per‑guild sync is not explicitly coded

### Message commands (examples)
- Search: `!search <query>`
- RAG: `!rag ...` (see `bot/commands/rag_commands.py`)
- Admin/Config: `!reload-config`, `!config-status`, `!alert`
- TTS: `!tts ...`, `!say ...`
- Video: `!watch <url>` (transcribe), `!video-help`
- Screenshot: `!ss <url>`
- Memory: `!memory ...`

## Logging & Observability
- Dual sinks:
  - Rich console with pretty tracebacks (locals on DEBUG)
  - JSONL structured file at `logs/bot.jsonl` (keys: ts, level, name, subsys, guild_id, user_id, msg_id, event, detail)
- Enforcer requires exactly two handlers named `pretty_handler` and `jsonl_handler` or startup aborts
- SensitiveDataFilter scrubs common secret keys in logged dict extras
- Prometheus metrics initialized if `PROMETHEUS_ENABLED=true` (port from `PROMETHEUS_PORT`)

## RAG / LLM Providers

### Text backends
- `openai` (OpenAI/OpenRouter): requires `OPENAI_API_KEY`, optionally `OPENAI_API_BASE`
- `ollama` (local): `OLLAMA_BASE_URL`, `OLLAMA_MODEL` or `TEXT_MODEL`

### RAG
- Hybrid search over ChromaDB (vector + keyword) with configurable thresholds and weights
- Control with `RAG_*` envs; `kb/` as default source and `chroma_db/` for index storage (configurable)

### Vision
- Providers: Together, Novita (set `VISION_ALLOWED_PROVIDERS`, `VISION_DEFAULT_PROVIDER`, and `VISION_API_KEY`)
- Artifacts and job state under `vision_data/…`

## Deployment
- No top‑level Dockerfile/compose for the bot is included. A TTS service Dockerfile exists at `tts/service/Dockerfile`.
- For production: supervise the process, configure log rotation for `logs/*.jsonl`, and set restricted perms on data files.

## Security & Privacy
- Never commit secrets; use `.env` locally and secret stores in production
- Message content intent processes user messages; ensure policy compliance
- Prompt/context files may contain sensitive data; restrict file permissions
- Logging scrubs common secrets but never log raw credentials

## Troubleshooting
Dependency alignment is critical.

- DOCX parsing: `ImportError: No module named docx.Document` → ensure `python-docx` is installed (this project pins `python-docx`, not the unrelated `docx` package)
- PDF parsing: `ImportError: fitz` → ensure `pymupdf` is installed (this project pins `pymupdf`, not the unrelated `fitz` package)
- Async tests failing immediately → ensure `pytest-asyncio` is installed and pytest reads `pytest.ini` (`asyncio_mode=auto`)
- Playwright/browser not found → `uv run playwright install chromium`
- Poppler missing → OCR fallback uses `pdftoppm`; install poppler‑utils on your system
- Missing intents/scopes → enable Message Content Intent; re‑invite bot with `applications.commands` + `bot` scopes

## Contributing
- Keep functions tidy and add tests where possible
- Respect the dual‑sink logging setup; do not log secrets
- Update `.env.example` and docs under `docs/` when adding features
- Describe changes, risks, and required env/schema updates in PRs

## License
MIT — see [LICENSE](LICENSE).

## Acknowledgments
- discord.py
- Rich
- Kokoro‑ONNX
- ChromaDB
- PyMuPDF, Tesseract OCR
- Playwright

