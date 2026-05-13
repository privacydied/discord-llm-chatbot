# discord-llm-chatbot

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Production-ready Discord bot** that blends chat, search/RAG, and multimodal (vision, OCR, TTS/STT). Built on `discord.py 2.x` with robust routing, retries, structured logs, and optional Prometheus metrics.

> Text via OpenAI/OpenRouter, **NVIDIA NIM**, or local Ollama. RAG via ChromaDB. Vision via Together/Novita. STT via faster-whisper/whispercpp. TTS via Kokoro. OCR via PyMuPDF + Tesseract. Server Archive via SQLite. Admin alerts, config hot-reload, voice publishing, and syndication-based X/Twitter text extraction.

---

## ✨ Features

* **Chat & Tools**
  * General chat, search, screenshots, memory, admin/config.
  * Hybrid RAG (vector + keyword) over a local ChromaDB KB.
  * Hot-reload of config from `.env` file changes.
  * Admin alert sessions in DMs for interactive configuration.
  * Janitor for cache and log cleanup.
* **Multimodal Input**
  * Sequential multimodal processing: extracts ALL attachments, URLs, and embeds in message order.
  * **Images** → Vision/VL models (see.py) for evidence extraction.
  * **Video/Audio** → STT pipeline (yt-dlp + ffmpeg + faster-whisper/whispercpp, or YouTube transcript-first).
  * **PDFs** → PyMuPDF text extraction + Tesseract OCR fallback.
  * **Documents** → docx/txt ingestion.
  * **URLs** → Web extraction, X/Twitter syndication (fxtwitter/vxtwitter), screenshot fallback.
  * **X/Twitter** → Thread unrolling of author self-replies, syndication probes with image→VL routing.
* **Vision Generation**
  * Image & video generation with provider budgeting, safety filtering, job management, artifact caching, and cost tracking.
  * Provider ladder: Together.ai → Novita.ai.
  * Slash commands: `/image`, `/imgedit`, `/video`, `/vidref`.
* **Voice & Speech**
  * **TTS** via Kokoro (local, G2P-based) with phoneme vocabulary loading, IPA-based assets, and engine abstraction.
  * **STT** orchestrator with multi-provider support (local whisper cascade, caching, confidence thresholds).
  * **Native voice publishing** for Discord voice channels.
* **Memory & Archival**
  * Persistent user/server profiles with auto-save.
  * Curated memory service with semantic memory (explicit + inferred).
  * Memory distiller: background summarization of archived conversations.
  * Server Archive: SQLite-backed message archival with distillation and search.
* **Ops**
  * Dual-sink logging: Rich console + JSONL (`logs/bot.jsonl`) with secret scrubbing.
  * Graceful shutdown, backoff/retries, health monitoring, SLO tracking.
  * **Prometheus** (optional) for metrics — Discord retries, gate counters, vision/X routing, syndication tiers.
  * Public output safety: sanitizes all send/edit boundaries.
  * Per-user message queues with dedup guards and concurrency locks.
  * Resource monitoring (CPU, RAM, low-resource mode).

---

## 🧭 Architecture (high-level)

```mermaid
flowchart TD
    %% ── Entry & Bootstrap ──────────────────────────────────────
    RUNPY["run.py"] --> MAIN["bot.main (async entry)"]
    MAIN --> CLI["CLI args: --version / --config-check / --debug"]
    MAIN --> CFG["config: load_config() + hot-reload watcher"]
    MAIN --> PREFLIGHT["pre-flight checks (env, Playwright, ports)"]
    MAIN --> BOT["LLMBot (discord.py Bot)"]

    %% ── Setup Hook ─────────────────────────────────────────────
    BOT --> SH["setup_hook() — asynchronous bootstrap"]
    SH --> TSHIM["torch compat shim (deferred)"]
    SH --> SANITY["public-output safety hooks (monkey-patch send/edit)"]
    SH --> MP["MessageProcessor (per-user queue + dedup)"]
    SH --> METRICS["Metrics: PrometheusMetrics or NoopMetrics"]
    SH --> CM["ContextManager + EnhancedContextManager"]
    SH --> COGS["Command Cogs (12+ cogs loaded dynamically)"]
    SH --> ROUTER["Router (multimodal dispatch)"]
    SH --> VO["VisionOrchestrator (eager start)"]
    SH --> TTS["TTSManager (Kokoro engine)"]
    SH --> RAG["RAG (optional eager ChromaDB load)"]

    %% ── Message Flow ───────────────────────────────────────────
    DG["Discord Gateway Events (on_message)"] --> MP

    subgraph MessageProcessor["MessageProcessor (per-user orchestration)"]
        DEDUP["dedup guard (OrderedDict, FIFO, 1000 max)"]
        ARCHIVE["best-effort server-archive enqueue"]
        ALERT["admin alert DM suppression gate"]
        QUEUE["per-user asyncio.Queue"]
        WORKER["_process_user_messages (one-at-a-time, 300s idle)"]
        DEDUP --> ARCHIVE --> ALERT --> QUEUE --> WORKER
    end

    WORKER --> DISPATCH["LLMBot._process_single_message(message)"]
    DISPATCH --> GATE["gating: prefiler gates (DMs, threads, ignore lists)"]
    GATE --> RENTRY["fast-path: bot replies to self (re-entry guard)"]
    RENTRY --> CMDCHECK["command delegation via command_parser"]
    CMDCHECK --> COGS
    RENTRY --> MM["Multimodal collector (collect_input_items)"]

    %% ── Component Subgraphs ────────────────────────────────────
    subgraph InputItems["Sequential Multimodal Input"]
        II["InputItem(source=attachment|url|embed)"]
        II_AT["🖼 Attachments (image, audio/video, PDF, txt/docx)"]
        II_URL["🔗 URLs (general, video, X/Twitter, screenshot)"]
        II_EMBED["📎 Discord embeds (auto-resolved)"]
    end

    MM --> II_AT
    MM --> II_URL
    MM --> II_EMBED

    %% ── Handlers ───────────────────────────────────────────────
    II_AT --> ATT_CLASS["attachment_classifier → bucket: image/pdf/doc/media"]
    ATT_CLASS --> HANDLE_IMG["handle_image → see.py"]
    ATT_CLASS --> HANDLE_STT["handle_audio_video_video_url → STT pipeline"]
    ATT_CLASS --> HANDLE_DOC["handle_document → document_ingest"]

    II_URL --> URL_CLASS["url_classifier → classified: twitter/video/general"]
    URL_CLASS --> X_ROUTE["x_routing: syndication probe → oembed → raw fetch"]
    X_ROUTE --> X_UNROLL["x_thread_unroll: author self-reply chain"]
    URL_CLASS --> VIDEO_ROUTE["video_ingest → yt-dlp → ffmpeg → whisper"]
    URL_CLASS --> WEB_ROUTE["web_extractor (Playwright/curl)"]
    URL_CLASS --> SS_ROUTE["screenshot → playwright_remote (PW_SERVER_URL)"]

    %% ── Evidence Assembly ──────────────────────────────────────
    HANDLE_IMG --> EVID["EvidenceBundle (aggregated perception text)"]
    HANDLE_STT --> EVID
    X_ROUTE --> EVID
    X_UNROLL --> EVID
    WEB_ROUTE --> EVID
    SS_ROUTE --> EVID
    HANDLE_DOC --> EVID

    subgraph STTPipeline["STT Pipeline (multi-provider)"]
        YTF["YouTube transcript-first (fast path)"]
        YTDLP["yt-dlp download"]
        FFMPEG["ffmpeg preprocess (speedup, trim)"]
        STT_ORCH["stt_orchestrator (single / cascade)"]
        LWHISPER["LocalWhisper (faster-whisper)"]
        CACHED_STT["single-flight cache (TTL)"]
        YTF -->|hit| CACHED_STT
        YTF -->|miss| YTDLP --> FFMPEG --> STT_ORCH --> LWHISPER --> CACHED_STT
        CACHED_STT --> EVID
    end

    %% ── Router Components ──────────────────────────────────────
    subgraph RouterComponents["Router Components (refactored helper layer)"]
        XR["x_routing.py: URL/X/media/syndication helpers"]
        CMP["compose.py: perception/text composition"]
        GT["gating.py: mention/reply guards"]
        IH["input_harvest.py: item extraction + normalization"]
        PA["prompt_access.py: prompt loading"]
        RT["runtime.py: compat/runtime access"]
    end

    %% ── Text Flow ──────────────────────────────────────────────
    EVID --> TF["_invoke_text_flow()"]
    TF --> CMP
    TF --> PA
    TF --> RT

    subgraph Knowledge["Context & Knowledge Assembly"]
        MEM["memory: explicit + distilled + relevant block"]
        MENTION["mention_context: quoted/mentioned users"]
        T_THREAD["thread_tail: conversation history in threads"]
        IMPL_ANCHOR["implicit_anchor: resolved reply targets"]
        RAG_SEARCH["RAG hybrid_search (vector + keyword)"]
        CTX["context_manager + enhanced_context_manager"]
    end

    TF --> MEM
    TF --> MENTION
    TF --> T_THREAD
    TF --> IMPL_ANCHOR
    TF --> RAG_SEARCH
    TF --> CTX

    subgraph Brain["Text Generation Pipeline"]
        BRAIN["brain_infer() → composed prompt"]
        BACKEND["ai_backend router"]
        OPENAI["OpenAI/OpenRouter/NIM text ladder"]
        NVIDIA["NVIDIA NIM backend"]
        OLLAMA["Ollama local backend"]
        RETRY["enhanced_retry (backoff, fallback ladder)"]
        STREAM["streaming async generator"]
    end

    MEM --> BRAIN
    MENTION --> BRAIN
    CTX --> BRAIN
    RAG_SEARCH --> BRAIN
    BRAIN --> BACKEND
    BACKEND --> OPENAI
    BACKEND --> NVIDIA
    BACKEND --> OLLAMA
    OPENAI --> RETRY
    NVIDIA --> RETRY
    OLLAMA --> RETRY
    RETRY --> STREAM

    %% ── Action Execution ───────────────────────────────────────
    STREAM --> ACT["BotAction (send, edit, TTS, voice, file)"]
    ACT --> PUBLIC_SAN["sanitize_public_message_payload"]
    PUBLIC_SAN --> TTS_PATH["optional TTS synthesis (Kokoro)"]
    PUBLIC_SAN --> DISC_SEND["Discord reply / edit (with retry)"]
    TTS_PATH --> VOICE_PUB["VoiceMessagePublisher (native voice)"]
    VOICE_PUB --> DISC_SEND
    DISC_SEND --> CTX_UPDATE["enhanced_context_manager.append_message"]

    %% ── Vision System ──────────────────────────────────────────
    subgraph Vision["Vision Generation System"]
        VIR["VisionIntentRouter (direct vs intent-based)"]
        VORCH["VisionOrchestrator (job queue, async exec)"]
        VSTORE["VisionJobStore (JSON persistence)"]
        VSF["VisionSafetyFilter (content moderation)"]
        VBM["VisionBudgetManager (cost quotas)"]
        VAC["VisionArtifactCache (dedup cache)"]
        VGW["VisionGateway (Together/Novita ladder)"]
        VPROV_T["Together provider"]
        VPROV_N["Novita provider"]
        MONITOR["VisionJobWatcher (poll + Discord upload)"]
        VORCH --> VSTORE
        VORCH --> VSF --> VBM --> VAC --> VGW
        VGW --> VPROV_T
        VGW --> VPROV_N
        VORCH --> MONITOR
    end

    COGS --> VIR
    ROUTER --> VORCH
    METRICS --> VORCH

    %% ── TTS System ─────────────────────────────────────────────
    subgraph TTS["TTS Engine (Kokoro)"]
        KOKORO_V8["Kokoro v8 engine"]
        G2P["eng_g2p_local (grapheme-to-phoneme)"]
        IPA["ipa_vocab_loader + kokoro_v1 vocabulary"]
        TTS_I18N["ipa_vocab_kokoro_v1 (localized IPA)"]
        TTS_STUB["TTS stub (fallback no-op)"]
        MANAGER["TTSManager (orchestrator, instrumentation)"]
        TTS_STATE["tts_state (per-user on/off, queues)"]
    end

    TTS_PATH --> MANAGER
    MANAGER --> KOKORO_V8
    KOKORO_V8 --> G2P --> IPA
    MANAGER --> TTS_I18N
    MANAGER --> TTS_STUB
    MANAGER --> TTS_STATE

    %% ── Background Tasks ───────────────────────────────────────
    subgraph Background["Background Tasks (spawn_background_tasks)"]
        MEM_SAVE["memory_profiler autosave (periodic .json)"]
        MEM_SERVICE["curated memory service (semantic store)"]
        DISTILLER["memory distiller (archive summarization)"]
        ARCH_SERVICE["server archive service (SQLite ingestion)"]
        JANITOR["janitor (cache/log cleanup, periodic)"]
        HEALTH["health check (memory, guilds, readiness)"]
        LOG_CLEANUP["log cleanup (30-day retention)"]
    end

    BOT --> MEM_SAVE
    BOT --> MEM_SERVICE
    BOT --> DISTILLER
    BOT --> ARCH_SERVICE
    BOT --> JANITOR
    BOT --> HEALTH
    BOT --> LOG_CLEANUP

    %% ── Server Archive ─────────────────────────────────────────
    subgraph ServerArchive["Server Archive System"]
        SA_ING["ingestion_queue (per-channel)"]
        SA_SVC["archive_service (live message enqueue)"]
        SA_STORE["SQLite store (messages, metadata)"]
        SA_SYNC["sync: periodic flush + compaction"]
        SA_SEARCH["search: full-text + semantic over archive"]
    end

    SA_SVC --> SA_ING --> SA_STORE
    SA_STORE --> SA_SYNC
    SA_STORE --> SA_SEARCH
    ARCHIVE --> SA_SVC

    %% ── Observability ──────────────────────────────────────────
    subgraph Observability["Observability"]
        RICH["Rich console (pretty)"]
        JSONL["JSONL file logs (structured, secret-scrubbed)"]
        PROM["Prometheus /metrics endpoint"]
        SLO["SLO monitor (latency percentiles)"]
        RES_MON["resource_monitor (CPU/RAM/low-resource)"]
        DIAG["maintenance diagnostics"]
    end

    SH --> RICH
    SH --> JSONL
    METRICS --> PROM
    BOT --> SLO
    BOT --> RES_MON
    BOT --> DIAG

    %% ── Storage ────────────────────────────────────────────────
    subgraph Storage["Files / Storage"]
        KB["kb/ + chroma_db/ (RAG vectors)"]
        CTXSTORE["context.json + enhanced_context.json"]
        CACHE_STT_FILES["cache/stt_* + cache/video_audio + cache/youtube_transcripts"]
        PROF["user_profiles/ + server_profiles/"]
        VDATA["vision_data/ + logs/"]
        MEMDB["memory/ (semantic store JSON)"]
        ARCHDB["archive/ (SQLite)"]
    end

    RAG_SEARCH --> KB
    TF --> CTXSTORE
    YTF --> CACHE_STT_FILES
    YTDLP --> CACHE_STT_FILES
    FFMPEG --> CACHE_STT_FILES
    STT_ORCH --> CACHE_STT_FILES
    BOT --> PROF
    VO --> VDATA
    RICH --> JSONL
    MEM_SERVICE --> MEMDB
    SA_STORE --> ARCHDB

    %% ── Router Components wiring ───────────────────────────────
    ROUTER --> RouterComponents
    XR --> X_ROUTE
    XR --> X_UNROLL
    IH --> MM
    CMP --> TF
    PA --> BRAIN
    GT --> GATE
    RT --> TF
```

Entrypoint: `run.py` → `bot.main:run_bot()` → `asyncio.run(main_with_cleanup())` → `LLMBot.start()` → `setup_hook()`.

Commands autoload in `LLMBot.setup_hook()` via `bot.commands.__init__:setup_commands()`; vision slash commands via `discord.app_commands`. Config is `.env`-driven with hot-reload via filesystem watcher.

---

## 🚀 5-Minute Quickstart

### 1) Create the bot in Discord

* In **Developer Portal**: create an app → add **Bot**.
* Enable **Message Content Intent** (and others if you use them).
* Note the **TOKEN** and **CLIENT_ID**.

**Invite URL** (replace placeholders):

```
https://discord.com/api/oauth2/authorize?client_id=<CLIENT_ID>&scope=bot%20applications.commands&permissions=<PERMISSIONS_INT>
```

Minimal permissions: Send Messages, Embed Links, Attach Files, Read Message History.

### 2) Clone & create env

```bash
git clone https://github.com/privacydied/discord-llm-chatbot.git 
cd discord-llm-chatbot
cp .env.example .env
```

### 3) Install (choose one)

**uv (recommended)**

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip sync requirements.txt
```

**pip**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

### 4) Fill the minimum config

In `.env` set:

* `DISCORD_TOKEN=<token>`
* `PROMPT_FILE=prompts/prompt-yoroi-super-chill.txt`
* `VL_PROMPT_FILE=prompts/vl-prompt.txt`
* **Either** set OpenRouter/OpenAI: `OPENAI_API_KEY=...`
  **Or** local: `TEXT_BACKEND=ollama`, `OLLAMA_MODEL=llama3`

### 5) Run

```bash
uv run python -m bot.main
# first run may need: uv run playwright install chromium
```

**Smoke test**: bot responds in a server DM with `!help` (or one of the commands below).

---

## ⚙️ Configuration (core)

> Full set lives in [docs/ENV_VARS.md](docs/ENV_VARS.md), `.env.example`, and `bot/config.py`. Here are the essentials:

| Name                 | Required | Default                        | Example                 | Notes                                 |
| -------------------- | -------- | ------------------------------ | ----------------------- | ------------------------------------- |
| `DISCORD_TOKEN`      | ✅        | —                              | `A1B2...`               | Bot token                             |
| `BOT_PREFIX`         | ❌        | `!`                            | `!,?`                   | Comma-separated allowed               |
| `TEXT_BACKEND`       | ❌        | `openai`                       | `ollama`                | Text provider                           |
| `OPENAI_API_KEY`     | "Maybe"  | —                              | `sk-or-...`             | Needed for OpenAI/OpenRouter          |
| `OPENAI_API_BASE`    | ❌        | `https://openrouter.ai/api/v1` | custom                  | OpenRouter base                       |
| `OPENAI_TEXT_MODEL`  | ❌        | —                              | `deepseek/...`          | Model id when using OpenRouter/OpenAI |
| `OLLAMA_BASE_URL`    | ❌        | `http://localhost:11434`       | custom                  | Local Ollama                          |
| `OLLAMA_MODEL`       | ❌        | `llama3`                       | `qwen3`                 | Local model                           |
| `PROMPT_FILE`        | ✅        | —                              | `prompts/...`           | System prompt (text)                  |
| `VL_PROMPT_FILE`     | ✅        | —                              | `prompts/vl-prompt.txt` | Vision system prompt                  |
| `LOG_LEVEL`          | ❌        | `INFO`                         | `DEBUG`                 | Log verbosity                         |
| `LOG_JSONL_PATH`     | ❌        | `logs/bot.jsonl`               | custom                  | Structured logs                       |
| `PROMETHEUS_ENABLED` | ❌        | `true`                         | `false`                 | Metrics switch                        |
| `PROMETHEUS_PORT`    | ❌        | `8000`                         | `9100`                  | Metrics port                          |

**Feature-based prerequisites**

* **OCR/PDF**: Tesseract (`tesseract`, language packs), PyMuPDF, Poppler (`pdftoppm`).
* **Tier-B Web**: Playwright Chromium (auto-installed on first run or `uv run playwright install chromium`).
* **STT**: `ffmpeg` recommended for robust media handling.
* **TTS**: Kokoro (bundled) — requires no additional services.
* **Voice**: Discord voice channel permissions for native voice publishing.

---

### Twitter/X Thread Unroll (author self-replies)

- Enabled by default. Toggle off with `TWITTER_UNROLL_ENABLED=false`.
- Limits (env): `TWITTER_UNROLL_MAX_TWEETS` (default 30), `TWITTER_UNROLL_MAX_CHARS` (default 6000), `TWITTER_UNROLL_TIMEOUT_S` (default 15).
- Behavior: When a `x.com/twitter.com` status URL is shared, the bot collects the author's self-reply chain (contiguous) and packages it as a single context block; on any failure, it silently falls back to existing single-tweet handling. Non-Twitter links unaffected.
- Syndication probe: fxtwitter/vxtwitter tiers with image→VL routing for rich content extraction.
- Validate manually:
  - Post a long author thread → bot replies with consolidated context (count shown as [i/N] lines).
  - Post a single tweet → behavior unchanged.
  - Interleaved replies by others → only author tweets included.
  - Toggle `TWITTER_UNROLL_ENABLED=false` → identical to prior behavior.

---

## 🧰 Usage

**Message commands (examples)**

* Image generation: `!img <prompt>` (alias `!image`)
* Search: `!search <query>`
* Screenshot: `!ss <url>`
* Video transcription: `!watch <url>` (aliases `!transcribe`, `!listen`)
* TTS and speech: `!tts <text>` / `!tts on|off` / `!say <text>` / `!speak <text>`
* Memory: `!memory add <content>`, `!memory list`, `!memory clear`
* Context & privacy: `!context_reset`, `!context_stats`, `!privacy_optout`, `!privacy_optin`
* Admin/Config: `!reload-config`, `!config-status`, `!alert`, `!rag <subcommand>`
* Janitor: `!janitor run`, `!janitor status`
* Server Archive: `!archive search <query>`, `!archive stats`

**Slash (vision)**

* `/image`, `/imgedit`, `/video`, `/vidref` (enable `VISION_ENABLED=true` and provider key(s)).

---

## 📈 Observability

* **Logs (two sinks)**
  * Pretty console (Rich)
  * JSONL file: `logs/bot.jsonl` (keys: ts, level, name, subsys, guild_id, user_id, msg_id, event, detail)
  * Secrets scrubbed by default filter.
* **Metrics**
  * Enable with `PROMETHEUS_ENABLED=true`, port via `PROMETHEUS_PORT`.
  * Discord HTTP retry counters, gate allowed/blocked, X routing/syndication tiers, vision routing, TTS/STT instrumentation.
* **Health**
  * SLO monitor: latency percentiles for text/STT/vision flows.
  * Resource monitor: CPU/RAM/low-resource mode support.
  * Maintenance diagnostics: `!diagnostics` for runtime state.

---

## 🧪 Troubleshooting (quick)

* **Slash commands not visible** → Re-invite with `applications.commands` scope; give it a few minutes or test in a specific guild.
* **"Missing intents"** → Enable **Message Content Intent** in the Developer Portal.
* **Playwright errors** → `uv run playwright install chromium`; install system deps if prompted.
* **OCR errors** → Ensure `tesseract` + language packs and `pdftoppm` are installed.
* **Ollama not found** → Start Ollama locally and confirm `OLLAMA_BASE_URL`.
* **High RAM** → Set `LOW_RESOURCE_MODE=true` (reduces Discord message cache, defers heavy imports). TTS can save ~700MB with `enable_cpu_mem_arena=False`.
* **Playwright remote** → Set `PW_SERVER_URL=http://localhost:3006` for Docker-based Chromium. Version must match requirements.txt.

---

## 🤝 Contributing

* Keep functions tidy, add tests where practical (`pytest`, `pytest-asyncio`).
* Don't log secrets; keep the two logging handlers intact.
* Update `.env.example` and docs when adding features.
* In PRs, note risks, new envs, and any schema changes.

---

## 🔒 Security & Privacy

* Never commit secrets; use `.env` locally, secret stores in prod.
* Message Content Intent processes user content; ensure policy compliance.
* Restrict permissions on prompt/context files.
* JSONL logs scrub common secrets.
* Public output safety hooks sanitize all Discord send/edit boundaries against prompt-injection leaks.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
