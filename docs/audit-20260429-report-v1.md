# PRODUCT AUDIT — Discord LLM Chatbot

## Date: 2026-04-29
## Version: v1

---

## 1. Executive Summary

This is a sophisticated Discord bot (`discord-llm-chatbot` v0.2.0) that integrates large language models (LLMs) with multimodal capabilities including text chat, voice synthesis (TTS), speech recognition (STT), image analysis (VL), web content extraction, and Retrieval-Augmented Generation (RAG). The bot connects to Discord and processes messages through a central router that intelligently routes different input modalities to appropriate handlers.

**Key Value Propositions:**
- Unified multimodal AI interface within Discord
- Support for text, image, video, audio, PDF, and web content processing
- Multiple AI backends including OpenRouter, NVIDIA NIM, and Ollama
- Persistent user/server profiles with memory
- RAG-powered knowledge base search
- Voice channel TTS capabilities
- Extensible command system with cog-based architecture

**Target Audience:** Discord server administrators and users seeking AI-powered assistance with support for rich media content.

---

## 2. Architecture Overview

### 2.1 Deployment Model
- Self-hosted Python application
- Requires Discord Bot Token
- Optional remote Playwright server for web scraping (port 3006)
- Prometheus metrics endpoint (port 8000)

### 2.2 Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Discord Framework | discord.py 2.6.3 |
| HTTP Clients | aiohttp, httpx |
| LLM APIs | OpenAI/OpenRouter, NVIDIA NIM, Ollama |
| RAG | ChromaDB + sentence-transformers |
| TTS | Kokoro ONNX |
| STT | faster-whisper |
| Web Scraping | Playwright, trafilatura, BeautifulSoup |
| Vision | Vision-Language models via OpenRouter |
| Image Gen | Together.ai, Novita.ai |
| Logging | Rich + JSONL dual sink |

### 2.3 Directory Structure

```
.
├── bot/                          # Main application code
│   ├── commands/                 # Discord command cogs
│   ├── core/                     # Bot core (startup, config, bot class)
│   ├── events/                   # Event handlers
│   ├── memory/                   # User/server profile management
│   ├── metrics/                  # Prometheus metrics
│   ├── rag/                      # RAG system (ChromaDB integration)
│   ├── router_components/        # Router helper modules
│   ├── search/                   # Web search providers
│   ├── stt_module/               # STT failure handling
│   ├── stt_pipeline/             # STT orchestration
│   ├── syndication/              # X/Twitter syndication
│   ├── threads/                  # X thread unroll
│   ├── tts/                      # TTS system
│   ├── types/                    # Type definitions
│   ├── utils/                    # Utility functions
│   ├── vision/                   # Image/video generation
│   ├── vl/                       # Vision-language postprocessing
│   ├── voice/                    # Voice message publishing
│   └── router.py                 # Central message router (466KB+, 10K+ lines)
├── config/                       # Configuration files
├── docs/                         # Documentation
├── prompts/                      # System prompts
├── tests/                        # Pytest test suite (200+ test files)
└── utils/                        # One-off scripts
```

---

## 3. Startup & Bootstrap

### 3.1 Entry Point
**File:** `bot/main.py`

```python
# Main entry: main() -> LLMBot instance -> bot.start()
```

**Startup Sequence:**
1. Initialize logging (Rich + JSONL dual sink)
2. Parse CLI arguments (--debug, --config-check, --version)
3. Load configuration from `.env` via `bot/config.py`
4. Run pre-flight checks (`bot/core/startup.py`):
   - Validate Discord token
   - Verify Discord intents (message_content, guilds, messages, voice_states)
   - Check Playwright browsers (local or remote server)
5. Create bot intents and instantiate `LLMBot`
6. Set up signal handlers (`bot/shutdown.py`)
7. Spawn background tasks (`bot/tasks.py`)
8. Start config file watcher (`bot/config_reload.py`)
9. Connect to Discord with retry logic (3 attempts, exponential backoff)

### 3.2 Pre-Flight Checks
**File:** `bot/core/startup.py`

- **Token Validation:** SHA256 hash logged (never logs full token)
- **Intent Verification:** Ensures message_content, guilds, members, voice_states enabled
- **Playwright Check:** Validates remote server at `PW_SERVER_URL` or local browser binaries
- **Version Alignment:** Ensures Playwright client version matches server

### 3.3 Bot Class Initialization
**File:** `bot/core/bot.py`  
**Class:** `LLMBot(commands.Bot)`

Key attributes initialized:
- `config`: Loaded from environment
- `context_manager`: Basic context persistence
- `enhanced_context_manager`: Multi-user conversation tracking
- `user_profiles`: User memory/profiles cache
- `server_profiles`: Server-specific settings
- `tts_manager`: Text-to-speech engine
- `router`: Central message router
- `metrics`: Prometheus metrics collector

---

## 4. Complete Route Map

### 4.1 Discord Events

| Event | Handler | Description |
|-------|---------|-------------|
| `on_ready` | `bot/core/bot.py` | Bot startup confirmation |
| `on_message` | `bot/core/bot.py` | Primary message router entry |
| `on_command_error` | `bot/events/command_error_handler.py` | Command error handling |
| `on_voice_state_update` | `bot/core/bot.py` | Voice channel monitoring |

### 4.2 Command Router
**File:** `bot/router.py`
**Class:** `Router`

**Input Modality Detection:**
- `TEXT_ONLY`: Plain text messages
- `SINGLE_IMAGE` / `MULTI_IMAGE`: Image attachments/URLs
- `VIDEO_URL`: YouTube, TikTok, Twitter/X video URLs
- `AUDIO_VIDEO_FILE`: Audio/video attachments
- `PDF_DOCUMENT` / `PDF_OCR`: PDF files
- `GENERAL_URL`: Generic web URLs
- `SCREENSHOT_URL`: URLs requiring visual extraction

**Processing Flow:**
1. `dispatch_message()` → Entry point
2. `parse_command()` → Check for explicit `!` commands
3. `collect_input_items()` → Gather all inputs (text, URLs, attachments, embeds)
4. `_process_multimodal_message_internal()` → Sequential item processing
5. For each item:
   - Image: `_handle_image()` → `see_infer()` (VL model)
   - Video URL: `_handle_video_url()` → `hear_infer()` (STT/transcription)
   - PDF: `_handle_pdf()` → PDF text extraction or OCR
   - URL: `_handle_url()` → Web extraction
   - Text: `_flow_process_text()` → LLM response
6. Result aggregation and response formatting

### 4.3 Registered Commands
**File:** `bot/commands/__init__.py`

| Command | Cog | Description |
|---------|-----|-------------|
| `!ping` | - | Simple connectivity test |
| `!chat` | - | Explicit chat command |
| `!search` | `search_commands.py` | Web search (DuckDuckGo) |
| `!tts` | `tts_cmds.py` | Toggle TTS for user |
| `!ttsall` | `tts_cmds.py` | Toggle TTS for all users |
| `!speak` | `tts_cmds.py` | Single TTS response |
| `!say` | `tts_cmds.py` | Say message via TTS |
| `!memory-*` | `memory_cmds.py` | User memory management |
| `!rag` | `rag_commands.py` | RAG search/system commands |
| `!alert` | `admin_alert_commands.py` | Admin DM alerts |
| `!img` | `img_commands.py` | Image generation |
| `!vision` | `vision_commands.py` | Vision command variants |

---

## 5. Platform/Service Inventory

### 5.1 AI Backends
**File:** `bot/ai_backend.py`, `bot/openai_backend.py`, `bot/nvidia_backend.py`, `bot/ollama.py`

| Backend | Config | Purpose |
|---------|--------|---------|
| OpenAI/OpenRouter | `TEXT_BACKEND=openai` | Primary LLM text generation |
| NVIDIA NIM | `TEXT_BACKEND=nvidia` | Alternative LLM inference |
| Ollama | `TEXT_BACKEND=ollama` | Local LLM hosting |

**Vision-Language (VL):** Configured separately via `VL_MODEL` env var  
**Vision Generation:** Together.ai, Novita.ai (Image/Video generation)

### 5.2 Discord Integration
- **Framework:** discord.py 2.6.3
- **Intents Required:** message_content, guilds, members, voice_states
- **Features:** DMs, Guild messages, Voice channels, Thread replies

### 5.3 Speech-to-Text
**File:** `bot/stt.py`

- **Engine:** faster-whisper
- **Models:** Configurable (base, small, medium, large-v3)
- **Features:** Async lazy loading, model downgrade on failure, CPU-optimized threading

### 5.4 Text-to-Speech
**File:** `bot/tts/interface.py`, `bot/tts/manager.py`

- **Engine:** Kokoro ONNX
- **Features:** Sentence-aware chunking, prosodic pause insertion, audio postprocessing (fade, padding)
- **Voice Selection:** `TTS_VOICE` env var (default: am_michael)

### 5.5 RAG System
**File:** `bot/rag/hybrid_search.py`, `bot/rag/chroma_backend.py`

- **Vector DB:** ChromaDB
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Search:** Hybrid (vector + keyword) with configurable weights
- **Chunking:** Configurable size/overlap with overlap preservation

### 5.6 Web Extraction
**File:** `bot/web_extraction_service.py`

**Two-Tier System:**
- **Tier A:** Fast HTTPX fetching (6s timeout)
- **Tier B:** Playwright browser automation (12s timeout, optional)

### 5.7 X/Twitter Integration
**Files:** `bot/x_api_client.py`, `bot/syndication/extract.py`, `bot/threads/x_thread_unroll.py`

- **API v2:** Official API with bearer token
- **Syndication:** Fallback to syndication endpoints
- **Thread Unroll:** Author self-reply chain extraction

### 5.8 Web Search
**File:** `bot/search.py`

- **Provider:** DuckDuckGo (HTML scraping)
- **Features:** Safe search levels (off/moderate/strict), locale support
- **Cache:** 30-minute result caching

---

## 6. Shared Utilities

### 6.1 Logging System
**File:** `bot/utils/logging.py`

**Dual Sink Strategy:**
- **Pretty Console:** RichHandler with color codes, icons, tracebacks
- **Structured File:** JSONL rotating logs (10MB × 5 files)

**Format Fields:**
```json
{"ts": "ISO8601", "level": "INFO", "name": "module", "subsys": "component", "event": "action", "detail": {...}}
```

### 6.2 Retry Utilities
**File:** `bot/retry_utils.py`, `bot/enhanced_retry.py`

- Exponential backoff with jitter
- Retryable error classification
- Circuit breaker pattern support

### 6.3 HTTP Client
**File:** `bot/http_client.py`

- Async HTTPX with connection pooling
- Request configuration (timeouts, headers)
- HTTTP/2 support

### 6.4 File Utilities
**File:** `bot/utils/file_utils.py`

- Async file download with timeout
- MIME type detection
- Stream handling for large files

---

## 7. Database Schema

### 7.1 User Profiles
**File:** `user_profiles/{user_id}.json`

```json
{
  "discord_id": "string",
  "user_id": "string",
  "username": "string",
  "memories": ["list"],
  "history": [{"role": "user", "content": "string"}],
  "preferences": {},
  "context_notes": "string",
  "total_messages": 0,
  "last_updated": "ISO8601",
  "created_at": "ISO8601",
  "first_seen": "ISO8601",
  "is_bot": false,
  "tone": "neutral",
  "last_seen": null,
  "custom_data": {}
}
```

### 7.2 Context Storage
**File:** `context.json`

Per-channel conversation history with configurable max messages.

### 7.3 RAG ChromaDB
**Path:** `./chroma_db/`

- Vector collections for knowledge base
- Metadata: source, chunk_index, user_id, guild_id
- Embeddings: 384-dimensional (all-MiniLM-L6-v2)

---

## 8. Frontend Architecture

### 8.1 Discord UI Integration
- **Message Sending:** `channel.send()` with typing indicators
- **Embeds:** Discord embeds for search results, RAG results
- **Voice:** Voice channel connections for TTS playback

### 8.2 Response Formatting
**File:** `bot/action.py`

```python
@dataclass
class BotAction:
    content: str              # Text response
    embeds: List[Embed]       # Rich embeds
    files: List[File]         # Attachments
    audio_path: Optional[str] # TTS audio file
    error: bool               # Error flag
    meta: Dict[str, Any]      # Metadata
```

### 8.3 Streaming Status Cards
Feature for showing generation progress for media-heavy operations.
- Enabled via `STREAMING_ENABLE` env var
- Embeds show step-by-step progress

---

## 9. Background Jobs & Scheduled Tasks

### 9.1 RAG Background Indexing
**File:** `bot/rag/indexing_queue.py`
- Asynchronous document processing
- Configurable worker count
- Queue-based backpressure

### 9.2 Memory Auto-Save
**File:** `bot/tasks.py`
- Periodic user profile persistence
- Configurable interval (default: 30s)

### 9.3 File Cleanup
**File:** `bot/janitor.py`
- Old temp file removal
- TTS cache cleanup
- Stale artifact pruning

---

## 10. Middleware & Cross-Cutting Concerns

### 10.1 Circuit Breakers
HTTP client includes circuit breaker pattern for external APIs:
- Failure window tracking
- Half-open state with probabilistic recovery

### 10.2 Rate Limiting
- Per-user message queue processing
- Per-provider API rate tracking

### 10.3 Input Validation
- Attachment size limits
- MIME type validation
- URL format verification

### 10.4 Content Sanitization
**File:** `bot/public_output.py`, `bot/vl/postprocess.py`
- Chain-of-thought leakage removal
- Reasoning content stripping
- Token scrubbing from logs

---

## 11. Type System & Contracts

### 11.1 Core Types
**File:** `bot/types.py`

```python
class Command(Enum):
    CHAT, PING, HELP, SEARCH, TTS, TTS_ALL, SPEAK, SAY,
    MEMORY_ADD, MEMORY_DEL, MEMORY_SHOW, MEMORY_WIPE,
    RAG, RAG_BOOTSTRAP, RAG_SEARCH, RAG_STATUS,
    ALERT, IMG, IGNORE

class InputModality(Enum):
    TEXT_ONLY, SINGLE_IMAGE, MULTI_IMAGE, VIDEO_URL,
    AUDIO_VIDEO_FILE, PDF_DOCUMENT, PDF_OCR,
    GENERAL_URL, SCREENSHOT_URL, UNKNOWN

@dataclass
class BotAction:
    content: str
    embeds: List[Embed]
    files: List[File]
    audio_path: Optional[str]
    error: bool
    meta: Dict[str, Any]
```

### 11.2 API Contracts
- OpenAI-compatible API via OpenRouter
- Discord.py Message/Interaction objects
- ChromaDB embedding interface

---

## 12. Configuration & Environment

### 12.1 Critical Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_TOKEN` | Yes | Discord bot token |
| `PROMPT_FILE` | Yes | Path to system prompt file |
| `VL_PROMPT_FILE` | Yes | Path to vision prompt file |
| `OPENAI_API_KEY` | Yes* | OpenAI/OpenRouter API key |
| `OPENAI_API_BASE` | No | API base URL (default: OpenRouter) |
| `OPENAI_TEXT_MODEL` | No | Default text model |
| `VL_MODEL` | No | Vision-language model |

### 12.2 Optional Features

| Variable | Default | Feature |
|----------|---------|---------|
| `ENABLE_RAG` | true | RAG system |
| `TTS_ENGINE` | kokoro-onnx | TTS provider |
| `STT_ENABLE` | true | Speech-to-text |
| `VISION_ENABLED` | false | Image generation |
| `X_API_ENABLED` | false | X/Twitter API |
| `PW_SERVER_URL` | - | Remote Playwright |

### 12.3 Full Configuration
See `.env.example` (370 lines) for complete configuration options.

---

## 13. Security Posture

### 13.1 Authentication
- Discord token stored in environment (never logged)
- API keys redacted from all logs
- Bearer token presence-only logging

### 13.2 Input Validation
- Command name sanitization (alphanumeric + underscore + dash)
- URL format validation
- File size/type restrictions
- PDF sanitization before processing

### 13.3 Privacy
- No PII in structured logs
- User ID hashing available
- Configurable context persistence (memory-only mode)

### 13.4 Secrets Handling
- `.env` file for local development
- Environment variable validation on startup
- Token hash-only logging

---

## 14. Known Patterns & Conventions

### 14.1 Error Handling (REH)
- Typed exceptions per boundary
- Graceful degradation (fails to text-only if media fails)
- User-friendly error messages
- Structured error logging with context

### 14.2 Code Organization (CA)
- Layered architecture (domain/usecases/adapters/framework)
- No blocking calls in async paths
- Dependency injection for testability

### 14.3 Code Quality (CSD)
- Target: Functions ≤30 lines, nesting ≤3
- Type hints throughout
- Comprehensive test coverage

### 14.4 RAT Tags
Used in commits/PRs to indicate rule compliance:
- `[CA]` - Clean Architecture
- `[REH]` - Robust Error Handling
- `[SFT]` - Security-First Thinking
- `[PA]` - Performance Awareness
- `[RM]` - Resource Management

---

## 15. Dependency Map

### 15.1 Core Dependencies
```
discord.py==2.6.3        # Discord framework
openai==1.107.0         # LLM API client
aiohttp==3.12.15        # Async HTTP
httpx[http2]==0.28.1    # HTTP/2 client
chromadb==1.0.20        # Vector database
sentence-transformers   # Embeddings
faster-whisper==1.2.0   # STT
kokoro-onnx>=0.4.9      # TTS
playwright==1.58.0      # Browser automation
beautifulsoup4==4.13.5  # HTML parsing
trafilatura==2.0.0      # Article extraction
```

### 15.2 Infrastructure
- `rich==14.1.0` - Terminal UI/logging
- `numpy==2.3.2` - Numerical computing
- `torch==2.3.1` - ML inference
- `prometheus-client==0.22.1` - Metrics

---

## 16. Edge Cases & Operational Notes

### 16.1 Synology NAS Specific
- Playwright requires Docker container (port 3006)
- SSL CA bundle issues patched in OpenAI backend
- Chromium system libraries missing - use remote Playwright

### 16.2 Resource Constraints
- STT model on CPU designed for 2 threads max (torch.set_num_threads)
- TTS caching limited to 100 items (configurable)
- Playwright timeout after ~3h - requires restart

### 16.3 Known Limitations
- DuckDuckGo search may require rate limiting
- X API requires bearer token for full functionality
- Some LLM providers may be rate-limited on free tiers
- 1900 char Discord message limit enforced

### 16.4 Fallback Chains
- STT: Primary → Model downgrade → Caption-only
- VL: Model ladder with fallback timeouts
- Web extraction: HTTPX → Playwright → Screenshot
- X content: API → Syndication → Web scraping

---

## Appendix: File Count Summary

| Category | Count |
|----------|-------|
| Python source files | ~200 |
| Test files | ~200+ |
| Command cogs | 16 |
| Router modules | 12 |
| Total lines of code | ~88,000+ |

**Largest Modules:**
- `bot/router.py`: 466KB (10,574 lines)
- `bot/core/bot.py`: 130KB (3,003 lines)
- `bot/tts/interface.py`: 69KB (1,718 lines)

---

*Report generated from source code analysis. No stale documentation was consulted.*
