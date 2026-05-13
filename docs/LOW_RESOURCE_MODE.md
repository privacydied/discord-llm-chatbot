# LOW_RESOURCE_MODE

Conservative defaults for deployments with limited RAM / CPU (e.g. Synology NAS).

## What it changes

When `LOW_RESOURCE_MODE=true`, the following defaults are reduced. Every value
can be overridden by its own explicit environment variable.

## Config settings affected by LOW_RESOURCE_MODE

| Setting | Normal | Low Resource |
|---------|--------|-------------|
| `CONTEXT_MAX_MESSAGES` | 10 | 5 |
| `CONTEXT_MAX_CHARS_PER_MESSAGE` | 2000 | 500 |
| `CONTEXT_MAX_TOTAL_CHARS` | 8000 | 2000 |
| `RAG_DISABLE_EAGER_LOAD` | False | True |
| `RAG_DOCUMENT_WORKERS` | 4 | 1 |
| `PERSISTENT_MEMORY_TOP_K` | 6 | 3 |
| `PERSISTENT_MEMORY_WORKERS` | 1 | 1 |
| `MEMORY_DISTILLER_INTERVAL_SECONDS` | 900 | 3600 |
| `TTS_SKIP_WARMUP` | False | True |
| `TTS_CONCURRENCY` | 4 | 1 |
| `STT_LOCAL_CONCURRENCY` | 2 | 1 |
| `VISION_MAX_CONCURRENT_JOBS` | 3 | 1 |
| `PLAYWRIGHT_MAX_CONCURRENT` | 3 | 1 |
| `HTTP_POOL_MAX_CONNECTIONS` | 50 | 10 |
| `SEARCH_POOL_MAX_CONNECTIONS` | 10 | 3 |
| `ROUTER_MAX_CONCURRENCY_LIGHT` | 8 | 4 |
| `ROUTER_MAX_CONCURRENCY_NETWORK` | 32 | 8 |
| `ROUTER_MAX_CONCURRENCY_HEAVY` | 2 | 1 |
| `DISCORD_MESSAGE_CACHE_MAX` | 256 | 64 |
| `CONFIG_WATCH_DEBOUNCE_S` | 1.0 | 2.0 |
| `LOG_MAX_STRING_LENGTH` | 1000 | 300 |
| `MULTIMODAL_MAX_ITEMS` | 5 | 2 |
| `MULTIMODAL_MAX_TOTAL_BYTES` | 50 MB | 10 MB |
| `MULTIMODAL_CONCURRENCY` | 3 | 1 |
| `IMAGE_MAX_DIMENSION` | 2048 | 1024 |
| `PDF_MAX_PAGES` | 20 | 5 |
| `VIDEO_MAX_DURATION_S` | 300 | 60 |
| `TTS_MAX_CHARS` | 4000 | 2000 |
| `TTS_SKIP_LONG_RESPONSES` | False | True |
| `VL_MAX_IMAGES` | 5 | 2 |
| `VL_MAX_IMAGE_DIMENSION` | 2048 | 1024 |
| `VISION_LOW_RESOURCE_RETRIES` | 3 | 2 |
| `SCREENSHOT_MAX_BYTES` | 5 MB | 1 MB |
| `STT_MAX_AUDIO_DURATION_S` | 300 | 60 |
| `HTTP_READ_TIMEOUT_S` | 30 | 15 |
| `URL_MAX_RESPONSE_BYTES` | 500 KiB | 200 KiB |

## Thread caps

These are always set at startup (unless already set by the environment):

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

In LOW_RESOURCE_MODE, the following are also capped:

```bash
ORT_INTRA_OP_NUM_THREADS=1
ORT_INTER_OP_NUM_THREADS=1
```

## How to enable

Set this in your `.env`:

```bash
LOW_RESOURCE_MODE=true
```

This is a single toggle. You do not need to set every individual value. Any
explicit env var you provide overrides the low-resource default.

## Minimal idle footprint (optional)

For the absolute lowest idle memory/CPU, combine `LOW_RESOURCE_MODE=true` with
disabling features you do not use:

```bash
LOW_RESOURCE_MODE=true
RAG_ENABLED=false
PERSISTENT_MEMORY_ENABLED=false
SERVER_ARCHIVE_ENABLED=false
STT_ENABLED=false
TTS_ENABLED=false
X_ENABLED=false
PROMETHEUS_ENABLED=false
```

When a feature is disabled, its background workers do not start and its heavy
imports do not happen at startup.

## Checking resource usage

Run the profiling script to get a snapshot of current resource usage:

```bash
uv run python utils/profile_resources.py
```

## Tradeoffs

- First RAG/semantic query may be slower (models loaded on demand)
- TTS warm-up skipped (first synthesis may be cold)
- Reduced concurrency means sequential processing of media items
- Memory distiller runs less frequently (fewer inferred memories)
- Smaller context window (less conversation history)
- Vision requests use smaller images and fewer retries
- HTTP connection pools are smaller (may increase latency under load)
