# Operations Guide

Quick reference for running, maintaining, and troubleshooting the bot.

## Maintenance Status

Send `!status` for basic health, or use the operator diagnostics to see
resource usage, queue depths, cache sizes, and subsystem health.

Key status lines:
- `Embedding model cache:` — present / missing / unknown
- `TTS:` engine name, loaded state, warm-up status
- `Playwright:` health result, last check time, degraded flag
- `RAG:` enabled/disabled, queue depth
- `Memory:` enabled/disabled, queue depth
- `Server archive:` enabled/disabled, queue depth

## SQLite WAL Checkpointing

The bot uses WAL mode for owned SQLite databases (memory, server archive).
Large WAL files indicate uncheckpointed writes.

To manually checkpoint:
1. Check WAL size via `!status` (shows memory DB and archive DB sizes).
2. If WAL is large, the janitor will run periodic checkpoints automatically.
3. For emergency checkpoint, restart the bot — WAL is checkpointed on clean shutdown.

Do NOT manually VACUUM live SQLite files. Use checkpoint instead.

## Cache Inspection

All caches live under `cache/` by default. Check sizes with:

```bash
du -sh cache/*/
du -sh stt/cache/
du -sh logs/
du -sh vision_data/
```

Key directories:
- `cache/stt_pcm/` — Raw PCM audio (janitor: 12h TTL)
- `cache/stt_transcripts/` — STT JSON transcripts (janitor: 24h TTL)
- `cache/video_audio/` — Video audio extracts (janitor: 3d TTL)
- `cache/tts/` — Generated TTS audio (size-capped by TTS_CACHE_MAX_MB)
- `stt/cache/` — faster-whisper model cache
- `chroma_db/` — ChromaDB vector store
- `logs/` — JSONL structured logs (rotated, 7d retention)

## Model Load Status

To check if TTS/STT/RAG models are loaded without triggering a load:

```
!status
```

Look for:
- `STT: loaded=yes/no` — faster-whisper loaded
- `TTS: loaded=yes/no` — Kokoro engine loaded
- `Embedding model cache: present/missing` — SentenceTransformer on disk

These are read-only checks that do NOT trigger lazy loading.

## Safely Clearing Caches

You can safely delete these directories (they will be recreated on next use):

```bash
rm -rf cache/stt_pcm/*
rm -rf cache/stt_transcripts/*
rm -rf cache/video_audio/*
```

Do NOT delete:
- `chroma_db/` — contains RAG and memory vector data
- `stt/cache/` — downloaded Whisper models (slow to re-download)
- `memory.db` — persistent memory store
- `server_archive.db` — server message archive

## Low-Resource Mode

For constrained environments (Synology NAS, low-memory VPS):

```
LOW_RESOURCE_MODE=true
```

See `docs/LOW_RESOURCE_MODE.md` for all settings and tradeoffs.

## Thread Caps

For CPU-constrained environments, set these before starting the bot:

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

See `docs/LOW_RESOURCE_MODE.md` for full list.

## Playwright Health

If web extraction fails, the Playwright Docker container may be down:

```bash
docker ps | grep playwright
# If Exited:
docker start playwright
# If unresponsive:
docker restart playwright
```

The bot's `!status` shows Playwright health state.

## Docker/Runtime Recommendations

- Put `chroma_db/`, SQLite DBs, and `cache/` on a local Docker volume (not NFS).
- Keep Playwright container isolated from the bot container.
- Memory-limit both containers separately.
- Set `PYTHONUNBUFFERED=1` in container env.
- Avoid debug logging in production containers.
