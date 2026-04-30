# Audit Action Plan — 2026-04-29

Source: `docs/audit-20260429-report-v1.md`

## Top Risks Addressed

| # | Risk | Severity | Status |
|---|------|----------|--------|
| R1 | `_dispatch_metadata` unbounded dict growth — `clear_dispatch_metadata()` never called | High | **Fixed** |
| R2 | Dedupe TOCTOU race in `dispatch_message()` — check-then-mark not atomic | High | **Already fixed** (per-message asyncio.Lock) |
| R3 | No timeout budget on extraction-only multimodal items — hung handler stalls entire loop | High | **Fixed** |
| S6 | `tempfile.mktemp()` in TTS (insecure) | Medium | **Verified** — already uses `mkstemp` |
| S7 | Partial secret leak — SensitiveDataFilter only matched exact keys, not substring values | Medium | **Fixed** |
| S8 | Incomplete SECRET_KEYS — `WHISPER_API_KEY`, `DDG_API_KEY` missing from redaction | Medium | **Fixed** |
| S5 | Alert command auth — no owner check | Low | **Verified** — `@commands.is_owner()` + `is_admin_user()` both present |
| TTS | Duplicate `import tempfile` inside function shadowed module-level import → `UnboundLocalError` | Medium | **Fixed** |

## What Was Changed

### bot/router.py
- Added `self.clear_dispatch_metadata()` call in finally block after processing (fixes R1 — `_dispatch_metadata` was never cleaned, causing unbounded growth)
- Wrapped extraction-only `_handle_item_with_provider()` call in `asyncio.wait_for(timeout=selected_budget)` (fixes R3 — previously no timeout guard on extraction-only items)
- Added `except asyncio.TimeoutError:` catch with logging and graceful fallback before the generic `except Exception` in extraction-only branch

### bot/utils/logging.py
- Added `WHISPER_API_KEY`, `DDG_API_KEY`, `CUSTOM_SEARCH_API_KEY`, `secret`, `password` to `SECRET_KEYS` set (fixes S8)
- Added same new keys to `redact_sensitive_values()` env key list
- `_scrub_dict_inplace()` now calls `redact_sensitive_values()` on ALL string values, not just exact-key matches — catches secret values embedded in arbitrary fields (fixes S7)

### bot/tts/interface.py
- Removed duplicate `import tempfile` inside function that shadowed module-level import, causing `UnboundLocalError` on `tempfile.mkstemp()` calls

### tests/test_router.py
- Added `TestExtractionOnlyTimeout` class (2 tests):
  - Verifies `asyncio.wait_for` is present in extraction-only code path
  - Verifies `asyncio.TimeoutError` is caught before generic `Exception`

### tests/test_security_regression.py (new file)
- 13 regression tests for security/logging claims:
  - Discord token never logged (2 tests)
  - API keys redacted in structured logs (2 tests)
  - `redact_sensitive_values` redacts known secrets (1 test)
  - Partial secret values in strings are caught (3 tests)
  - New keys (WHISPER_API_KEY, DDG_API_KEY) in SECRET_KEYS (2 tests)
  - Reasoning/thinking blocks blocked from public output (4 tests)

### tests/memory/test_persistence.py
- Added 4 concurrency/interruption tests:
  - Concurrent async writes produce valid JSON (no corruption)
  - Interrupted write recovery (corrupt file → backup/default restore)
  - Temp file cleanup after successful atomic write
  - Lock file behavior with `atomic_save_json`

### docs/AUDIT.md
- Added reference link to full audit report

### docs/audit-action-plan-20260429.md (this file)
- Factual summary of changes, risks, and follow-ups

## What Was Deliberately NOT Changed

1. **Storage engine**: No migration from JSON files to SQLite/Redis — out of scope
2. **Router rewrite**: No restructuring of `_process_multimodal_message_internal` — too risky, behavior must be preserved
3. **Parallel item processing**: X-media deduplication state creates inter-item dependencies. Kept sequential processing with timeout guard only.
4. **concurrent_processing.py integration**: Module exists but not wired in — requires careful X-URL state analysis. Deferred.
5. **Request coalescing**: URL dedup coalescer exists but not integrated — deferred.
6. **Logging volume**: No new log levels or broader logging. Only targeted timeout/failure logs.
7. **Broad fallback chains**: No new catch-all fallbacks. Existing partial-success behavior preserved.

## Remaining Follow-Up Items

1. **Integrate `concurrent_processing.py`**: Wire bounded concurrency for truly independent items (no X-URL state). Requires refactoring X-media dedup logic.
2. **Request coalescing**: Integrate URL coalescer for duplicate URLs in same/close messages.
3. **Router modularization**: Extract pure helper logic from 10K-line router. Extraction seams identified but not acted on.
4. **Coverage gaps**: TTS, STT, web extraction, RAG modules lack unit tests.
5. **Admin alert command tests**: S5 auth verified by code inspection, but no automated regression test.
6. **Full suite timeout**: Some test files are slow (~26s for test_bot.py). Consider marking slow tests.

## Verification Commands

```bash
cd /volume1/py/discord-llm-chatbot
.venv/bin/python -m pytest tests/test_router.py -q           # 24 passed
.venv/bin/python -m pytest tests/test_security_regression.py -q  # 13 passed
.venv/bin/python -m pytest tests/memory/test_persistence.py -q   # 30 passed
.venv/bin/python -m pytest tests/scripts/test_bot.py -q          # 6 passed
```

## No New Features Added

All changes are surgical hardening: memory leak fix, timeout guards, secret redaction gaps, and regression tests. No new bot features, commands, or user-facing behavior changes.
