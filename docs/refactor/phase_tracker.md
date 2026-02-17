# Phase Tracker

## Status Legend
- `PENDING`: not started
- `IN_PROGRESS`: active work
- `DONE`: merged and validated
- `BLOCKED`: waiting on decision or external dependency

## Phase Board
| Phase | Scope | Status | Validation | Notes |
|---|---|---|---|---|
| 0 | Baseline + guardrails | DONE | Medium baseline captured | commit `d4ba459` |
| 1 | Scaffolding + compat seams | IN_PROGRESS | compile + targeted tests | router + STT compat loaders added |
| 2 | Router decomposition | PENDING | pending | preserve routing/log semantics |
| 3 | STT/media boundary cleanup | PENDING | pending | preserve transcript-first + fail-open |
| 4 | Text/vision prompt composition unification | PENDING | pending | preserve concat invariants |
| 5 | Reliability hardening | PENDING | pending | dead model handling + timeout consistency |
| 6 | Contract tests + cleanup | PENDING | pending | finalize diagram + docs parity |

## Current Slice (Phase 1)
- Goal: introduce neutral scaffolding files and compatibility entry points.
- Planned changes:
  - create shared constants module for pipeline budgets/timeouts
  - create compatibility context dataclass
  - wire reads through compatibility helpers (no semantic changes)
- Exit criteria:
  - changed files compile
  - no new behavior branches introduced

## Slice Log
- 2026-02-17:
  - Added `bot/router_components/runtime.py` with `RouterRuntimeCompat`.
  - Wired `bot/router.py` init-time X syndication settings through shared loader.
  - Validation: `./.venv/bin/python -m py_compile bot/router.py bot/router_components/__init__.py bot/router_components/runtime.py`
- 2026-02-17:
  - Added `bot/stt_pipeline/runtime.py` with `STTRuntimeCompat` and `ensure_stt_manager_ready`.
  - Wired `bot/hear.py` YouTube transcript-first toggle and manager readiness through compatibility helper.
  - Maintained `_run_whisper` invocation compatibility for positional stubs.
  - Validation:
    - `./.venv/bin/python -m py_compile bot/hear.py bot/stt_pipeline/__init__.py bot/stt_pipeline/runtime.py`
    - `./.venv/bin/pytest -q tests/test_video_ingest.py` -> `8 passed`
