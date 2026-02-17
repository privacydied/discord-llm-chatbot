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
| 1 | Scaffolding + compat seams | DONE | compile + targeted tests | router + STT compat loaders added |
| 2 | Router decomposition | IN_PROGRESS | compile + targeted tests | composition helpers extracted |
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
- 2026-02-17:
  - Added `bot/router_components/compose.py` for extracted pure composition helpers.
  - Routed `Router._format_x_tweet_with_transcription` and `Router._compose_x_tweet_with_visual_facts` through extracted helpers.
  - Added focused tests: `tests/router/test_router_components_compose.py`.
  - Validation:
    - `./.venv/bin/python -m py_compile bot/router.py bot/router_components/__init__.py bot/router_components/runtime.py bot/router_components/compose.py`
    - `./.venv/bin/pytest -q tests/router/test_router_components_compose.py tests/test_video_ingest.py` -> `11 passed`
- 2026-02-17:
  - Added `bot/router_components/gating.py` and `bot/router_components/input_harvest.py`.
  - Routed router mention/reply checks through extracted gating helpers.
  - Replaced duplicated text-attachment and URL cleanup helpers with shared input-harvest helpers.
  - Added focused tests:
    - `tests/router/test_router_components_gating.py`
    - `tests/router/test_router_components_input_harvest.py`
  - Validation:
    - `./.venv/bin/python -m py_compile bot/router.py bot/router_components/__init__.py bot/router_components/gating.py bot/router_components/input_harvest.py`
    - `./.venv/bin/pytest -q tests/router/test_router_components_compose.py tests/router/test_router_components_gating.py tests/router/test_router_components_input_harvest.py tests/test_video_ingest.py` -> `19 passed`
- 2026-02-17:
  - Added `bot/router_components/x_routing.py` for pure X/Twitter URL/media helper extraction.
  - Routed router wrapper methods through extracted X helper module:
    - URL detection/status-id parse
    - candidate URL collection
    - twitter thumbnail/media-cdn/media-path checks
  - Added focused tests: `tests/router/test_router_components_x_routing.py`.
  - Validation:
    - `./.venv/bin/python -m py_compile bot/router.py bot/router_components/__init__.py bot/router_components/x_routing.py`
    - `./.venv/bin/pytest -q tests/router/test_router_components_compose.py tests/router/test_router_components_gating.py tests/router/test_router_components_input_harvest.py tests/router/test_router_components_x_routing.py tests/test_video_ingest.py` -> `22 passed`
- 2026-02-17:
  - Added `bot/stt_pipeline/youtube_path.py` and extracted transcript-first result payload assembly from `bot/hear.py`.
  - Router/STT behavior unchanged; extraction is a pure payload-builder seam for Phase 3.
  - Added focused tests: `tests/test_stt_pipeline_youtube_path.py`.
  - Validation:
    - `./.venv/bin/python -m py_compile bot/hear.py bot/stt_pipeline/__init__.py bot/stt_pipeline/youtube_path.py`
    - `./.venv/bin/pytest -q tests/router/test_router_components_compose.py tests/router/test_router_components_gating.py tests/router/test_router_components_input_harvest.py tests/router/test_router_components_x_routing.py tests/test_video_ingest.py tests/test_stt_pipeline_youtube_path.py` -> `24 passed`
- 2026-02-17:
  - Added `bot/stt_pipeline/ffmpeg_runtime.py` and extracted ffmpeg candidate/AAC-probe helpers from `bot/hear.py`.
  - Kept `bot/hear.py` wrappers to preserve runtime behavior and logging semantics.
  - Added focused tests: `tests/test_stt_pipeline_ffmpeg_runtime.py`.
  - Validation:
    - `./.venv/bin/python -m py_compile bot/hear.py bot/stt_pipeline/__init__.py bot/stt_pipeline/ffmpeg_runtime.py`
    - `./.venv/bin/pytest -q tests/test_stt_pipeline_ffmpeg_runtime.py tests/test_stt_pipeline_youtube_path.py tests/test_video_ingest.py tests/router/test_router_components_compose.py tests/router/test_router_components_gating.py tests/router/test_router_components_input_harvest.py tests/router/test_router_components_x_routing.py` -> `27 passed`
