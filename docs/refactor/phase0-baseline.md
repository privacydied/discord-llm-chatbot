# Phase 0 Baseline Snapshot

## Metadata
- Timestamp (UTC): 2026-02-17T06:58:45Z
- Baseline commit: `a75b829`
- Python: `3.12.11`
- Pytest: `8.4.1`
- Host: `synology_r1000_923+`

## Goals
1. Freeze baseline behavior and safety constraints.
2. Define test tiers for autonomous refactor execution.
3. Capture medium-tier test baseline before structural changes.

## Test Tier Commands
### Fast tier
```bash
python -m py_compile <changed_files>
```

### Medium tier
```bash
./.venv/bin/pytest -q \
  tests/core \
  tests/router \
  tests/syndication \
  tests/vision \
  tests/test_hear_ffmpeg_resolution.py \
  tests/test_hear_stream_abort.py \
  tests/test_media_ingestion.py \
  tests/test_video_ingest.py
```

### Full tier
```bash
./.venv/bin/pytest -q
```

## Medium Tier Baseline Result
- Status: `FAILED` (`32 failed, 138 passed, 3 warnings`)
- Duration: `13.15s`
- Notes:
  - Run date: `2026-02-17`
  - Command: medium tier command documented above
  - Major failure clusters:
    - `tests/core/test_router.py` (response shape/compat assumptions)
    - `tests/router/test_x_api_routing.py` (x routing expectations and test doubles)
    - `tests/syndication/test_extract_policy.py` (source tag expectation drift)
    - `tests/vision/test_pricing_calculations.py` (pricing estimator expectations)
    - `tests/test_media_ingestion.py` (patched symbols no longer present / API drift)
    - `tests/test_video_ingest.py` (`stt_manager.ensure_ready` assumption in stubs)

## Invariant Checklist
- [x] Single response contract documented
- [x] X/Twitter composition invariants documented
- [x] YouTube transcript-first order documented
- [x] Config compatibility requirement documented
- [x] Observability parity requirement documented
