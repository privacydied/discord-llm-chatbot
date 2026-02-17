# Autonomous Refactor Plan

## Purpose
Refactor core bot architecture for maintainability and reliability while preserving user-visible behavior and routing semantics.

## Baseline Reality (Frozen)
- Baseline commit: `a75b829`
- Guardrails: `docs/refactor/refactor_guard.md`
- Medium baseline snapshot: `docs/refactor/phase0-baseline.md` (`32 failed, 138 passed`)
- Constraint: existing medium suite is partially stale, so progress must be measured by:
  - no new regressions in targeted contracts
  - syntax/type integrity on changed files
  - runtime smoke checks against known user-critical flows

## Refactor Strategy
Use a strangler-pattern refactor:
1. Add thin, testable modules beside current monoliths.
2. Route existing code through compatibility adapters.
3. Preserve call signatures and log breadcrumbs.
4. Move logic in slices with parity checks after each slice.

## Phase Plan

## Phase 1: Execution Scaffolding (No Behavior Change)
Deliverables:
- `docs/refactor/phase_tracker.md` with phase checklist and status.
- Shared compatibility context object for router/hear orchestration.
- Central constants for pipeline budgets/timeouts (read-through to existing env keys).

Acceptance:
- No routing behavior change.
- Existing env keys still work.
- `python -m py_compile` passes for touched files.

## Phase 2: Router Decomposition (Structure, Not Semantics)
Deliverables:
- New package: `bot/router_components/`
- Extract modules:
  - `gating.py` (DM/mention/reply checks)
  - `input_harvest.py` (URL/attachment collection)
  - `x_routing.py` (x/syndication decision helpers)
  - `compose.py` (prompt composition helpers)
- Keep `bot/router.py` as orchestrator + compatibility wrappers.

Acceptance:
- Existing log markers preserved (`route.guard`, `mm.items.after_harvest`, etc.).
- User-visible output unchanged for golden inputs.

## Phase 3: Media/STT Boundary Cleanup
Deliverables:
- Add `bot/stt_pipeline/` helpers:
  - transcript-first resolver bridge
  - ffmpeg preparation wrapper
  - whisper invocation wrapper
- Keep current fail-open fallback order unchanged.
- Normalize STT result schema in one place.

Acceptance:
- X video and YouTube transcript-first behavior preserved.
- Caption/transcript concat invariants preserved.

## Phase 4: Text/Vision Integration Boundary
Deliverables:
- Extract reusable prompt assembly service used by:
  - text-only flow
  - x image+caption flow
  - x video+transcript flow
  - x article text+image flow
- Remove duplicated injection logic in router.

Acceptance:
- One canonical composition path.
- No loss of tweet/article text in multimodal prompts.

## Phase 5: Reliability Hardening
Deliverables:
- Provider ladder hygiene:
  - ignore hard-dead free models without noisy retries
  - preserve circuit breaker behavior for transient failures
- Timeout budget normalization between media and vision jobs.
- Add explicit error taxonomy for user-facing fallback messages.

Acceptance:
- Reduced timeout-induced false failures.
- Cleaner fallback logs with actionable reasons.

## Phase 6: Contract Tests and Cleanup
Deliverables:
- Add contract tests focused on user-critical invariants:
  - x video -> STT -> text flow
  - x images -> VL + tweet text concat
  - x article -> article text + image perception concat
  - YouTube -> transcript-first -> text flow
- Remove dead shims and duplicate paths after parity confidence.

Acceptance:
- Contract suite green.
- README architecture and mermaid diagram aligned to actual runtime paths.

## Work Rules for Autonomous Execution
1. Keep commits small and phase-scoped.
2. Every commit includes:
   - what moved
   - why behavior is unchanged
   - how verified
3. Stop and escalate before:
   - adding new dependency
   - changing persistent data format
   - changing user-facing reply semantics

## Verification Ladder per Change
1. `python -m py_compile <changed_files>`
2. Targeted pytest for touched area
3. One runtime smoke scenario from logs/user flows
4. Update `docs/refactor/phase_tracker.md`

## Rollback Plan
- Keep compatibility wrappers until contract tests pass.
- If a slice regresses behavior:
  - revert the slice commit only
  - keep scaffolding/docs
  - re-attempt with smaller extraction unit
