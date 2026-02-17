# Refactor Guard (Autonomous Execution)

## Objective
Refactor architecture for maintainability and scalability **without changing user-visible behavior**.

## Non-Negotiable Invariants
1. Keep single-response contract: `1 input message -> 1 bot response`.
2. Preserve message gating semantics (DM/mention/reply rules).
3. Preserve X/Twitter routing behavior, including:
   - image tweet caption + VL facts concat
   - video tweet caption + transcript concat
4. Preserve YouTube transcript-first STT order:
   - transcript-first resolver
   - fallback to `yt-dlp -> python-ffmpeg -> whisper`.
5. Preserve `.env` compatibility and existing config keys.
6. Preserve existing slash/command names and user-facing command APIs.
7. Preserve observability breadcrumbs for key pipeline stages.

## Do-Not-Break Areas
1. `bot/core/bot.py` dispatch send/edit/reply behavior.
2. `bot/router.py` multimodal sequencing and fallback to text flow.
3. `bot/hear.py` STT fail-open behavior.
4. `bot/openai_backend.py` fallback ladder semantics.

## Required Checks Per Phase
1. Run syntax check for changed files.
2. Run phase-appropriate pytest tier (fast/medium/full).
3. Record outcomes in `docs/refactor/phase0-baseline.md` (or later phase report).
4. Update README/docs only if architecture/path changed materially.

## Autonomy Rules
1. Proceed autonomously for internal modularization and non-breaking extractions.
2. Use compatibility shims during migration; remove only after parity tests pass.
3. Stop and ask before:
   - introducing new external dependencies
   - changing persistent storage format incompatibly
   - changing user-visible response semantics/tone
