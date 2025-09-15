# DEVNOTES — Restore Twitter/X Image Harvest → VL

Purpose: Surgical restore of Twitter/X photo harvesting into VL flow, keep video routing intact, preserve reply/thread behavior, and avoid hangs.

## What changed (surgical)
- Router `bot/router.py`:
  - Added `_extract_x_status_urls_from_text()` and `_gather_prioritized_x_urls()` to collect canonical status URLs within the active scope only.
  - Extended `_resolve_x_media()` to attempt short, bounded image harvest via syndication probe when no video formats are detected. Single retry, strict timeouts.
  - Updated early X resolve in `dispatch_message()` to use priority stack: trigger > reply-parent > thread-tail. Adds minimal structured logs and falls through cleanly on unknown/timeout.
  - Do not force-enable flags. Respects existing `X_EARLY_RESOLVE_ENABLED` and syndication probe settings.

No public interfaces, schemas, or dependencies were changed.

## Deterministic priority stack
- Trigger layer (current message)
- Reply-parent layer (REPLY_CASE only)
- Thread-tail layer (THREAD_CASE only, near reply target)
- None → default text handling

Pick the first non-empty layer. Do not merge item sets across layers. Hard fence: never leave the resolved scope.

## Routing rule
- If video found → route to video/STT path (never VL at the same time).
- Else if ≥1 photo found → route to VL (photo URLs attached via existing plumbing).
- Else → default to text.
- Video outranks photos deterministically for the same tweet.

## Timeouts and retries
- Video probe: `yt-dlp` metadata probe is already bounded.
- Image probe: per-URL syndication extraction is time-boxed (~≤4s) with a single retry.
- On timeout or extraction error: log `subsys=tw event=timeout_fallback reason=timeout` and continue. Partial results are accepted; otherwise fall back to text.

## Minimal logs to verify
- `subsys=route event=scope_resolved case=<thread|reply|lone> scope=<id>`
- `subsys=tw event=normalize_ok url=<canonical>`
- `subsys=tw event=media_found kind=<photos|video> count=<n>`
- `subsys=tw event=media_fallback kind=<og_image|none>`
- `subsys=tw event=timeout_fallback reason=<timeout|selector|other>`
- `subsys=route event=route_selected kind=<vl|video|text>`
- `subsys=route event=reply_target_ok id=<target_id>` (already emitted by scope resolver)
- `subsys=mem event=local_context count=<n> truncated=<bool>` (existing)

Tip: The router also prints concise human-readable breadcrumbs (e.g., `route.media: ...`). The structured fields appear under `extra`.

## Test checklist (manual)
Photos
1) Plain message with a twitter link to a photo tweet → photos harvested → VL route → no hang.
2) Reply + mention to a parent that has a photo tweet → parent photos harvested → VL route → reply to parent.
3) Thread where newest is text but previous tail message has a photo tweet → harvest from tail only if trigger/parent lacks media → VL route.

Video
4) Plain message with a twitter link to a video → detect video → video route (not VL).
5) Reply + mention to parent with video tweet → video route; reply to parent; no VL attempt.

Priority & isolation
6) Trigger has a new photo tweet while previous topic has unrelated links → choose trigger layer; ignore stale memory.
7) Both parent and thread tail have media but trigger does not → prefer parent layer; log that tail layer was dropped implicitly by priority.

Fallbacks & timeouts
8) Extraction timeout or DOM mismatch → `og:image` fallback attempted by probe; else text route; no hangs.
9) Partial extraction (one tweet ok, one fails) → proceed with available media; clear fallback log line.

Non-regressions
10) Minimal mention text like `@Bot yo` with no links → text route; no link-nag.
11) Twitter unroll still functions for text-only threads; image/video routing decisions above remain respected.

## Flags of interest
- `X_EARLY_RESOLVE_ENABLED` (default false)
- `X_SYNDICATION_PROBE_ENABLED` (default true)
- `X_SYNDICATION_TIMEOUT_S` (~3.0s)
- `X_SYNDICATION_MAX_IMAGES` (cap, default 4)

## How to read logs quickly
- Look for `scope_resolved` then a subsequent `route.media` line showing the `layer` chosen.
- Expect `media_found kind=video` OR `media_found kind=photos`. If neither appears, router will fall through to multimodal/text.
- Finally, `route_selected kind=<vl|video|text>` confirms which path executed.
