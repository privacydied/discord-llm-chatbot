# DEVNOTES — X/Twitter Routing (Video → STT, Photos → VL)

This document summarizes the surgical router changes for X/Twitter items. No public contracts were changed; no new dependencies were added.

## Detection Order (authoritative)
1. URL normalization
   - Canonicalizes x/twitter/fx/vx/mobile hosts to `https://x.com/<path>` and drops trackers.
2. Syndication/Probe (bounded)
   - A tiny media-kind step is time‑boxed (config `X_SYNDICATION_TIMEOUT_S`, default 3s) and produces exactly one of:
     - `kind=video` (player/variants present)
     - `kind=photos` (photo URLs only)
     - `kind=unknown`
3. Decision is final
   - If any step indicates video → final kind is `video`.
   - Else if photos → `photos`.
   - Else → `none` (text route).

## Route Precedence (single path)
- `video` → yt‑dlp URL → STT → summarize clip.
  - On failure or timeout, degrade to `text` (tweet text/unrolled), never VL.
- `photos` → VL (describe/analyze) using `pbs.twimg.com` URLs. No STT.
- `none` → text route.
- Mixed tweets (video + images): `video` outranks photos. VL will NOT run for poster/thumbnail frames when a playable video is detected.

## Budgets and Timeouts
- Detection step time‑boxed by `X_SYNDICATION_TIMEOUT_S` (default 3s).
- STT bounded by `X_STT_TIMEOUT_S` (default 60s). On timeout → degrade to text.
- Multimodal total budget is unchanged and still enforced with `MULTIMODAL_TOTAL_BUDGET_S` (default 240s).

## Listener Gating and Dedupe
- Skip bot/self messages (including preview/unfurl bots) before heavy work: logs `gate.skip reason=bot_or_self`.
- Per‑message dedupe: one enqueue → one dispatch. Duplicates log `gate.skip reason=duplicate`.
- Not addressed (not DM/mention/reply/thread‑in‑scope): `gate.block reason=not_addressed`.

## Logging (single‑shot markers)
Per `msg_id`, the router emits at most one of each of these markers:
- `ingest.dispatch_started`
- Pre‑gates: `gate.skip reason=bot_or_self` or `gate.block reason=not_addressed`
- `scope_resolved case=<thread|reply|lone>`
- `x.detect kind=<video|image|unknown> src=<synd_light|ytdlp_probe|both> ms=<n>`
- If kind=video:
  - `x.video.url_ok src=ytdlp ms=<n>` or `x.video.url_fail reason=<...>`
  - `stt.start dur=<mm:ss>` → `stt.ok ms=<n> chars=<n>` or `stt.fail reason=<timeout|fetch_or_decode>`
- If kind=photos:
  - `x.photos.ok count=<n> domain=pbs.twimg.com` (or `x.photos.fail`)
- Final route: `route.final kind=<video|photos|text>`
- Send‑phase (unchanged): `reply_target_ok`, `recipient_resolved`, `ping_strategy`, `send`, `dispatch:ok`

## Test Checklist (descriptive)
- Video tweet → `x.detect kind=video` → `x.video.url_ok` → `stt.ok` → `route.final kind=video` (no VL).
- Video tweet (fetch or STT timeout) → `x.video.url_fail` or `stt.fail timeout` → `route.final kind=text` (no VL).
- Photo tweet → `x.detect kind=photos` → `x.photos.ok` → VL path only → `route.final kind=photos` (no STT).
- Text‑only tweet → `x.detect kind=unknown|none` → `route.final kind=text`.
- Priority/scope unchanged: reply anchors to triggering user, thread anchors to newest human.
- Listener & dedupe: Single user post → one dispatch; embed/relay echoes are skipped.
- Fallback hygiene: Mixed tweets pick video; region/age blocked resolve quickly to text.
- Performance steady within existing budgets; no duplicate heavy routes.

## Config Keys
- `X_SYNDICATION_TIMEOUT_S` — detection step timeout (default 3.0s)
- `X_STT_TIMEOUT_S` — STT timeout (default 60.0s)
- `MULTIMODAL_TOTAL_BUDGET_S` — overall router budget (default 240.0s)

