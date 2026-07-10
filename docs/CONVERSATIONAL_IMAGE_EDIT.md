# Conversational Image Editing

Mention or reply to the bot with an image and an edit instruction — e.g.
"@bot give this man a beard" — and it edits the image (img2img) and replies
with the result. No slash command required. `/imgedit` remains the explicit,
always-reliable path and is unchanged in behavior (aside from the bug fix
below, which makes it actually work).

## Why this exists

Previously, any addressed message with an image was routed unconditionally to
VL analysis (`Router._run_perception_notes`). An edit instruction like "give
this man a beard" would be sent to the vision-language model for *captioning*,
which can't edit pixels — the router had no img2img route outside the
`/imgedit` slash command, and no way to resolve an image from a **replied-to**
message for editing purposes.

## Routing decision

`Router._maybe_route_conversational_edit()` (`bot/router.py`) runs inside
`_process_multimodal_message_internal()`, immediately **before** the existing
reply-image → VL-perception branch, under the same gate the VL branch already
uses: `(is_dm or mentioned_me or is_reply) and combined_count >= 1 and not
has_x_url`. This guarantees:

- It only runs on messages already addressed to the bot (mention/DM/reply) —
  zero added cost for ordinary chat traffic.
- It only runs when an image is already known to be present.
- X/Twitter link handling is untouched (same exclusion as the VL branch).

Order of checks inside `_maybe_route_conversational_edit`:

1. `_conversational_edit_enabled(message)` — global `VISION_ENABLED` +
   `VISION_CONVERSATIONAL_EDIT_ENABLED` + per-guild `!feature image_editing`.
2. `classify_edit_intent(text)` — cheap, no I/O. Returns `is_edit=False`
   immediately if the text reads as a question/analysis request.
3. `resolve_edit_source_image(message, max_size_mb)` — only runs (does I/O)
   once we know this is actually an edit instruction.
4. If all three pass: submit the img2img job and reply with the result. If any
   check fails, return `None` and the caller falls through unchanged to the
   existing VL-analysis branch (or, further down, the existing text2img
   triggers if there was in fact no image).

## Intent classification (v1: keyword heuristics)

`bot/router_components/conversational_edit.py::classify_edit_intent()`:

- Analysis/question patterns (`what/who/where/when/why/how ...`, `describe`,
  `analyze`, `explain`, `is this`, trailing `?`) always win ties — matches the
  spec's "ambiguous → prefer VL analysis" safe default, even if an edit verb
  also appears in the same sentence.
- Otherwise, checks against a trigger-phrase list: `configs/vision_policy.json`
  → `intent_patterns.image_editing.trigger_phrases`, unioned with a built-in
  default list (covers the spec's example verbs — "give him/it", "make
  it/this", "add", "remove", "turn this into", "put a/some", etc.), plus any
  guild/deployment-specific additions from `VISION_EDIT_INTENT_KEYWORDS`
  (comma-separated env var).
- A future, sturdier classifier (LLM-based intent parse) can replace the body
  of `classify_edit_intent()` without changing its signature or any caller.

## Image sourcing

`resolve_edit_source_image()` resolves, in priority order:

1. An attachment or image embed on the **triggering** message
   (`collect_image_urls_from_message(message)`).
2. An attachment or image-URL on the message being **replied to**
   (`message.reference` → `channel.fetch_message()` →
   `collect_image_urls_from_message(ref_message)`) — this is the concrete bug
   from the original report: the photo was on the replied-to message, and
   nothing resolved it for editing purposes.
3. A bare image URL typed in the triggering message's text
   (`extract_urls_strict` + `is_direct_image_url`, for the case where Discord
   hasn't generated an embed yet).

Every candidate is downloaded via the existing `download_robust_image()`
(SSRF-validated, `max_size_mb` enforced both via `Content-Length` and a
streamed size guard) — no new download/validation code was written.

## Execution: same gates, same provider path as `/imgedit`

`Router._run_conversational_edit_job()` builds a
`VisionRequest(task=IMAGE_TO_IMAGE, input_image_data=<bytes>, ...)` and calls
the **existing** `VisionOrchestrator.submit_job()` — the same safety filter
(`VisionSafetyFilter.validate_request`) and budget ledger
(`VisionBudgetManager.check_budget`/`reserve_budget`) that gate `/imgedit`, no
new pool or provider integration. `_await_conversational_edit_job()` polls
`orchestrator.get_job_status()` on a bounded timeout
(`VISION_CONVERSATIONAL_EDIT_TIMEOUT_S`, default 90s; poll interval
`VISION_CONVERSATIONAL_EDIT_POLL_INTERVAL_S`, default 2s) and best-effort
cancels the job on timeout — no unbounded waits.

On success, `_finish_conversational_edit()` returns
`BotAction(content="", files=[discord.File(...), ...])`. On safety
block / budget exhaustion / provider error / timeout, it returns a short
`BotAction(error=True, content=<user_message>)` — same error surface
convention as `/imgedit` (safety/budget/provider string-mapped messages).

## Reply delivery

The router's send sink, `LLMBot._execute_action()` (`bot/core/bot.py`),
previously read `BotAction.audio_path` for attachments but **never**
`BotAction.files` — that field existed on the dataclass (and was already
counted in `BotAction.has_payload`) but nothing forwarded it into the actual
`message.reply(...)` call. This was a dead field for every prior caller (grep
confirms no code constructed `BotAction(files=...)` before this change), so
wiring it up is purely additive: `_execute_action` now merges `action.files`
into the files list it already passes to `message.reply(...)` /
`channel.send(...)`. This is what gives the conversational edit path a true
Discord **reply-with-reference** (not a bare `channel.send`), and it does so
through the existing single-send pipeline rather than a bespoke direct-send +
duplicate-message hack.

## Metrics & logging

- Prometheus counter `vision.route.conversational_edit` (labels: `outcome` ∈
  `fired`, `success`, `safety_blocked`, `budget_blocked`, `provider_error`),
  defined alongside the other `vision.route.*` counters in `bot/core/bot.py`.
- Structured log `event=edit_route`, `subsys=vision` on each routing decision
  that fires (`Router._log_edit_route_fired`).

## Configuration

| Var | Default | Purpose |
|---|---|---|
| `VISION_CONVERSATIONAL_EDIT_ENABLED` | `true` | Global kill switch |
| `VISION_CONVERSATIONAL_EDIT_TIMEOUT_S` | `90.0` | Bounded wait for the edit job |
| `VISION_CONVERSATIONAL_EDIT_POLL_INTERVAL_S` | `2.0` | Job-status poll cadence |
| `VISION_EDIT_INTENT_KEYWORDS` | `""` | Extra comma-separated edit-intent phrases |
| `!feature image_editing on\|off` | enabled | Per-guild toggle (`bot/server_features.py`) |

## Bugs found and fixed while wiring this up

Two pre-existing bugs in the reused vision pipeline would have made this
feature (and `/imgedit` itself) non-functional; both are fixed as part of this
change since the conversational route depends on them:

1. **`/imgedit` never actually edited an image.** `VisionRequest.input_image`
   (a `Path`) was the only field `/imgedit` populated, but every provider
   plugin in `bot/vision/unified_adapter.py` and `VisionSafetyFilter` only
   read `input_image_data` (raw bytes) / `input_image_url`. Every `/imgedit`
   call was therefore blocked at the safety-filter stage with
   `"missing_input_image"` before ever reaching a provider. Fixed by reading
   the downloaded attachment's bytes and setting `input_image_data` in
   `bot/commands/vision_commands.py::imgedit_command`. The new conversational
   route populates `input_image_data` directly from the start.
   See `tests/commands/test_vision_commands_imgedit.py`.

2. **Safety-filter severity escalation was silently broken.**
   `VisionSafetyFilter.validate_request()` escalated its overall safety level
   by comparing `SafetyLevel.value` *strings* (`"blocked" < "safe" <
   "warning"` alphabetically) instead of severity rank — so a sub-check
   returning `BLOCKED` (e.g. the `missing_input_image` check above, or a
   blocked prompt keyword) never actually escalated the aggregate result past
   `SAFE`, meaning `approved` could be `True` even with a blocked issue
   recorded in `detected_issues`. Fixed with an explicit `_more_severe()`
   helper ranked `SAFE < WARNING < BLOCKED`. See
   `tests/vision/test_safety_filter_severity.py`.

**Known follow-up (not fixed here, out of scope for this feature):**
`VisionSafetyFilter._load_safety_policy()` reads `policy_data.get("safety_filter",
{})` from `configs/vision_policy.json`, but the file's actual schema nests
content-filter keywords/patterns under `safety.content_filter.*` — so
`blocked_keywords`/`blocked_patterns` loaded from the policy file are always
empty (silently falls through to an empty dict, not the hardcoded default
policy, since no exception is raised). The hardcoded task-specific checks
(missing-input-image, deepfake-indicator keywords, NSFW server policy) are
unaffected and still work. Reconciling the policy file's schema with what the
loader expects is a separate, larger change and needs a decision on which
shape is canonical.
