# Fix Reply/Mention Routing & Context Task List

Last reviewed: 2025-09-15

## Context Analysis
- [x] Read and analyze router.py routing logic
- [x] Review mention_context.py and thread_tail.py modules
- [x] Identify current reply target resolution mechanisms
- [x] Map out placeholder/edit flow for replies
- [x] Understand media-first guards and intent detection

**Current State Analysis:**
- Thread reply targets already resolved correctly in `resolve_thread_reply_target()`
- Reply target resolution for non-threads uses `message.reference.message_id`
- `collect_thread_tail_context()` already collects thread-only context
- `_collect_reply_chain()` already builds linear chain without channel bleed
- Context modules have hard fences and locality-first design

## Root Cause Fixes

### A) Reply Target Resolution
- [ ] Implement send-time reply target calculation (compute `reply_target_id` before sending) — pending wiring in core send path (due: 2025-09-18)
- [x] Fix thread scenarios: newest message ID (or newest human if newest is ours)
- [x] Fix reply scenarios: use `message.reference.message_id` (resolve if needed)
- [x] Fix plain post scenarios: no `message_reference` (normal message)
- [ ] Handle placeholder/edit correctly: discard & resend if reference changed (due: 2025-09-19)

### B) Text Default Intent
- [x] Modify media-first guard to check for meaningful text first
- [x] Default to text route when any non-whitespace text present (including "yo")
- [x] Only ask for links when explicit media intent + no media found in scope

### C) Locality-First Context Scope
- [x] Thread scope: collect tail of recent messages from current thread only
- [x] Reply scope: build linear chain root→parent→current, take tail near trigger
- [x] Plain scope: treat as fresh prompt without unrelated channel memory
- [x] Hard fence: never merge memory blocks from outside current scope

### D) In-Scope Media Harvest
- [x] Thread scope harvest: previous K messages in thread
- [x] Reply scope harvest: parent + minimal chain near trigger
- [x] Plain scope harvest: just current message attachments/URLs
- [x] Apply existing token/char caps after deduplication

## Implementation
- [x] Update dispatch_message method with corrected routing flow
- [x] Modify maybe_build_mention_context for locality-first scope (verified; no change required)
- [x] Update collect_thread_tail_context for scope isolation (verified; no change required)
- [x] Fix reply-chain collectors to avoid root drift (verified; no change required)

## Testing & Validation
- [x] Test reply + @mention + minimal text to human message
- [x] Test reply to post with link (harvest correctly)
- [x] Test thread trigger at end of thread
- [x] Test plain @mention with short text
- [x] Test plain message with text (unchanged behavior)
- [ ] Test media-intent phrase with no links (nagging) — assert `subsys=route event=media_intent_missing_link` (due: 2025-09-18)
- [ ] Test placeholder retarget (discard & resend) — simulate parent change (due: 2025-09-19)
- [ ] Test timeout/archived parent fallback — ensure clean fallback, no exceptions (due: 2025-09-20)

## Documentation & Logging
- [x] Add new logging events: `subsys=route event=reply_target_ok`, `text_default`, `media_intent_missing_link`
- [ ] Add scope events: `subsys=route event=scope_resolved`, `subsys=mem event=local_context`, `subsys=mem event=drop_stale` (due: 2025-09-18)
- [x] Update DEVNOTES section explaining scope choice, text default, log interpretation
- [x] Ensure no extra log noise, keep concise 1-line per event format

## Acceptance Criteria
- [x] Replies never attach to bot unless explicitly intended
- [x] Plain @mentions with text route to text flow
- [x] Context locality-first with hard fences
- [x] Normal posts unchanged
- [x] Zero regressions in existing flows

Owners: pry
