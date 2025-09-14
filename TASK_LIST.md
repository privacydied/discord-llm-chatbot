# Fix Reply/Mention Routing & Context Task List

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
- [ ] Implement send-time reply target calculation (compute `reply_target_id` before sending)
- [ ] Fix thread scenarios: newest message ID (or newest human if newest is ours)
- [ ] Fix reply scenarios: use `message.reference.message_id` (resolve if needed)
- [ ] Fix plain post scenarios: no `message_reference` (normal message)
- [ ] Handle placeholder/edit correctly: discard & resend if reference changed

### B) Text Default Intent
- [ ] Modify media-first guard to check for meaningful text first
- [ ] Default to text route when any non-whitespace text present (including "yo")
- [ ] Only ask for links when explicit media intent + no media found in scope

### C) Locality-First Context Scope
- [ ] Thread scope: collect tail of recent messages from current thread only
- [ ] Reply scope: build linear chain root→parent→current, take tail near trigger
- [ ] Plain scope: treat as fresh prompt without unrelated channel memory
- [ ] Hard fence: never merge memory blocks from outside current scope

### D) In-Scope Media Harvest
- [ ] Thread scope harvest: previous K messages in thread
- [ ] Reply scope harvest: parent + minimal chain near trigger
- [ ] Plain scope harvest: just current message attachments/URLs
- [ ] Apply existing token/char caps after deduplication

## Implementation
- [ ] Update dispatch_message method with corrected routing flow
- [ ] Modify maybe_build_mention_context for locality-first scope
- [ ] Update collect_thread_tail_context for scope isolation
- [ ] Fix reply-chain collectors to avoid root drift

## Testing & Validation
- [ ] Test reply + @mention + minimal text to human message
- [ ] Test reply to post with link (harvestery correctly)
- [ ] Test thread trigger at end of thread
- [ ] Test plain @mention with short text
- [ ] Test plain message with text (unchanged behavior)
- [ ] Test media-intent phrase with no links (nagging)
- [ ] Test placeholder retarget (discard & resend)
- [ ] Test timeout/archived parent fallback

## Documentation & Logging
- [ ] Add new logging events: `subsys=route event=reply_target_ok`, `text_default`, `media_intent_missing_link`
- [ ] Add scope resolution events: `subsys=mem event=scope_resolved`, `local_context`, `drop_stale`
- [ ] Update DEVNOTES section explaining scope choice, text default, log interpretation
- [ ] Ensure no extra log noise, keep concise 1-line per event format

## Acceptance Criteria
- [ ] Replies never attach to bot unless explicitly intended
- [ ] Plain @mentions with text route to text flow
- [ ] Context locality-first with hard fences
- [ ] Normal posts unchanged
- [ ] Zero regressions in existing flows
