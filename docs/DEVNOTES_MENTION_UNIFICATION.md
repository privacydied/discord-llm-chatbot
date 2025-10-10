# DEVNOTES — Mention Handling Unification (Text‑First, Scope‑First)

Change set touches only router behavior and logging. No new dependencies, schemas, or storage. Code changes live in:
- [bot/router.py](bot/router.py)

## Deterministic Decision Order

1) Resolve scope first (no guesses)
- THREAD: scope = thread id; reply target = newest at send-time (or newest human if newest is ours)
  - Context: previous K from this thread only (bounded + deduped)
- REPLY: scope = parent id; reply target = referenced parent
  - Context: linear chain root → … → parent → current, then tail near the trigger (bounded + deduped)
- PLAIN: scope = this message; reply target = none
  - Context: just this message (or your tiny local window if supported)
- Hard fence: never merge memory from outside the chosen scope

2) Extract chat text robustly (after mention cleaning)
- Strip leading mention token “<@id>” without swallowing adjacent content; preserve punctuation and emoji
- Normalize whitespace; keep “yo”, “?”, “👍”, “ok”, “hm”, etc.
- “Has text” is true if any of:
  - At least one non‑whitespace character (letters, digits, symbols, emoji)
  - A single punctuation token like ?, !, … (including “??”, “!!”)
  - A short single token (<= 3 chars) like “yo”, “ok”, “hm”

3) Text‑first default for any @mention that has chat text
- If cleaned, in‑scope content has any chat signal above → route to TEXT
- Do not require a “question” shape; tiny chat is valid

4) Media intent and nag
- Show the “send a link/media” nag only when BOTH:
  - The user’s words explicitly ask for media analysis (“summarize this video/image/link…”, “what’s in this pic…”, “analyze this thread…”), AND
  - No URL/media is present in the chosen scope (trigger + parent in reply; thread tail in thread)
- If URLs/media exist in scope → route to the corresponding media flow
- If intent is ambiguous → prefer TEXT

5) Inputs come only from the chosen scope
- Harvest URLs/attachments exclusively from:
  - REPLY: parent (plus immediate trigger), bounded/unique
  - THREAD: thread tail only, bounded/unique
  - PLAIN: the message itself
- Do not reach into older channel history; apply existing budgets and mark truncation

6) Send with the final reply target
- Guild: always reply directly to the user message that triggered the mention or follow-up reply (no channel-wide fallbacks)
- DM: reply to the incoming DM message (mention remains optional)
- THREAD/REPLY: use the resolved target; if a placeholder anchored incorrectly is emitted upstream, prefer delete & resend over editing into the wrong anchor

## Minimal Logging (one concise line per step)
- subsys=route event=scope_resolved case=<thread|reply|lone> scope=<id> reply_target=<id|none>
- subsys=route event=reply_target_ok target=<id|none>
- subsys=route event=dispatch.reply_target context=<guild|dm> channel_id=<id> thread_id=<id|none> trigger_message_id=<id>
- subsys=route event=text_default reason=<mention_has_text|ambiguous_intent>
- subsys=route event=media_intent_missing_link
- subsys=mem event=local_context count=<n> truncated=<bool>
- subsys=mem event=drop_stale reason=scope_mismatch dropped=<n>

Notes:
- No stack traces at info level
- Keep all keys tiny and stable

## Implementation Notes (what changed)
- Scope resolution centralized before intent/routing (thread/reply/plain)
- Relaxed “has text” detector accepts punctuation/emoji/short tokens to avoid false “no text”
- Explicit media‑intent detector (phrase list) for the nag path when scope has no media/URL
- Text‑first default applied when no media items are harvested
- Added reply_target_ok breadcrumb when a final target is chosen
- Added dispatch.reply_target breadcrumb with context + channel/thread IDs
- Reply target metadata (`context`, `mention_detected`, `reply_target_ok`) bundled into dispatch extras for downstream logging

## Edge Cases Addressed
- @Bot yo, @Bot ?, @Bot 👍, hey @Bot → TEXT
- @Bot summarize this video (no URL) → media_intent_missing_link nag
- @Bot summarize this video https://… → media route (no nag)
- Reply + @Bot yo → reply to parent; use reply‑tail context near the trigger
- Thread mention → reply to newest (or newest human if newest is ours), with thread‑tail only
- Sanitizer: handles double spaces, em-dash, and newline after mention without eating the next token

## Runtime Flags
- `REQUIRE_MENTION_IN_GUILDS` (default `1`): require an explicit mention before routing guild messages.
- `ALLOW_REPLY_TO_BOT_WITHOUT_MENTION` (default `1`): allow guild follow-ups that are direct replies to the bot even without a fresh mention.
- `DM_REQUIRE_MENTION` (default `0`): enable if you want DMs to require an explicit mention.

## Tests (described)
- Mentions (unified): tiny tokens route to TEXT; no nag
- Media intent vs nag: nag only for explicit intent without in‑scope media/URLs
- Reply & thread correctness: parent/newest resolution verified
- Sanitizer edge cases: spacing, em‑dashes, newlines
- No stale bleed: busy channel does not import old memory; verify drop_stale
- Parent fetch failure/archived threads: clean fallback to replying to the trigger; no exceptions

## Files touched
- [bot/router.py](bot/router.py)
- Tests live in: [tests/test_router_reply_routing.py](tests/test_router_reply_routing.py)
