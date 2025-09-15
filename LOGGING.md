# Logging

Helpers emit single-line breadcrumbs using existing schema:

- `scope_resolved`: {case, scope}
- `chat_text`: {has_text, length}
- `media_intent`: boolean
- `harvest_complete`: counts of urls/attachments
- `route_selected`: final route
- `reply_target_ok`: chosen target id
- `local_context`: item and character counts after truncation

---

## Runbook: Reply Anchoring & Mention Monitoring

These single-line events are emitted during send-time anchoring. Use them to validate human-first anchoring and single-ping behavior.

- subsys=route event=scope_resolved case=<thread|reply|plain> scope=<id>
- subsys=route event=reply_target_ok target=<id|none>
- subsys=mention event=recipient_resolved user=<id|none> reason=<target_author|fallback_human|no_human>
- subsys=mention event=ping_strategy mode=<reply_ping|explicit_mention|none>
- subsys=route event=send mode=<delete_and_resend|direct>

### What to watch

- Self-anchor detector: any `reply_target_ok` whose target author == bot.
- Double-ping detector: more than one `ping_strategy` for the same msg send.
- Scope fencing: look for `scope_resolved` changes across messages, and for memory pruning lines (e.g., `local_context` + truncation flags) when tails are long.
- Safe fallback: after failures, expect a plain send with no mention; check `send mode` and absence of `replied_user` pings.

### Grep examples (pretty console sink)

```bash
# Scope resolution trail
grep -F "subsys=route" logs/app.log | grep -F "event=scope_resolved"

# Reply target selection
grep -F "event=reply_target_ok" logs/app.log

# Recipient and ping strategy
grep -F "subsys=mention" logs/app.log | grep -E "recipient_resolved|ping_strategy"

# Detect potential self-anchors (join with author lookup in your viewer)
grep -F "event=reply_target_ok" logs/app.log | grep -F "target="

# Delete-and-resend path (placeholder retarget)
grep -F "event=send" logs/app.log | grep -F "delete_and_resend"
```

### JSON sink filtering (jq examples)

```bash
jq 'select(.subsys=="route" and .event=="scope_resolved") | {ts,level,case: .detail.case, scope: .detail.scope}' logs/structured.jsonl

jq 'select(.subsys=="route" and .event=="reply_target_ok") | {ts, target: .detail.id}' logs/structured.jsonl

jq 'select(.subsys=="mention" and .event=="recipient_resolved") | {ts, user: .detail.user, reason: .detail.reason}' logs/structured.jsonl

jq 'select(.subsys=="mention" and .event=="ping_strategy") | {ts, mode: .detail.mode}' logs/structured.jsonl

jq 'select(.subsys=="route" and .event=="send") | {ts, mode: .detail.mode}' logs/structured.jsonl
```

Notes: Dual sinks remain in place (pretty + JSON). Timestamps and color palette unchanged. Events are concise to keep grepping fast.
