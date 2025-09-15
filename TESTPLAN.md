# Test Plan

## Unit
- `resolve_scope`: thread, reply, plain cases.
- `extract_chat_text` + `detect_media_intent` + `choose_route`: ensures nag when media intent without media.
- `select_reply_target` and `compose_context`: verify target resolution and dedupe/truncation.

## Integration
- `tests/decision/test_decision_helpers.py` covers above helpers with deterministic data.
