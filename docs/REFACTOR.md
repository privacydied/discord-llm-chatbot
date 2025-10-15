# Refactor Notes

- Introduced `bot/decision_helpers.py` housing pure helpers for scope resolution, text extraction, media intent, I/O harvest, route selection, reply targeting, and context composition.
- `contextual_brain_infer` now delegates scope resolution to `resolve_scope` reducing inline complexity and improving log consistency.
