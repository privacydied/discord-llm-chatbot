# Audit

## Hotspots
- **Scope & memory**: scope resolver mixes thread, reply, and plain logic inline in `contextual_brain_infer`.
- **Router & gates**: text presence and media intent intertwined with send-time routing.

## Invariants
- Exactly one of `thread`, `reply`, or `plain` is selected for any trigger.
- Route selection depends on cleaned text, media intent, and in-scope I/O.
