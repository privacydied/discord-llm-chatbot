"""Soft-delete stored inferred memories that the current curator would reject.

Earlier curator rules stored ordinary chat (generic "you always ...", questions,
text about other members, pasted output) as memories. This re-runs every live
``inferred_curated`` row through today's ``curate_inferred_candidate`` and
soft-deletes the ones it rejects. Explicit memories (``!remember``-style) are never touched.

SQLite only: retrieval reads rows through ``fetch_active_by_ids``, which skips
soft-deleted rows, so stale Chroma vectors are harmless and the running bot's
Chroma client is left alone.

    uv run python utils/purge_rejected_memories.py            # dry run
    uv run python utils/purge_rejected_memories.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.memory.curator import CuratedMemoryCurator  # noqa: E402
from bot.memory.persistent_store import PersistentMemoryStore  # noqa: E402

DEFAULT_DB = PROJECT_ROOT / "data" / "memory.db"
INFERRED_SOURCE = "inferred_curated"


def _live_inferred(db: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT memory_id, text FROM curated_memories WHERE deleted_at IS NULL AND source = ?",
            (INFERRED_SOURCE,),
        ).fetchall()
    return [(str(mid), str(text or "")) for mid, text in rows]


def rejected_ids(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    curator = CuratedMemoryCurator(default_ttl_days=180, temp_ttl_days=14, min_importance=0.55)
    out = []
    for memory_id, text in rows:
        if curator.curate_inferred_candidate(user_id="0", text=text) is None:
            out.append((memory_id, text))
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--apply", action="store_true", help="soft-delete (default: dry run)")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    rows = _live_inferred(args.db)
    doomed = rejected_ids(rows)
    for memory_id, text in doomed:
        print(f"{memory_id}  {text[:100]!r}")
    print(f"\n{len(doomed)}/{len(rows)} live inferred memories rejected by current rules")
    if not args.apply:
        print("dry run: pass --apply to soft-delete")
        return 0

    store = PersistentMemoryStore(args.db)
    await store.initialize()
    deleted = sum([await store.soft_delete_memory(memory_id) for memory_id, _ in doomed])
    print(f"soft-deleted {deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
