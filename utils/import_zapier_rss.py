"""Convert a Zapier "RSS -> Discord channel message" export into RSS_FEEDS_FILE.

    python utils/import_zapier_rss.py exported-zap.json [--channel ID] [--out data/rss_feeds.json]

Each zap's read step (RSSCLIAPI new_feed) gives the feed URL and trigger style
(smart -> new, everything -> changed); its Discord write step gives the channel,
message template ({{1__raw__title}} mappings converted), username and avatar.
``--channel`` overrides every zap's channel (e.g. the old one was deleted).
Existing feeds in --out with the same URL are replaced; others are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.rss.feeds import TRIGGER_CHANGED, TRIGGER_NEW, zapier_template  # noqa: E402

_TRIGGER = {"smart": TRIGGER_NEW, "everything": TRIGGER_CHANGED}


def convert(export: dict, channel_override: int | None = None) -> list[dict]:
    feeds = []
    for zap in export.get("zaps") or []:
        nodes = list((zap.get("nodes") or {}).values())
        read = next((n for n in nodes if n.get("type_of") == "read" and n.get("action") == "new_feed"), None)
        write = next((n for n in nodes if n.get("type_of") == "write" and n.get("action") == "send_channel_message"), None)
        if read is None or write is None:
            print(f"skip zap {zap.get('title')!r}: not an RSS -> Discord message zap", file=sys.stderr)
            continue
        params = write.get("params") or {}
        feeds.append(
            {
                "name": params.get("username") or zap.get("title") or "feed",
                "url": read["params"]["url"],
                "channel_id": int(channel_override or params["channel_id"]),
                "username": params.get("username") or "",
                "avatar_url": params.get("avatar_url") or "",
                "template": zapier_template(params.get("content") or ""),
                "trigger": _TRIGGER.get((read.get("params") or {}).get("trigger_style"), TRIGGER_NEW),
                "enabled": True,
            },
        )
    return feeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("export")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--out", default="data/rss_feeds.json")
    args = parser.parse_args()
    feeds = convert(json.loads(Path(args.export).read_text()), args.channel)
    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else {"feeds": []}
    urls = {f["url"] for f in feeds}
    merged = [f for f in existing.get("feeds", []) if f.get("url") not in urls] + feeds
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"feeds": merged}, indent=2, ensure_ascii=False) + "\n")
    for feed in feeds:
        print(f"imported {feed['name']!r} -> channel {feed['channel_id']} trigger={feed['trigger']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
