#!/usr/bin/env python
import asyncio
import os
import sys
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

# Import Router and XApiClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot.router import Router  # noqa: E402
from bot.x_api_client import XApiClient  # noqa: E402

console = Console()


class MetricsStub:
    def increment(self, name: str, labels: Optional[Dict[str, str]] = None):
        console.log(f"[bold cyan]metric[/] {name} labels={labels or {}}")

    def inc(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.increment(metric_name, labels)


class BotStub:
    def __init__(self):
        # Minimal config pulled from env with sensible defaults
        self.config: Dict[str, Any] = {
            "X_SYNDICATION_ENABLED": os.getenv("X_SYNDICATION_ENABLED", "true").lower() == "true",
            "X_SYNDICATION_FIRST": os.getenv("X_SYNDICATION_FIRST", "false").lower() == "true",
            "X_SYNDICATION_TTL_S": float(os.getenv("X_SYNDICATION_TTL_S", "900")),
            "X_SYNDICATION_TIMEOUT_MS": int(os.getenv("X_SYNDICATION_TIMEOUT_MS", "4000")),
            "X_API_REQUIRE_API_FOR_TWITTER": os.getenv("X_API_REQUIRE_API_FOR_TWITTER", "false").lower() == "true",
        }
        self.metrics = MetricsStub()
        self.loop = asyncio.get_event_loop()
        self.tts_manager = None


async def main():
    if len(sys.argv) < 2:
        console.print("Usage: uv run python utils/check_syndication.py <tweet_id_or_url>", style="yellow")
        sys.exit(2)

    raw = sys.argv[1]
    tweet_id = XApiClient.extract_tweet_id(raw)
    if not tweet_id:
        console.print(f"Could not extract tweet id from: {raw}", style="red")
        sys.exit(1)

    bot = BotStub()
    router = Router(bot)

    console.rule("Syndication Fetch Test")
    console.print(Panel.fit(f"Testing syndication for tweet_id: [bold]{tweet_id}[/]"))
    data = await router._get_tweet_via_syndication(tweet_id)
    if not data:
        console.print(Panel.fit("No data returned (None)", title="Result", style="red"))
        sys.exit(1)

    console.print(Panel.fit("Success", title="Result", style="green"))
    # Pretty-print some key fields
    fields = {
        k: data.get(k)
        for k in ("id_str", "created_at", "text", "full_text", "user", "photos")
        if isinstance(data, dict)
    }
    console.print(JSON.from_data(fields))


if __name__ == "__main__":
    asyncio.run(main())
