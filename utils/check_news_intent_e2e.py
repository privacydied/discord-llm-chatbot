"""End-to-end check: phrasing -> intent -> live Guardian fetch -> digest. [REH]

Usage: uv run python utils/check_news_intent_e2e.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv("/volume1/py/discord-llm-chatbot/.env")

from bot.news.headlines import fetch_headlines, render_digest  # noqa: E402
from bot.news.intent import detect_news_intent  # noqa: E402

PHRASES = [
    "what's happening in the world today?",
    "whats going on in the news this week",
    "any news about artificial intelligence",
    "give me the headlines",
    "what's wrong with my deploy?",  # must NOT trigger
]


async def main() -> None:
    cfg = {"GUARDIAN_API_KEY": os.getenv("GUARDIAN_API_KEY")}
    for phrase in PHRASES:
        query = detect_news_intent(phrase)
        if query is None:
            print(f"\n{phrase!r}\n  -> no news intent (correct for non-news phrasing)")
            continue
        headlines = await fetch_headlines(query.topic, cfg=cfg, days=query.days, limit=3)
        print(f"\n{phrase!r}\n  -> topic={query.topic!r} days={query.days} hits={len(headlines)}")
        for headline in headlines[:2]:
            print(f"     • {headline.title[:88]}")
        if headlines:
            print(f"     digest chars: {len(render_digest(headlines, query.topic, query.days))}")


if __name__ == "__main__":
    asyncio.run(main())
