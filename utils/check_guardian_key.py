"""Ad-hoc check that GUARDIAN_API_KEY works end to end. [REH]

Usage: uv run python utils/check_guardian_key.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv("/volume1/py/discord-llm-chatbot/.env")

from bot.news.providers.guardian import GuardianNewsProvider  # noqa: E402

ARTICLE = "https://www.theguardian.com/us-news/2026/jul/10/trump-climate-report-matthew-wielicki"


async def main() -> None:
    key = os.getenv("GUARDIAN_API_KEY")
    print("key loaded:", bool(key), "len:", len(key or ""))
    provider = GuardianNewsProvider(key)
    print("enabled:", provider.enabled, "supports:", provider.supports(ARTICLE))

    article = await provider.fetch(ARTICLE)
    if not article:
        print("FETCH RETURNED NONE")
        return
    print("provider:", article.provider)
    print("title:", article.title)
    print("author:", article.author)
    print("body chars:", len(article.body))
    print("body head:", article.body[:200].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
