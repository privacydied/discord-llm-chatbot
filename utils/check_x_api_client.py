from __future__ import annotations

import asyncio
from bot.x_api_client import XApiClient
from bot.util.logging import get_logger

logger = get_logger(__name__)

TEST_INPUTS = [
    "https://twitter.com/someuser/status/1777777777777777777",
    "https://x.com/someuser/status/1777777777777777777?s=20",
    "https://vxtwitter.com/someuser/status/1777777777777777777/photo/1",
    "1777777777777777777",
    "https://example.com/not-twitter",
]


async def main() -> None:
    logger.info("Sanity-checking XApiClient helpers (no network)")

    # Instantiate with no token to ensure headers logic works safely
    client = XApiClient(bearer_token=None)

    for val in TEST_INPUTS:
        tid = client.extract_tweet_id(val)
        logger.info("extract_tweet_id", extra={"detail": {"input": val, "tweet_id": tid}})

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
