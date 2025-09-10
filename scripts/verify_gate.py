#!/usr/bin/env python3
"""
Quick verification for SSOT gate: ensures non-addressed messages are blocked.

Runs Router._should_process_message() with a fake message in a guild channel that:
- is not a DM
- has no mention
- is not a reply to the bot
- is not in a bot-owned thread
- is not from an owner
- has no command prefix

Expected: gate returns False (blocked).
"""

import asyncio

from bot.router import Router
from bot.metrics.null_metrics import NoopMetrics


class FakeUser:
    def __init__(self, user_id: int):
        self.id = user_id


class FakeAuthor(FakeUser):
    pass


class FakeGuild:
    def __init__(self, gid: int):
        self.id = gid


class FakeChannel:
    """A generic guild text channel (not DM, not Thread)."""

    def __init__(self):
        # No special fields needed; importantly, not an instance of discord.Thread
        pass


class FakeMessage:
    def __init__(self, content: str, author_id: int, guild_id: int):
        self.id = 1
        self.content = content
        self.author = FakeAuthor(author_id)
        self.guild = FakeGuild(guild_id)
        self.channel = FakeChannel()
        self.mentions = []  # no mention of the bot
        self.reference = None  # not a reply


class FakeBot:
    def __init__(self, bot_user_id: int, config: dict):
        self.user = FakeUser(bot_user_id)
        self.config = config
        self.metrics = NoopMetrics()
        self.loop = asyncio.get_event_loop()
        self.tts_manager = None


async def main():
    # Config: gate enabled, all triggers present
    config = {
        "BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO": True,
        "REPLY_TRIGGERS": [
            "dm",
            "mention",
            "reply",
            "bot_threads",
            "owner",
            "command_prefix",
        ],
        "COMMAND_PREFIX": "!",
        "OWNER_IDS": [42],  # owners not including our author
    }

    bot = FakeBot(bot_user_id=9999, config=config)
    router = Router(bot)

    # Non-addressed message in a guild channel
    msg = FakeMessage(content="hello there", author_id=123, guild_id=654321)

    allowed = router._should_process_message(msg)
    if allowed:
        print("✖ FAIL: Gate allowed a non-addressed message")
        exit(1)
    else:
        print("✔ PASS: Gate blocked a non-addressed message as expected")
        exit(0)


if __name__ == "__main__":
    asyncio.run(main())
