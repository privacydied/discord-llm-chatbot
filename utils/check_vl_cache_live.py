"""Live check: the second look at the same image must not re-run inference. [PA]

Calls the real vision model once, then again with a re-signed URL, and reports
timings plus the cache's own hit counters.

Usage: uv run python utils/check_vl_cache_live.py
"""

import asyncio
import time

from dotenv import load_dotenv

load_dotenv("/volume1/py/discord-llm-chatbot/.env")

from bot.single_flight_cache import CacheFamily, get_cache  # noqa: E402
from bot.tools.builtins.vision import _describe, cache_identity  # noqa: E402


class _Ref:
    def __init__(self, url, filename="pic.png"):
        self.url = url
        self.filename = filename


class _Msg:
    def __init__(self, id_):
        self.id = id_


# Same image, different expiring signature — exactly what Discord does.
IMAGE_A = "https://picsum.photos/id/237/320/240?ex=aaaa&is=bbbb&hm=cccc"
IMAGE_B = "https://picsum.photos/id/237/320/240?ex=zzzz&is=yyyy&hm=xxxx"
QUESTION = "What animal is this? Answer in one short sentence."


MESSAGE = _Msg(123456789)


async def timed(url: str, question: str) -> tuple[str | None, float]:
    """Describe via the same identity path view_image uses."""
    identity = cache_identity(MESSAGE, _Ref(url))
    start = time.monotonic()
    result = await _describe(url, question, {}, identity)
    return result, time.monotonic() - start


async def main() -> None:
    ident_a = cache_identity(MESSAGE, _Ref(IMAGE_A))
    ident_b = cache_identity(MESSAGE, _Ref(IMAGE_B))
    print(f"identity A: {ident_a}")
    print(f"identity B: {ident_b}")
    print(f"identities match: {ident_a == ident_b}\n")

    first, t1 = await timed(IMAGE_A, QUESTION)
    print(f"1st call (cold)          {t1:6.2f}s -> {str(first)[:90]}")

    second, t2 = await timed(IMAGE_B, QUESTION)
    print(f"2nd call (re-signed URL) {t2:6.2f}s -> {str(second)[:90]}")

    third, t3 = await timed(IMAGE_A, "What colour is it? One word.")
    print(f"3rd call (new question)  {t3:6.2f}s -> {str(third)[:90]}")

    metrics = get_cache().metrics
    print(f"\ncache: requests={metrics.total_requests} hits={metrics.cache_hits} misses={metrics.cache_misses}")
    print(f"family TTL: {get_cache().family_ttls[CacheFamily.VL_DESCRIPTION]:.0f}s")

    if first and second:
        print(f"\nsame answer reused: {first == second}")
        print(f"speedup on repeat : {(t1 / t2):.0f}x" if t2 > 0 else "")


if __name__ == "__main__":
    asyncio.run(main())
