import asyncio

import pytest

from bot.hear import BasePCMStream


class _BlockedProducerStream(BasePCMStream):
    """Producer that blocks on queue put when no consumer drains frames."""

    async def _produce(self) -> None:
        await self._queue.put(b"a")
        await self._queue.put(b"b")
        # With queue_depth=2 and no consumer, this put blocks indefinitely
        # unless abort() cancels producer and signals stream end.
        await self._queue.put(b"c")


@pytest.mark.asyncio
async def test_abort_finalize_does_not_hang_with_full_queue():
    stream = _BlockedProducerStream(sample_rate=16000, frame_samples=160, queue_depth=2)
    await stream.start()
    await asyncio.sleep(0.05)

    await stream.abort()
    await asyncio.wait_for(stream.finalize(success=False), timeout=1.0)
