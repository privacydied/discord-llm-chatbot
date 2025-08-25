import os
import time

# Respect user rules: tests/util scripts live in utils/
# Configure via env or command line export

start = time.time()
from bot.stt import stt_manager  # noqa: E402
import asyncio  # noqa: E402

import_time = time.time() - start
print(f"Imported bot.stt in {import_time:.3f}s")

async def main():
    t0 = time.time()
    # Wait briefly for background init to complete if running
    await asyncio.get_running_loop().run_in_executor(None, stt_manager._ready_event.wait, float(os.getenv("STT_INIT_TIMEOUT", "8")))
    ready_time = time.time() - t0
    print(f"Waited {ready_time:.3f}s for ready event")
    print(f"Available: {stt_manager.available}")

if __name__ == "__main__":
    asyncio.run(main())
