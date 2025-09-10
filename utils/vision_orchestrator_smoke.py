"""
Vision Orchestrator readiness smoke test (no external API calls).

- Forces minimal env for Vision to initialize
- Starts VisionOrchestrator and reports readiness + provider list
- Does not perform any network requests (startup only creates aiohttp sessions)

Run:
    uv run python -m utils.vision_orchestrator_smoke
"""

from __future__ import annotations

import os
import asyncio
from typing import List, Optional

from bot.utils.logging import get_logger

logger = get_logger("utils.vision_orch_smoke")


async def main() -> int:
    # Ensure vision is enabled and a dummy API key is present so providers initialize
    os.environ.setdefault("VISION_ENABLED", "true")
    os.environ.setdefault("VISION_T2I_ENABLED", "true")
    # Non-empty placeholder; real key not required for this test
    os.environ.setdefault("VISION_API_KEY", "TEST_KEY")

    orch: Optional["VisionOrchestrator"] = None
    try:
        from bot.vision import VisionOrchestrator  # type: ignore

        orch = VisionOrchestrator()
        await orch.start()

        # Collect provider list without touching network
        providers: List[str] = []
        try:
            adapter = getattr(orch.gateway, "adapter", None)
            if adapter and getattr(adapter, "providers", None):
                providers = list(adapter.providers.keys())
        except Exception:
            providers = []

        logger.info(
            "Vision orchestrator smoke:",
            extra={
                "event": "vision.orch.smoke",
                "detail": {
                    "ready": orch.ready,
                    "reason": getattr(orch, "reason", ""),
                    "providers": providers,
                },
            },
        )
        print(
            f"VISION_ORCH_SMOKE_OK ready={orch.ready} reason={getattr(orch, 'reason', '')} providers={providers}"
        )
        return 0
    except Exception as e:
        logger.error(
            "Vision orchestrator smoke failed",
            exc_info=True,
            extra={"event": "vision.orch.smoke.fail", "detail": {"error": str(e)}},
        )
        print("VISION_ORCH_SMOKE_FAIL")
        return 1
    finally:
        if orch is not None:
            try:
                await orch.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
