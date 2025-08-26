"""
Vision subsystem import and init smoke test.

- Imports vision exports from bot.vision
- Imports OptimizedRouter and constructs with a dummy bot
- Imports vision commands cog module
- Prints simple status so CI/logs can assert success quickly
"""
from __future__ import annotations

import sys

from bot.util.logging import get_logger

logger = get_logger("utils.vision_smoke")


class _DummyUser:
    id = 0


class _DummyBot:
    def __init__(self) -> None:
        self.user = _DummyUser()
        # Ensure vision stays disabled for this smoke test (no external calls)
        self.config = {"VISION_ENABLED": False}


def main() -> int:
    try:
        logger.info("Importing bot.vision exports...")
        from bot.vision import VisionIntentRouter, VisionOrchestrator  # noqa: F401
        logger.info("Importing OptimizedRouter...")
        from bot.optimized_router import OptimizedRouter  # noqa: F401

        logger.info("Instantiating OptimizedRouter with dummy bot...")
        router = OptimizedRouter(_DummyBot())
        logger.info(
            "OptimizedRouter created",
            extra={
                "event": "router.init",
                "detail": {
                    "vision_enabled_import": True,
                    "vision_config_flag": False,
                    "vision_intent_router": router._vision_intent_router is not None,
                    "vision_orchestrator": router._vision_orchestrator is not None,
                },
            },
        )

        logger.info("Importing vision_commands module...")
        import bot.commands.vision_commands  # noqa: F401

        print("VISION_IMPORT_SMOKE_TEST_OK")
        return 0
    except Exception as e:
        logger.error("Vision smoke test failed", exc_info=True, extra={"event": "router.smoke.fail", "detail": {"error": str(e)}})
        print("VISION_IMPORT_SMOKE_TEST_FAIL", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
