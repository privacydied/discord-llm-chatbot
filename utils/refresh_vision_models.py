#!/usr/bin/env python3
"""Manually refresh the OpenRouter free vision-model cache and print the ladder.

Usage: uv run python utils/refresh_vision_models.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.vision.free_model_discovery import cache_path, discover_free_vision_models  # noqa: E402
from bot.vision.free_model_probe import load_quarantine, quarantine_path  # noqa: E402


async def main() -> int:
    models = await discover_free_vision_models(force=True)
    if not models:
        print("No free image-capable models discovered (cache unchanged).")
        return 1
    print(f"Ladder cache: {cache_path()}")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")

    banned = load_quarantine()
    if banned:
        print(f"\nQuarantined (see {quarantine_path()}):")
        for model, meta in sorted(banned.items()):
            print(f"  - {model}: {str(meta.get('reason', ''))[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
