#!/usr/bin/env python3
"""Test Novita.ai image generation integration."""

import pytest

pytestmark = pytest.mark.skip(reason="Requires live Novita API access")

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.utils.logging import init_logging
from bot.vision.providers.novita_adapter import NovitaAdapter
from bot.vision.types import VisionRequest, VisionTask


async def test_novita_image_generation() -> bool | None:
    """Test Novita.ai text-to-image generation."""
    # Initialize logging
    init_logging()

    # Load config from yoroi.env
    config = {}
    env_path = Path(__file__).parent.parent / "yoroi.env"

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key] = value
    else:
        return False

    # Check required config
    if not config.get("VISION_API_KEY"):
        return False

    # Create adapter
    adapter = NovitaAdapter(config, {})

    # Create test request
    request = VisionRequest(
        task=VisionTask.TEXT_TO_IMAGE,
        prompt="A quiet girl with short brown hair sitting by a misty lake at dawn",
        width=1024,
        height=1024,
        idempotency_key="test_novita_img_001",
    )

    try:
        # Test generation
        response = await adapter.generate(request, "qwen-image-txt2img")

        if response.success:
            for artifact in response.artifacts:
                if artifact.exists():
                    size_mb = artifact.stat().st_size / (1024 * 1024)
                else:
                    pass

            return True
        return False

    except Exception as e:
        return False

    finally:
        await adapter.close()


if __name__ == "__main__":
    success = asyncio.run(test_novita_image_generation())
    sys.exit(0 if success else 1)
