from unittest.mock import Mock

import pytest

from bot.commands.image_upgrade_commands import ImageUpgradeManager


@pytest.fixture
def upgrade_manager():
    bot = Mock()
    bot.config = {
        "IMAGE_UPGRADE_REACTIONS": "🖼️,🔎,🏷️,🧠,↩️",
        "VISION_CAPTION_STYLE": "neutral",
    }
    bot.logger = Mock()
    bot.get_channel = Mock(return_value=Mock())
    return ImageUpgradeManager(bot)


@pytest.fixture(autouse=True)
def _disable_vision_auto_discovery(monkeypatch):
    """Keep the OpenRouter vision-model discovery cache out of unrelated tests.

    The discovery ladder reads a real on-disk cache (vision_data/), which would
    otherwise make ladder-shaped assertions depend on whatever OpenRouter listed
    the last time the bot ran. Tests that exercise discovery re-enable it.
    """
    monkeypatch.setenv("VISION_AUTO_DISCOVERY", "0")
    # Belt and braces: never let a test fire a live OpenRouter liveness probe.
    monkeypatch.setenv("VISION_DISCOVERY_PROBE", "0")
