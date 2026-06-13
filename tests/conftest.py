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
