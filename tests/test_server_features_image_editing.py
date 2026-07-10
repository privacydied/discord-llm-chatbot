"""Coverage for the new `image_editing` per-guild feature toggle (conversational
image-edit route), added to bot/server_features.py's existing FEATURE_DEFAULTS/
FEATURE_ALIASES dict-based system.
"""

from unittest.mock import patch

from bot.server_features import (
    FEATURE_DEFAULTS,
    is_server_feature_enabled,
    normalize_feature_name,
    set_server_feature_toggle,
)


def test_image_editing_defaults_to_enabled() -> None:
    assert FEATURE_DEFAULTS["image_editing"] is True
    assert is_server_feature_enabled(None, "image_editing") is True


def test_image_editing_aliases_normalize() -> None:
    for alias in ("imgedit", "image_edit", "edit", "IMAGE_EDITING", "Image-Edit"):
        assert normalize_feature_name(alias) == "image_editing"


def test_toggle_off_is_respected(tmp_path) -> None:
    profile = {"custom_data": {}}
    with (
        patch("bot.server_features.get_server_profile", return_value=profile),
        patch("bot.server_features.save_server_profile"),
    ):
        toggles = set_server_feature_toggle(123, "imgedit", False)
        assert toggles["image_editing"] is False
        assert is_server_feature_enabled(123, "image_editing") is False
