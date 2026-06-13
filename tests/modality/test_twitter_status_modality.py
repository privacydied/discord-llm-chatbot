import pytest

from bot.modality import InputModality, _map_url_to_modality


@pytest.mark.asyncio
async def test_twitter_status_is_general_url() -> None:
    url = "https://x.com/user/status/1234567890123456789"
    modality = await _map_url_to_modality(url)
    assert modality == InputModality.GENERAL_URL


@pytest.mark.asyncio
async def test_twitter_broadcast_is_video_url() -> None:
    url = "https://x.com/i/broadcasts/AbCdEfGh"
    modality = await _map_url_to_modality(url)
    assert modality == InputModality.VIDEO_URL
