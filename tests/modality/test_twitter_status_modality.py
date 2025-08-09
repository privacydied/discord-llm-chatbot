import pytest

from bot.modality import _map_url_to_modality, InputModality


@pytest.mark.asyncio
async def test_twitter_status_is_general_url():
    url = "https://x.com/user/status/1234567890123456789"
    modality = await _map_url_to_modality(url)
    assert modality == InputModality.GENERAL_URL


@pytest.mark.asyncio
async def test_twitter_broadcast_is_video_url():
    url = "https://x.com/i/broadcasts/AbCdEfGh"
    modality = await _map_url_to_modality(url)
    assert modality == InputModality.VIDEO_URL
