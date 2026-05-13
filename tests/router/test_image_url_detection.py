import pytest

from bot.modality import InputModality, _map_url_to_modality
from bot.router_components.input_harvest import is_direct_image_url


def test_direct_image_url_detects_twitter_format_query() -> None:
    assert is_direct_image_url("https://pbs.twimg.com/media/ABC123?format=jpg&name=large")
    assert is_direct_image_url("https://pbs.twimg.com/media/ABC123?format=png&name=small")


@pytest.mark.asyncio
async def test_map_url_to_modality_routes_twitter_format_query_as_image() -> None:
    modality = await _map_url_to_modality("https://pbs.twimg.com/media/ABC123?format=jpg&name=large")

    assert modality == InputModality.SINGLE_IMAGE
