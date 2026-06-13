import pytest

from bot import openai_backend
from bot.exceptions import APIError


@pytest.mark.asyncio
async def test_generate_openai_response_without_openai_package_raises_clean_apierror() -> None:
    if openai_backend._openai is not None:
        pytest.skip("openai package is installed in this environment")

    with pytest.raises(APIError) as exc_info:
        await openai_backend.generate_openai_response(
            "hello",
            system_prompt="test",
        )

    assert "openai" in str(exc_info.value).lower()
