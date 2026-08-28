"""NotSoBot provider plugin tests.

Covers submit/poll/fetch_result success and error paths, provider registration,
and auto-promotion in provider_order when NOTSOBOT_API_TOKEN is configured.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.vision.types import VisionError, VisionErrorType, VisionTask
from bot.vision.unified_adapter import (
    NotSoBotPlugin,
    NormalizedRequest,
    UnifiedStatus,
    UnifiedVisionAdapter,
)


# --- Fixtures ----------------------------------------------------------------


def _make_request(
    task: VisionTask = VisionTask.IMAGE_TO_IMAGE,
    prompt: str = "a cat",
    input_image_url: str | None = None,
    input_image_data: bytes | None = None,
) -> NormalizedRequest:
    return NormalizedRequest(
        task=task,
        prompt=prompt,
        input_image_url=input_image_url,
        input_image_data=input_image_data,
        width=1024,
        height=1024,
        seed=42,
        steps=20,
    )


def _make_plugin(token: str = "test-notsobot-token-0123456789") -> NotSoBotPlugin:
    config = {
        "base_url": "https://notsobot.com",
        "api_path": "/api",
        "NOTSOBOT_API_TOKEN": token,
    }
    return NotSoBotPlugin("notsobot", config, token)


# --- Capabilities ------------------------------------------------------------


class TestNotSoBotCapabilities:
    def test_supports_image_tasks(self):
        plugin = _make_plugin()
        caps = plugin.capabilities()
        modes = caps["modes"]
        assert VisionTask.TEXT_TO_IMAGE in modes
        assert VisionTask.IMAGE_TO_IMAGE in modes
        assert VisionTask.VIDEO_GENERATION in modes

    def test_max_size_and_steps(self):
        plugin = _make_plugin()
        caps = plugin.capabilities()
        assert caps["max_size"] == (2048, 2048)
        assert caps["max_steps"] == 50


# --- Header & body building --------------------------------------------------


class TestNotSoBotRequestBuilding:
    def test_build_headers_with_token(self):
        plugin = _make_plugin(token="my-token")
        req = _make_request()
        headers = plugin._build_headers(req)
        assert headers["authorization"] == "Bot my-token"
        assert headers["content-type"] == "application/json"

    def test_build_headers_with_discord_user(self):
        plugin = _make_plugin()
        req = _make_request()
        # Attach discord user context (optional)
        req.discord_user = {
            "id": "123456",
            "username": "testuser",
            "discriminator": "0001",
            "bot": False,
            "avatar": "abc123",
        }
        req.channel_id = "999"
        req.guild_id = "777"
        headers = plugin._build_headers(req)
        assert headers["x-user-id"] == "123456"
        assert "x-user" in headers
        assert headers["x-channel-id"] == "999"
        assert headers["x-guild-id"] == "777"
        assert headers["x-server-id"] == "777"

    def test_build_body_text_to_image(self):
        plugin = _make_plugin()
        req = _make_request(task=VisionTask.TEXT_TO_IMAGE, prompt="sunset")
        body = plugin._build_body(req)
        assert body["query"] == "sunset"
        assert body["safe"] is True
        assert "strength" not in body  # strength only for IMAGE_TO_IMAGE

    def test_build_body_image_to_image_with_url(self):
        plugin = _make_plugin()
        req = _make_request(input_image_url="https://example.com/img.png")
        body = plugin._build_body(req)
        assert body["urls"] == ["https://example.com/img.png"]

    def test_build_body_image_to_image_with_data(self):
        plugin = _make_plugin()
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        req = _make_request(input_image_data=raw)
        body = plugin._build_body(req)
        assert "urls" in body
        assert body["urls"][0].startswith("data:image/png;base64,")


# --- submit() with mocked session -------------------------------------------


def _mock_response(status: int, json_data: dict | None = None, text: str = ""):
    """Create a mocked aiohttp response supporting async context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=text or json.dumps(json_data or {}))
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)
    else:
        resp.json = AsyncMock(return_value={})
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)
    return resp


class TestNotSoBotSubmit:
    @pytest.mark.asyncio
    async def test_submit_success_returns_job_id(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(200, {"id": "job-abc-123", "status": "queued"}))
        plugin.session = mock_session

        req = _make_request()
        job_id = await plugin.submit(req)
        assert job_id == "job-abc-123"

    @pytest.mark.asyncio
    async def test_submit_401_raises_auth_error(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(401, text="Unauthorized"))
        plugin.session = mock_session

        req = _make_request()
        with pytest.raises(VisionError) as exc_info:
            await plugin.submit(req)
        assert exc_info.value.error_type == VisionErrorType.AUTHENTICATION_ERROR

    @pytest.mark.asyncio
    async def test_submit_403_raises_quota_exceeded(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(403, text="NOT_ENOUGH_BALANCE"))
        plugin.session = mock_session

        req = _make_request()
        with pytest.raises(VisionError) as exc_info:
            await plugin.submit(req)
        assert exc_info.value.error_type == VisionErrorType.QUOTA_EXCEEDED

    @pytest.mark.asyncio
    async def test_submit_404_raises_unsupported_task(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(404, text="This model is unavailable for free."))
        plugin.session = mock_session

        req = _make_request()
        with pytest.raises(VisionError) as exc_info:
            await plugin.submit(req)
        assert exc_info.value.error_type == VisionErrorType.UNSUPPORTED_TASK

    @pytest.mark.asyncio
    async def test_submit_429_raises_rate_limited(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(429, text="rate limited"))
        plugin.session = mock_session

        req = _make_request()
        with pytest.raises(VisionError) as exc_info:
            await plugin.submit(req)
        assert exc_info.value.error_type == VisionErrorType.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_submit_500_raises_server_error(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(500, text="internal error"))
        plugin.session = mock_session

        req = _make_request()
        with pytest.raises(VisionError) as exc_info:
            await plugin.submit(req)
        assert exc_info.value.error_type == VisionErrorType.SERVER_ERROR

    @pytest.mark.asyncio
    async def test_submit_no_job_id_raises_provider_error(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(200, {"status": "queued"}))
        plugin.session = mock_session

        req = _make_request()
        with pytest.raises(VisionError) as exc_info:
            await plugin.submit(req)
        assert exc_info.value.error_type == VisionErrorType.PROVIDER_ERROR


# --- poll() with mocked session ---------------------------------------------


class TestNotSoBotPoll:
    @pytest.mark.asyncio
    async def test_poll_completes(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        # First call: still running (no result yet). Second call: completed.
        mock_session.get = MagicMock(
            side_effect=[
                _mock_response(200, {"result": {}}),
                _mock_response(
                    200,
                    {"result": {"response": {"urls": ["https://cdn.notsobot.com/img.png"]}}},
                ),
            ]
        )
        plugin.session = mock_session

        status = await plugin.poll("job-123")
        assert status.status == UnifiedStatus.COMPLETED
        assert status.progress_percentage == 100

    @pytest.mark.asyncio
    async def test_poll_job_error(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(200, {"result": {"error": "something went wrong"}}))
        plugin.session = mock_session

        status = await plugin.poll("job-123")
        assert status.status == UnifiedStatus.FAILED
        assert "something went wrong" in status.phase

    @pytest.mark.asyncio
    async def test_poll_404_returns_failed(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(404, text="not found"))
        plugin.session = mock_session

        status = await plugin.poll("job-123")
        assert status.status == UnifiedStatus.FAILED


# --- Provider registration & auto-promotion ----------------------------------


class TestNotSoBotRegistration:
    def test_notsobot_initialized_when_token_present(self):
        config = {
            "NOTSOBOT_API_TOKEN": "test-notsobot-token-0123456789",
            "VISION_API_KEY": "test-vision-api-key-0123456789",
        }
        adapter = UnifiedVisionAdapter(config)
        assert "notsobot" in adapter.providers

    def test_notsobot_skipped_when_no_token(self):
        config = {
            "VISION_API_KEY": "test-vision-api-key-0123456789",
        }
        adapter = UnifiedVisionAdapter(config)
        assert "notsobot" not in adapter.providers

    def test_notsobot_auto_promoted_in_order(self):
        config = {
            "NOTSOBOT_API_TOKEN": "test-notsobot-token-0123456789",
            "VISION_API_KEY": "test-vision-api-key-0123456789",
        }
        adapter = UnifiedVisionAdapter(config)
        assert "notsobot" in adapter.providers
        # Check that _has_valid_credentials works for notsobot
        assert adapter._has_valid_credentials("notsobot") is True

    def test_notsobot_credentials_check(self):
        config = {
            "NOTSOBOT_API_TOKEN": "test-notsobot-token-0123456789",
        }
        adapter = UnifiedVisionAdapter(config)
        assert adapter._has_valid_credentials("notsobot") is True

    def test_notsobot_credentials_missing(self):
        config = {}
        adapter = UnifiedVisionAdapter(config)
        assert adapter._has_valid_credentials("notsobot") is False


# --- Model names (verified against notsobot.ts MLDiffusionModels enum) --------


class TestNotSoBotModelNames:
    def test_text_to_image_model_is_flux_klein(self):
        plugin = _make_plugin()
        assert plugin.model_map[VisionTask.TEXT_TO_IMAGE] == "FLUX_KLEIN"

    def test_image_to_image_model_is_flux_klein(self):
        plugin = _make_plugin()
        assert plugin.model_map[VisionTask.IMAGE_TO_IMAGE] == "FLUX_KLEIN"

    def test_video_generation_model_is_flux_klein(self):
        plugin = _make_plugin()
        assert plugin.model_map[VisionTask.VIDEO_GENERATION] == "FLUX_KLEIN"


# --- Task routing (submit dispatches to the correct endpoint) ----------------


class TestNotSoBotTaskRouting:
    @pytest.mark.asyncio
    async def test_text_to_image_routes_to_imagine(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(200, {"id": "job-img-001"}))
        plugin.session = mock_session
        req = _make_request(task=VisionTask.TEXT_TO_IMAGE, prompt="a sunset")
        job_id = await plugin.submit(req)
        assert job_id == "job-img-001"
        call_args = mock_session.get.call_args
        assert "/utilities/ml/imagine" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_image_to_image_routes_to_edit(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(200, {"id": "job-edit-001"}))
        plugin.session = mock_session
        req = _make_request(task=VisionTask.IMAGE_TO_IMAGE, prompt="make it blue")
        job_id = await plugin.submit(req)
        assert job_id == "job-edit-001"
        call_args = mock_session.post.call_args
        assert "/utilities/ml/edit" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_video_generation_routes_to_imagine_video(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(200, {"id": "job-vid-001"}))
        plugin.session = mock_session
        req = _make_request(task=VisionTask.VIDEO_GENERATION, prompt="animate this")
        job_id = await plugin.submit(req)
        assert job_id == "job-vid-001"
        call_args = mock_session.get.call_args
        assert "/utilities/ml/imagine/video" in call_args[0][0]


# --- Query param building (imagine/video use GET query params) ---------------


class TestNotSoBotQueryParams:
    def test_build_query_params_basic(self):
        plugin = _make_plugin()
        req = _make_request(task=VisionTask.TEXT_TO_IMAGE, prompt="a cat")
        params = plugin._build_query_params(req)
        assert params["query"] == "a cat"
        assert params["model"] == "FLUX_KLEIN"
        assert params["safe"] == "true"
        assert params["seed"] == "42"
        assert params["steps"] == "20"

    def test_build_query_params_safe_off(self):
        plugin = _make_plugin()
        req = _make_request()
        req.safety_mode = "off"
        params = plugin._build_query_params(req)
        assert params["safe"] == "false"

    def test_build_query_params_no_optional(self):
        plugin = _make_plugin()
        req = NormalizedRequest(
            task=VisionTask.TEXT_TO_IMAGE,
            prompt="test",
            seed=None,
            steps=None,
        )
        params = plugin._build_query_params(req)
        assert "seed" not in params
        assert "steps" not in params
        assert params["query"] == "test"


# --- Header: x-server-owner-id ----------------------------------------------


class TestNotSoBotOwnerHeader:
    def test_build_headers_with_guild_owner_id(self):
        plugin = _make_plugin()
        req = _make_request()
        req.guild_id = "777"
        req.guild_owner_id = "12345"
        headers = plugin._build_headers(req)
        assert headers["x-server-owner-id"] == "12345"

    def test_build_headers_without_guild_owner_id(self):
        plugin = _make_plugin()
        req = _make_request()
        req.guild_id = "777"
        headers = plugin._build_headers(req)
        assert "x-server-owner-id" not in headers


# --- fetch_result: reads storage.urls (canonical) ----------------------------


class TestNotSoBotFetchResult:
    @pytest.mark.asyncio
    async def test_fetch_result_from_storage_urls(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=_mock_response(
                200,
                {
                    "result": {
                        "response": {
                            "storage": {
                                "urls": {
                                    "default": "https://cdn.notsobot.com/out.png",
                                    "thumbnail": "https://cdn.notsobot.com/out_thumb.png",
                                }
                            }
                        }
                    }
                },
            )
        )
        plugin.session = mock_session
        result = await plugin.fetch_result("job-123")
        assert "https://cdn.notsobot.com/out.png" in result.assets
        assert "https://cdn.notsobot.com/out_thumb.png" in result.assets

    @pytest.mark.asyncio
    async def test_fetch_result_deduplicates(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=_mock_response(
                200,
                {
                    "result": {
                        "response": {
                            "storage": {
                                "urls": {
                                    "default": "https://cdn.notsobot.com/same.png",
                                }
                            },
                            "urls": ["https://cdn.notsobot.com/same.png"],
                        }
                    }
                },
            )
        )
        plugin.session = mock_session
        result = await plugin.fetch_result("job-123")
        assert result.assets.count("https://cdn.notsobot.com/same.png") == 1

    @pytest.mark.asyncio
    async def test_fetch_result_legacy_urls_fallback(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=_mock_response(
                200,
                {"result": {"response": {"urls": ["https://cdn.notsobot.com/legacy.png"]}}},
            )
        )
        plugin.session = mock_session
        result = await plugin.fetch_result("job-123")
        assert "https://cdn.notsobot.com/legacy.png" in result.assets

    @pytest.mark.asyncio
    async def test_fetch_result_no_assets_raises(self):
        plugin = _make_plugin()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(200, {"result": {"response": {}}}))
        plugin.session = mock_session
        with pytest.raises(VisionError) as exc_info:
            await plugin.fetch_result("job-123")
        assert exc_info.value.error_type == VisionErrorType.PROVIDER_ERROR
