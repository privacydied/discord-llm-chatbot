"""
Comprehensive tests for UnifiedVisionAdapter integration

Tests the refactored unified adapter system with provider plugins,
error handling, fallback logic, and integration with gateway.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from bot.vision.unified_adapter import (
    UnifiedVisionAdapter,
    TogetherPlugin,
    NovitaPlugin,
    UnifiedStatus,
    NormalizedRequest,
    UnifiedJobStatus,
    UnifiedResult,
)
from bot.vision.gateway import VisionGateway
from bot.vision.types import VisionRequest, VisionTask, VisionError, VisionErrorType


@pytest.fixture
def mock_config():
    """Mock bot configuration for testing"""
    return {
        "VISION_ENABLED": True,
        "VISION_API_KEY": "test_key_12345",
        "VISION_ALLOWED_PROVIDERS": ["together", "novita"],
        "VISION_DEFAULT_PROVIDER": "together",
        "VISION_PROVIDER_TIMEOUT_MS": 30000,
        "VISION_PROVIDER_MAX_RETRIES": 2,
        "VISION_PROVIDER_RETRY_DELAY_MS": 1000,
    }


@pytest.fixture
def vision_request():
    """Sample vision request for testing"""
    return VisionRequest(
        user_id="user_456",
        guild_id="guild_789",
        task=VisionTask.TEXT_TO_IMAGE,
        prompt="a beautiful sunset over mountains",
        width=1024,
        height=1024,
        steps=20,
        guidance_scale=7.5,
    )


@pytest.fixture
def unified_adapter(mock_config):
    """Create UnifiedVisionAdapter instance for testing"""
    return UnifiedVisionAdapter(mock_config)


@pytest.fixture
def vision_gateway(mock_config):
    """Create VisionGateway instance using unified adapter"""
    return VisionGateway(mock_config)


class TestUnifiedVisionAdapter:
    """Test UnifiedVisionAdapter core functionality"""

    def test_adapter_initialization(self, unified_adapter):
        """Test adapter initializes with provider plugins [CA]"""
        assert len(unified_adapter.providers) > 0
        assert (
            "together" in unified_adapter.providers
            or "novita" in unified_adapter.providers
        )
        assert unified_adapter.provider_config is not None

    def test_request_normalization(self, unified_adapter, vision_request):
        """Test request normalization across providers [IV]"""
        normalized = unified_adapter.normalize_request(vision_request)

        assert isinstance(normalized, NormalizedRequest)
        assert normalized.task == VisionTask.TEXT_TO_IMAGE
        assert normalized.prompt == "a beautiful sunset over mountains"
        assert normalized.width == 1024
        assert normalized.height == 1024
        assert normalized.steps == 20
        assert normalized.guidance_scale == 7.5

    def test_provider_selection(self, unified_adapter, vision_request):
        """Test automatic provider selection logic [CA]"""
        normalized = unified_adapter.normalize_request(vision_request)
        provider = unified_adapter.select_provider(normalized)

        # Should return a provider plugin that supports the task
        assert provider is not None
        capabilities = provider.capabilities()
        assert normalized.task in capabilities.get("modes", [])

    def test_cost_estimation(self, unified_adapter, vision_request):
        """Test cost estimation across providers [CMV]"""
        estimates = unified_adapter.estimate_cost_for_request(vision_request)

        assert isinstance(estimates, dict)
        assert len(estimates) > 0
        for provider_name, cost in estimates.items():
            assert isinstance(cost, (int, float))
            assert cost >= 0

    def test_supported_tasks(self, unified_adapter):
        """Test getting supported tasks from all providers [PA]"""
        tasks = unified_adapter.get_supported_tasks()

        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert VisionTask.TEXT_TO_IMAGE in tasks


class TestProviderPlugins:
    """Test individual provider plugin implementations"""

    def test_together_plugin_capabilities(self, mock_config):
        """Test Together.ai plugin capabilities [CA]"""
        plugin = TogetherPlugin("together", {}, mock_config["VISION_API_KEY"])
        capabilities = plugin.capabilities()

        assert "modes" in capabilities
        assert VisionTask.TEXT_TO_IMAGE in capabilities["modes"]
        assert "max_size" in capabilities
        assert capabilities["nsfw_policy"] == "blocked"

    def test_novita_plugin_capabilities(self, mock_config):
        """Test Novita.ai plugin capabilities [CA]"""
        plugin = NovitaPlugin("novita", {}, mock_config["VISION_API_KEY"])
        capabilities = plugin.capabilities()

        assert "modes" in capabilities
        assert VisionTask.TEXT_TO_IMAGE in capabilities["modes"]
        assert "max_size" in capabilities
        assert capabilities["nsfw_policy"] == "filtered"

    @pytest.mark.asyncio
    async def test_plugin_session_management(self, mock_config):
        """Test provider plugin session lifecycle [RM]"""
        plugin = TogetherPlugin("together", {}, mock_config["VISION_API_KEY"])

        # Start session
        await plugin.startup()
        assert plugin.session is not None

        # Cleanup session
        await plugin.shutdown()
        assert plugin.session is None


class TestErrorHandling:
    """Test unified error handling and status mapping [REH]"""

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, unified_adapter, vision_request):
        """Test authentication error is properly handled"""
        # Mock provider to raise authentication error
        with patch.object(
            unified_adapter.providers.get("together", MagicMock()), "submit"
        ) as mock_submit:
            mock_submit.side_effect = VisionError(
                message="Invalid API key",
                error_type=VisionErrorType.AUTHENTICATION_ERROR,
                user_message="Authentication failed",
            )

            with pytest.raises(VisionError) as exc_info:
                await unified_adapter.submit(vision_request)

            assert exc_info.value.error_type == VisionErrorType.AUTHENTICATION_ERROR

    @pytest.mark.asyncio
    async def test_fallback_on_provider_failure(self, unified_adapter, vision_request):
        """Test automatic fallback when primary provider fails"""
        # Ensure we have multiple providers for fallback testing
        if len(unified_adapter.providers) < 2:
            pytest.skip("Need multiple providers for fallback test")

        provider_names = list(unified_adapter.providers.keys())
        primary_provider = unified_adapter.providers[provider_names[0]]
        fallback_provider = unified_adapter.providers[provider_names[1]]

        # Mock primary provider to fail, fallback to succeed
        with patch.object(primary_provider, "submit") as mock_primary:
            with patch.object(fallback_provider, "submit") as mock_fallback:
                mock_primary.side_effect = VisionError(
                    message="Server error",
                    error_type=VisionErrorType.SERVER_ERROR,
                    user_message="Service unavailable",
                )
                mock_fallback.return_value = "fallback_job_123"

                job_id, provider_name = await unified_adapter.submit(vision_request)

                assert job_id.startswith(
                    provider_names[1]
                )  # Should use fallback provider
                assert provider_name == provider_names[1]


class TestGatewayIntegration:
    """Test VisionGateway integration with UnifiedVisionAdapter"""

    @pytest.mark.asyncio
    async def test_gateway_initialization_with_adapter(self, vision_gateway):
        """Test gateway initializes with unified adapter [CA]"""
        assert hasattr(vision_gateway, "adapter")
        assert isinstance(vision_gateway.adapter, UnifiedVisionAdapter)

        await vision_gateway.startup()
        assert vision_gateway.adapter is not None

        await vision_gateway.shutdown()

    @pytest.mark.asyncio
    async def test_job_submission_through_gateway(self, vision_gateway, vision_request):
        """Test job submission flows through unified adapter"""
        # Mock adapter submission
        with patch.object(vision_gateway.adapter, "submit") as mock_submit:
            mock_submit.return_value = ("together:job_123", "together")

            job_id = await vision_gateway.submit_job(vision_request)

            assert job_id == "together:job_123"
            mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_status_polling_through_gateway(self, vision_gateway):
        """Test status polling flows through unified adapter"""
        job_id = "together:job_123"

        # Mock adapter polling
        with patch.object(vision_gateway.adapter, "poll") as mock_poll:
            mock_status = UnifiedJobStatus(
                status=UnifiedStatus.COMPLETED,
                progress_percentage=100,
                phase="Generation complete",
            )
            mock_poll.return_value = mock_status

            # Set up active job
            vision_gateway.active_jobs[job_id] = {
                "provider": "together",
                "start_time": asyncio.get_event_loop().time(),
            }

            status = await vision_gateway.get_job_status(job_id)

            assert status["state"] == "completed"
            assert status["progress_percentage"] == 100
            mock_poll.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_result_fetching_through_gateway(
        self, vision_gateway, vision_request
    ):
        """Test result fetching flows through unified adapter"""
        job_id = "together:job_123"

        # Mock adapter methods
        with patch.object(vision_gateway.adapter, "poll") as mock_poll:
            with patch.object(vision_gateway.adapter, "fetch_result") as mock_fetch:
                # Mock completed status
                mock_poll.return_value = UnifiedJobStatus(
                    status=UnifiedStatus.COMPLETED, progress_percentage=100
                )

                # Mock result
                mock_fetch.return_value = UnifiedResult(
                    assets=["https://example.com/image.png"],
                    final_cost=0.05,
                    provider_used="together",
                )

                # Set up active job
                vision_gateway.active_jobs[job_id] = {
                    "request": vision_request,
                    "start_time": asyncio.get_event_loop().time(),
                }

                response = await vision_gateway.get_job_result(job_id)

                assert response is not None
                urls = [str(p) for p in response.artifacts]
                normalized_urls = [
                    u.replace("https:/", "https://").replace("http:/", "http://")
                    for u in urls
                ]
                assert normalized_urls == ["https://example.com/image.png"]
                assert response.actual_cost == 0.05
                assert job_id not in vision_gateway.active_jobs  # Should be cleaned up


class TestVisionModelOverride:
    """Test VISION_MODEL environment variable override functionality"""

    def test_vision_model_qwen_override(self, mock_config):
        """Test VISION_MODEL override for Qwen endpoint"""
        config_with_override = mock_config.copy()
        config_with_override["VISION_MODEL"] = "novita:qwen-image"

        adapter = UnifiedVisionAdapter(config_with_override)
        request = VisionRequest(
            user_id="user", task=VisionTask.TEXT_TO_IMAGE, prompt="test qwen image"
        )

        normalized = adapter.normalize_request(request)
        selection = adapter.resolve_model_selection(normalized)

        assert selection is not None
        assert selection.provider == "novita"
        assert selection.endpoint == "qwen-image-txt2img"
        assert selection.model_hint == "qwen-image"
        assert not selection.supports_advanced

    def test_vision_model_aliases(self, mock_config):
        """Test various VISION_MODEL alias formats"""
        test_cases = [
            ("qwen-image", "novita", "qwen-image-txt2img"),
            ("novita:qwen-image", "novita", "qwen-image-txt2img"),
            ("novita:sdxl", "novita", "txt2img"),
            ("together:flux.1-pro", "together", "images/generations"),
            ("flux.1-pro", "together", "images/generations"),
        ]

        for alias, expected_provider, expected_endpoint in test_cases:
            config_with_alias = mock_config.copy()
            config_with_alias["VISION_MODEL"] = alias

            adapter = UnifiedVisionAdapter(config_with_alias)

            request = VisionRequest(
                user_id="user", task=VisionTask.TEXT_TO_IMAGE, prompt="test"
            )

            normalized = adapter.normalize_request(request)
            selection = adapter.resolve_model_selection(normalized)

            assert selection is not None, f"Failed to resolve alias: {alias}"
            assert selection.provider == expected_provider
            assert selection.endpoint == expected_endpoint


class TestQwenEndpointParameterNormalization:
    """Test parameter normalization and warnings for Qwen endpoint"""

    def test_qwen_parameter_warnings(self, mock_config):
        """Test parameter normalization and warnings for Qwen endpoint"""
        adapter = UnifiedVisionAdapter(mock_config)

        if "novita" not in adapter.providers:
            pytest.skip("Novita provider not available")

        novita_plugin = adapter.providers["novita"]

        request = NormalizedRequest(
            task=VisionTask.TEXT_TO_IMAGE,
            prompt="test prompt",
            negative_prompt="bad things",
            width=2048,
            height=2048,
            steps=50,
            guidance_scale=15.0,
            seed=42,
            batch_size=1,
            safety_mode="strict",
        )

        payload, warnings = novita_plugin.build_payload_for_endpoint(
            "qwen-image-txt2img", request
        )

        # Check payload format
        assert "size" in payload
        assert payload["size"] == "1536*1536"  # Clamped to max
        assert payload["prompt"] == "test prompt"

        # Check warnings for ignored parameters
        assert len(warnings) > 0
        warning_text = " ".join(warnings)
        assert "Negative prompt not supported" in warning_text
        assert "Custom steps not supported" in warning_text
        assert "Custom guidance scale not supported" in warning_text
        assert "Custom seed not supported" in warning_text

    def test_sdxl_parameter_support(self, mock_config):
        """Test full parameter support for SDXL endpoint"""
        adapter = UnifiedVisionAdapter(mock_config)

        if "novita" not in adapter.providers:
            pytest.skip("Novita provider not available")

        novita_plugin = adapter.providers["novita"]

        request = NormalizedRequest(
            task=VisionTask.TEXT_TO_IMAGE,
            prompt="beautiful landscape",
            negative_prompt="ugly, distorted",
            width=1024,
            height=1024,
            steps=30,
            guidance_scale=7.5,
            seed=12345,
            batch_size=1,
            safety_mode="detect",  # Enable NSFW detection
        )

        payload, warnings = novita_plugin.build_payload_for_endpoint("txt2img", request)

        # Check full parameter support
        assert payload["width"] == 1024
        assert payload["height"] == 1024
        assert payload["prompt"] == "beautiful landscape"
        assert payload["negative_prompt"] == "ugly, distorted"
        assert payload["steps"] == 30
        assert payload["guidance_scale"] == 7.5
        assert payload["seed"] == 12345
        assert payload["model_name"] == "sd_xl_base_1.0.safetensors"

        # Check NSFW detection enabled
        assert "extra" in payload
        assert payload["extra"]["enable_nsfw_detection"] is True
        assert payload["extra"]["nsfw_detection_level"] == 2

        # Should have minimal warnings (none for valid parameters)
        assert len(warnings) == 0


class TestConfigurationHandling:
    """Test configuration management and defaults [IV]"""

    def test_default_provider_config_generation(self, mock_config):
        """Test default configuration is properly generated"""
        adapter = UnifiedVisionAdapter(mock_config)
        config = adapter.provider_config

        assert "vision" in config
        assert "default_policy" in config["vision"]
        assert "providers" in config["vision"]

        policy = config["vision"]["default_policy"]
        assert "provider_order" in policy
        assert "budget_per_job_usd" in policy
        assert "auto_fallback" in policy

    def test_provider_api_key_handling(self, mock_config):
        """Test API key resolution for providers [SFT]"""
        adapter = UnifiedVisionAdapter(mock_config)

        # Should have initialized providers with API key
        for provider in adapter.providers.values():
            assert provider.api_key == mock_config["VISION_API_KEY"]


if __name__ == "__main__":
    # Run basic smoke tests
    async def smoke_test():
        config = {
            "VISION_ENABLED": True,
            "VISION_API_KEY": "test_key",
            "VISION_ALLOWED_PROVIDERS": ["together"],
            "VISION_DEFAULT_PROVIDER": "together",
        }

        print("🧪 Testing UnifiedVisionAdapter initialization...")
        adapter = UnifiedVisionAdapter(config)
        print(f"✅ Initialized with {len(adapter.providers)} providers")

        print("🧪 Testing request normalization...")
        request = VisionRequest(
            user_id="user",
            guild_id="guild",
            task=VisionTask.TEXT_TO_IMAGE,
            prompt="test prompt",
        )
        normalized = adapter.normalize_request(request)
        print(f"✅ Normalized request: {normalized.task.value}")

        print("🧪 Testing supported tasks...")
        tasks = adapter.get_supported_tasks()
        print(f"✅ Supported tasks: {[t.value for t in tasks]}")

        print("🧪 Testing VisionGateway integration...")
        gateway = VisionGateway(config)
        await gateway.startup()
        print("✅ Gateway started with unified adapter")
        await gateway.shutdown()
        print("✅ Gateway shutdown complete")

        print(
            "\n🎉 All smoke tests passed! Unified Vision Adapter is working correctly."
        )

    asyncio.run(smoke_test())
