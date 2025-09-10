"""
Tests for ProviderUsageParser [REH][PA][CMV]
"""

import pytest
from unittest.mock import Mock, patch

from bot.vision.provider_usage_parser import ProviderUsageParser
from bot.vision.money import Money
from bot.vision.types import VisionProvider, VisionTask


class TestProviderUsageParser:
    """Test provider usage parsing and normalization"""

    @pytest.fixture
    def parser(self):
        """Create parser instance"""
        return ProviderUsageParser()

    def test_parse_generic_openai_usage_direct_cost(self, parser):
        """Test parsing OpenAI-like usage with direct cost via generic handler"""
        usage_data = {"cost": 0.15}

        # Create a mock provider with value "openai"
        mock_provider = Mock()
        mock_provider.value = "openai"

        result = parser.parse_usage(
            provider=mock_provider, task=VisionTask.TEXT_TO_IMAGE, usage_data=usage_data
        )

        assert result.to_float() == pytest.approx(0.15, rel=1e-4)

    def test_parse_generic_openai_usage_token_fallback(self, parser):
        """Test parsing OpenAI-like usage with token fallback via generic handler"""
        usage_data = {"tokens": {"prompt_tokens": 1000, "completion_tokens": 500}}

        # Create a mock provider with value "openai"
        mock_provider = Mock()
        mock_provider.value = "openai"

        result = parser.parse_usage(
            provider=mock_provider, task=VisionTask.TEXT_TO_IMAGE, usage_data=usage_data
        )

        # (1000/1000)*0.01 + (500/1000)*0.03 = 0.01 + 0.015 = 0.025
        assert result.to_float() == pytest.approx(0.025, rel=1e-4)

    def test_parse_generic_anthropic_usage_direct_cost(self, parser):
        """Test parsing Anthropic-like usage with direct cost via generic handler"""
        usage_data = {"cost": 0.25}

        # Create a mock provider with value "anthropic"
        mock_provider = Mock()
        mock_provider.value = "anthropic"

        result = parser.parse_usage(
            provider=mock_provider,
            task=VisionTask.IMAGE_TO_IMAGE,
            usage_data=usage_data,
        )

        assert result.to_float() == pytest.approx(0.25, rel=1e-4)

    def test_parse_generic_anthropic_usage_token_fallback(self, parser):
        """Test parsing Anthropic-like usage with token fallback"""
        usage_data = {"tokens": {"prompt_tokens": 1000, "completion_tokens": 500}}

        # Create a mock provider with value "anthropic"
        mock_provider = Mock()
        mock_provider.value = "anthropic"

        result = parser.parse_usage(
            provider=mock_provider,
            task=VisionTask.IMAGE_TO_IMAGE,
            usage_data=usage_data,
        )

        # Generic handler: (1000/1000)*0.01 + (500/1000)*0.03 = 0.025
        assert result.to_float() == pytest.approx(0.025, rel=1e-4)

    def test_parse_novita_usage_credits(self, parser):
        """Test parsing Novita usage with credits"""
        usage_data = {"credits": 5.0}

        with patch.object(
            parser.pricing_table, "normalize_provider_usage"
        ) as mock_norm:
            mock_norm.return_value = Money(0.01)  # 5 credits * $0.002/credit

            result = parser.parse_usage(
                provider=VisionProvider.NOVITA,
                task=VisionTask.TEXT_TO_IMAGE,
                usage_data=usage_data,
                model="novita-v1",
            )

            assert result.to_float() == pytest.approx(0.01, rel=1e-4)
            mock_norm.assert_called_once_with(
                provider=VisionProvider.NOVITA, usage_value=5.0, usage_unit="credits"
            )

    def test_parse_novita_usage_steps_fallback(self, parser):
        """Test parsing Novita usage with steps fallback"""
        usage_data = {"steps": 100}

        with patch.object(
            parser.pricing_table, "normalize_provider_usage"
        ) as mock_norm:
            mock_norm.return_value = Money(
                0.0002
            )  # 100 steps * 0.001 credits/step * $0.002/credit

            result = parser.parse_usage(
                provider=VisionProvider.NOVITA,
                task=VisionTask.TEXT_TO_IMAGE,
                usage_data=usage_data,
            )

            assert result.to_float() == pytest.approx(0.0002, rel=1e-4)

    def test_parse_generic_chutes_usage_gpu_seconds(self, parser):
        """Test parsing Chutes-like usage with GPU seconds via generic handler"""
        usage_data = {"gpu_seconds": 120}

        # Create a mock provider with value "chutes"
        mock_provider = Mock()
        mock_provider.value = "chutes"

        result = parser.parse_usage(
            provider=mock_provider,
            task=VisionTask.TEXT_TO_VIDEO,
            usage_data=usage_data,
            model="chutes-v1",
        )

        # Generic handler: 120 * 0.0006 = 0.072
        assert result.to_float() == pytest.approx(0.072, rel=1e-4)

    def test_parse_generic_chutes_usage_compute_time(self, parser):
        """Test parsing Chutes-like usage with compute_time via generic handler"""
        usage_data = {"compute_time": 60}

        # Create a mock provider with value "chutes"
        mock_provider = Mock()
        mock_provider.value = "chutes"

        result = parser.parse_usage(
            provider=mock_provider,
            task=VisionTask.IMAGE_TO_VIDEO,
            usage_data=usage_data,
        )

        # Generic handler: 60 * 0.0006 = 0.036
        assert result.to_float() == pytest.approx(0.036, rel=1e-4)

    def test_parse_generic_replicate_usage_direct_cost(self, parser):
        """Test parsing Replicate-like usage with direct cost via generic handler"""
        usage_data = {"cost": 0.50}

        # Create a mock provider with value "replicate"
        mock_provider = Mock()
        mock_provider.value = "replicate"

        result = parser.parse_usage(
            provider=mock_provider,
            task=VisionTask.VIDEO_GENERATION,
            usage_data=usage_data,
        )

        assert result.to_float() == pytest.approx(0.50, rel=1e-4)

    def test_parse_generic_replicate_usage_no_cost(self, parser):
        """Test parsing Replicate-like usage with no cost info"""
        usage_data = {"predictions": 5}

        # Create a mock provider with value "replicate"
        mock_provider = Mock()
        mock_provider.value = "replicate"

        result = parser.parse_usage(
            provider=mock_provider,
            task=VisionTask.VIDEO_GENERATION,
            usage_data=usage_data,
        )

        # Generic handler doesn't handle predictions, returns zero
        assert result.to_float() == pytest.approx(0.0, rel=1e-4)

    def test_parse_together_usage_direct_cost(self, parser):
        """Test parsing Together usage with direct cost"""
        usage_data = {"cost": 0.08}

        result = parser.parse_usage(
            provider=VisionProvider.TOGETHER,
            task=VisionTask.TEXT_TO_IMAGE,
            usage_data=usage_data,
        )

        assert result.to_float() == pytest.approx(0.08, rel=1e-4)

    def test_parse_together_usage_tokens_fallback(self, parser):
        """Test parsing Together usage with tokens fallback"""
        usage_data = {"tokens": 10000}

        result = parser.parse_usage(
            provider=VisionProvider.TOGETHER,
            task=VisionTask.TEXT_TO_IMAGE,
            usage_data=usage_data,
        )

        # (10000/1000) * 0.0008 = 0.008
        assert result.to_float() == pytest.approx(0.008, rel=1e-4)

    def test_parse_unknown_provider(self, parser):
        """Test parsing unknown provider returns zero"""
        # Create a mock provider with an unknown value
        mock_provider = Mock()
        mock_provider.value = "unknown_provider"

        usage_data = {"cost": 0.10}

        result = parser.parse_usage(
            provider=mock_provider, task=VisionTask.TEXT_TO_IMAGE, usage_data=usage_data
        )

        assert result.to_float() == pytest.approx(0.0, rel=1e-4)

    def test_parse_empty_usage_data(self, parser):
        """Test parsing empty usage data returns zero"""
        result = parser.parse_usage(
            provider=VisionProvider.TOGETHER,
            task=VisionTask.TEXT_TO_IMAGE,
            usage_data={},
        )

        assert result.to_float() == pytest.approx(0.0, rel=1e-4)

    def test_parse_malformed_usage_data(self, parser):
        """Test parsing malformed usage data handles gracefully"""
        usage_data = {"invalid_key": "invalid_value"}

        result = parser.parse_usage(
            provider=VisionProvider.NOVITA,
            task=VisionTask.TEXT_TO_IMAGE,
            usage_data=usage_data,
        )

        assert result.to_float() == pytest.approx(0.0, rel=1e-4)

    def test_extract_usage_from_response(self, parser):
        """Test extracting usage data from provider response"""
        # Together-style response
        response = Mock()
        response.usage = {"cost": 0.15}

        usage_data = parser.extract_usage_from_response(
            provider=VisionProvider.TOGETHER, response=response
        )

        assert usage_data["cost"] == 0.15

    def test_extract_usage_from_metadata(self, parser):
        """Test extracting usage from response metadata"""
        response = Mock()
        response.metadata = {"cost": 0.015, "credits": 5.0, "gpu_seconds": 10.2}

        # Mock hasattr to return False for 'usage' but True for 'metadata'
        with patch(
            "builtins.hasattr", side_effect=lambda obj, attr: attr == "metadata"
        ):
            usage_data = parser.extract_usage_from_response(
                provider=VisionProvider.NOVITA, response=response
            )

            assert usage_data["cost"] == 0.015
            assert usage_data["credits"] == 5.0
            assert usage_data["gpu_seconds"] == 10.2

    def test_extract_usage_provider_specific(self, parser):
        """Test provider-specific usage extraction"""
        # Novita response
        response = Mock()
        response.credits_consumed = 3.5

        # Mock hasattr to return True for 'credits_consumed'
        with patch(
            "builtins.hasattr", side_effect=lambda obj, attr: attr == "credits_consumed"
        ):
            usage_data = parser.extract_usage_from_response(
                provider=VisionProvider.NOVITA, response=response
            )

            assert usage_data["credits"] == 3.5

    def test_validate_usage_cost_valid(self, parser):
        """Test validating usage cost within threshold"""
        estimated = Money(1.00)
        actual = Money(1.05)

        is_valid, error_msg = parser.validate_usage_cost(
            provider=VisionProvider.TOGETHER,
            task=VisionTask.TEXT_TO_IMAGE,
            estimated_cost=estimated,
            actual_cost=actual,
            warning_threshold=1.5,
            error_threshold=3.0,
        )

        assert is_valid
        assert error_msg is None

    def test_validate_usage_cost_warning(self, parser):
        """Test validating usage cost triggers warning"""
        estimated = Money(1.00)
        actual = Money(1.30)

        with patch("logging.Logger.warning") as mock_warning:
            is_valid, error_msg = parser.validate_usage_cost(
                provider=VisionProvider.TOGETHER,
                task=VisionTask.TEXT_TO_IMAGE,
                estimated_cost=estimated,
                actual_cost=actual,
                warning_threshold=1.25,  # Lower threshold to trigger warning
                error_threshold=3.0,
            )

            assert is_valid  # Still valid, just warning
            assert error_msg is None
            mock_warning.assert_called()

    def test_validate_usage_cost_error(self, parser):
        """Test validating usage cost triggers error"""
        estimated = Money(1.00)
        actual = Money(2.50)

        is_valid, error_msg = parser.validate_usage_cost(
            provider=VisionProvider.TOGETHER,
            task=VisionTask.TEXT_TO_IMAGE,
            estimated_cost=estimated,
            actual_cost=actual,
            warning_threshold=1.5,
            error_threshold=2.0,  # Lower threshold to trigger error
        )

        assert not is_valid
        assert "2.5x higher" in error_msg

    def test_validate_usage_cost_zero_actual(self, parser):
        """Test validating with zero actual cost"""
        estimated = Money(1.00)
        actual = Money.zero()

        is_valid, error_msg = parser.validate_usage_cost(
            provider=VisionProvider.NOVITA,
            task=VisionTask.TEXT_TO_IMAGE,
            estimated_cost=estimated,
            actual_cost=actual,
        )

        assert is_valid  # Zero actual is OK
        assert error_msg is None

    def test_validate_usage_cost_zero_estimate(self, parser):
        """Test validating with zero estimate"""
        estimated = Money.zero()
        actual = Money(0.50)

        with patch("logging.Logger.warning") as mock_warning:
            is_valid, error_msg = parser.validate_usage_cost(
                provider=VisionProvider.TOGETHER,
                task=VisionTask.TEXT_TO_IMAGE,
                estimated_cost=estimated,
                actual_cost=actual,
            )

            assert is_valid  # Allow but log warning
            assert error_msg is None
            mock_warning.assert_called()
