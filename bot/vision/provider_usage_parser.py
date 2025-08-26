"""
Provider usage parser for normalizing provider-specific usage to USD [REH][CMV]

Handles conversion of provider-specific units (credits, GPU seconds, tokens) to Money (USD).
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

from bot.vision.money import Money
from bot.vision.pricing_loader import get_pricing_table
from bot.vision.types import VisionProvider, VisionTask

logger = logging.getLogger(__name__)


class UsageUnit(Enum):
    """Provider-specific usage units [CMV]"""
    USD = "usd"
    CREDITS = "credits"
    GPU_SECONDS = "gpu_seconds"
    TOKENS = "tokens"
    STEPS = "steps"


class ProviderUsageParser:
    """
    Parse and normalize provider usage to USD [REH][PA]
    
    Each provider returns usage in different formats:
    - OpenAI: USD directly
    - Anthropic: USD directly  
    - Novita: Credits
    - Chutes: GPU seconds
    - Replicate: USD (but may change)
    - Together: USD or tokens
    """
    
    def __init__(self):
        """Initialize with pricing table [CA]"""
        self.pricing_table = get_pricing_table()
        
    def parse_usage(
        self,
        provider: VisionProvider,
        task: VisionTask,
        usage_data: Dict[str, Any],
        model: Optional[str] = None
    ) -> Money:
        """
        Parse provider-specific usage data and return normalized USD cost [REH]
        
        Args:
            provider: Vision provider
            task: Vision task type
            usage_data: Raw usage data from provider
            model: Optional model name for pricing lookups
            
        Returns:
            Money: Normalized cost in USD
        """
        try:
            # Route to provider-specific parser
            if provider == VisionProvider.NOVITA:
                return self._parse_novita_usage(usage_data, task, model)
            elif provider == VisionProvider.TOGETHER:
                return self._parse_together_usage(usage_data)
            # Handle other providers generically
            elif provider.value in ["openai", "anthropic", "replicate", "chutes"]:
                return self._parse_generic_usage(usage_data, provider.value)
            else:
                logger.warning(f"Unknown provider for usage parsing: {provider}")
                return Money.zero()
                
        except Exception as e:
            logger.error(
                f"Failed to parse usage for {provider}: {e}, "
                f"usage_data: {usage_data}"
            )
            return Money.zero()
    
    def _parse_openai_usage(self, usage_data: Dict[str, Any]) -> Money:
        """
        Parse OpenAI usage (already in USD) [REH]
        
        OpenAI returns:
        - cost: float (USD)
        - tokens: dict with prompt_tokens, completion_tokens
        """
        # Direct USD cost
        if "cost" in usage_data:
            return Money(usage_data["cost"])
        
        # Token-based calculation (fallback)
        if "tokens" in usage_data:
            tokens = usage_data["tokens"]
            # Rough estimate: $0.01 per 1K prompt, $0.03 per 1K completion
            prompt_cost = (tokens.get("prompt_tokens", 0) / 1000) * 0.01
            completion_cost = (tokens.get("completion_tokens", 0) / 1000) * 0.03
            return Money(prompt_cost + completion_cost)
        
        logger.warning(f"No cost info in OpenAI usage: {usage_data}")
        return Money.zero()
    
    def _parse_anthropic_usage(self, usage_data: Dict[str, Any]) -> Money:
        """
        Parse Anthropic usage (already in USD) [REH]
        
        Anthropic returns:
        - cost: float (USD)
        - usage: dict with input_tokens, output_tokens
        """
        # Direct USD cost
        if "cost" in usage_data:
            return Money(usage_data["cost"])
        
        # Token-based calculation (fallback)
        if "usage" in usage_data:
            usage = usage_data["usage"]
            # Claude 3 pricing: $3 per 1M input, $15 per 1M output
            input_cost = (usage.get("input_tokens", 0) / 1_000_000) * 3.0
            output_cost = (usage.get("output_tokens", 0) / 1_000_000) * 15.0
            return Money(input_cost + output_cost)
        
        logger.warning(f"No cost info in Anthropic usage: {usage_data}")
        return Money.zero()
    
    def _parse_novita_usage(
        self,
        usage_data: Dict[str, Any],
        task: VisionTask,
        model: Optional[str] = None
    ) -> Money:
        """
        Parse Novita usage (credits) and convert to USD [REH]
        
        Novita returns:
        - credits: float (Novita credits consumed)
        - steps: int (inference steps)
        """
        if "credits" in usage_data:
            credits = float(usage_data["credits"])
            # Use pricing table to convert credits to USD
            return self.pricing_table.normalize_provider_usage(
                provider=VisionProvider.NOVITA,
                usage_value=credits,
                usage_unit="credits"
            )
        
        # Fallback: estimate from steps
        if "steps" in usage_data:
            steps = usage_data["steps"]
            # Rough estimate: 0.001 credits per step
            estimated_credits = steps * 0.001
            return self.pricing_table.normalize_provider_usage(
                provider=VisionProvider.NOVITA,
                usage_value=estimated_credits,
                usage_unit="credits"
            )
        
        logger.warning(f"No credits info in Novita usage: {usage_data}")
        return Money.zero()
    
    def _parse_chutes_usage(
        self,
        usage_data: Dict[str, Any],
        task: VisionTask,
        model: Optional[str] = None
    ) -> Money:
        """
        Parse Chutes usage (GPU seconds) and convert to USD [REH]
        
        Chutes returns:
        - gpu_seconds: float (GPU time consumed)
        - compute_time: float (alternative field)
        """
        gpu_seconds = 0.0
        
        if "gpu_seconds" in usage_data:
            gpu_seconds = float(usage_data["gpu_seconds"])
        elif "compute_time" in usage_data:
            gpu_seconds = float(usage_data["compute_time"])
        
        if gpu_seconds > 0:
            # Use pricing table to convert GPU seconds to USD
            return self.pricing_table.normalize_provider_usage(
                provider=VisionProvider.CHUTES,
                usage_value=gpu_seconds,
                usage_unit="gpu_seconds"
            )
        
        logger.warning(f"No GPU seconds info in Chutes usage: {usage_data}")
        return Money.zero()
    
    def _parse_replicate_usage(self, usage_data: Dict[str, Any]) -> Money:
        """
        Parse Replicate usage (usually USD) [REH]
        
        Replicate returns:
        - cost: float (USD)
        - predictions: int (number of predictions)
        """
        # Direct USD cost
        if "cost" in usage_data:
            return Money(usage_data["cost"])
        
        # Prediction-based estimate (fallback)
        if "predictions" in usage_data:
            predictions = usage_data["predictions"]
            # Rough estimate: $0.02 per prediction
            return Money(predictions * 0.02)
        
        logger.warning(f"No cost info in Replicate usage: {usage_data}")
        return Money.zero()
    
    def _parse_together_usage(self, usage_data: Dict[str, Any]) -> Money:
        """
        Parse Together usage (USD or tokens) [REH]
        
        Together returns:
        - cost: float (USD)
        - tokens: int (total tokens)
        """
        # Direct USD cost
        if "cost" in usage_data:
            return Money(usage_data["cost"])
        
        # Token-based calculation (fallback)
        if "tokens" in usage_data:
            tokens = usage_data["tokens"]
            # Together pricing: ~$0.0008 per 1K tokens
            return Money((tokens / 1000) * 0.0008)
        
        logger.warning(f"No cost info in Together usage: {usage_data}")
        return Money.zero()
    
    def _parse_generic_usage(self, usage_data: Dict[str, Any], provider_name: str) -> Money:
        """
        Parse generic provider usage (for providers not in enum) [REH]
        """
        # Direct USD cost
        if "cost" in usage_data:
            return Money(usage_data["cost"])
        
        # Token-based calculation (fallback)
        if "tokens" in usage_data:
            tokens = usage_data.get("tokens", 0)
            if isinstance(tokens, dict):
                # OpenAI style
                prompt_cost = (tokens.get("prompt_tokens", 0) / 1000) * 0.01
                completion_cost = (tokens.get("completion_tokens", 0) / 1000) * 0.03
                return Money(prompt_cost + completion_cost)
            else:
                # Simple token count
                return Money((tokens / 1000) * 0.001)
        
        # GPU seconds (for Chutes-like providers)
        if "gpu_seconds" in usage_data or "compute_time" in usage_data:
            gpu_seconds = usage_data.get("gpu_seconds", usage_data.get("compute_time", 0))
            return Money(gpu_seconds * 0.0006)  # Default GPU rate
        
        logger.warning(f"No cost info in {provider_name} usage: {usage_data}")
        return Money.zero()
    
    def extract_usage_from_response(
        self,
        provider: VisionProvider,
        response: Any
    ) -> Dict[str, Any]:
        """
        Extract usage data from provider response [REH]
        
        Different providers return usage in different places:
        - Some in response.usage
        - Some in response metadata
        - Some in headers
        """
        usage_data = {}
        
        try:
            # Check common locations
            if hasattr(response, 'usage'):
                if isinstance(response.usage, dict):
                    usage_data.update(response.usage)
                else:
                    # Handle object-like usage
                    usage_data['usage'] = response.usage
            
            if hasattr(response, 'metadata'):
                metadata = response.metadata
                if isinstance(metadata, dict):
                    # Look for usage-related fields
                    for key in ['cost', 'credits', 'gpu_seconds', 'compute_time', 'tokens']:
                        if key in metadata:
                            usage_data[key] = metadata[key]
            
            # Provider-specific extraction
            if provider == VisionProvider.NOVITA:
                # Novita may return credits in response body
                if hasattr(response, 'credits_consumed'):
                    usage_data['credits'] = response.credits_consumed
                    
            elif provider == VisionProvider.CHUTES:
                # Chutes may return GPU time in headers or body
                if hasattr(response, 'gpu_time'):
                    usage_data['gpu_seconds'] = response.gpu_time
                    
        except Exception as e:
            logger.error(f"Failed to extract usage from {provider} response: {e}")
        
        return usage_data
    
    def validate_usage_cost(
        self,
        provider: VisionProvider,
        task: VisionTask,
        estimated_cost: Money,
        actual_cost: Money,
        warning_threshold: float = 1.5,
        error_threshold: float = 3.0
    ) -> tuple[bool, Optional[str]]:
        """
        Validate actual cost against estimate [REH][PA]
        
        Returns:
            (is_valid, error_message)
        """
        if actual_cost.is_zero():
            # No actual cost reported - might be OK for some providers
            logger.debug(f"No actual cost reported by {provider}")
            return True, None
        
        if estimated_cost.is_zero():
            # No estimate but have actual cost - suspicious
            logger.warning(
                f"No estimate but actual cost {actual_cost} from {provider}"
            )
            return True, None  # Allow but log
        
        # Calculate discrepancy ratio
        ratio = actual_cost.to_float() / estimated_cost.to_float()
        
        if ratio > error_threshold:
            error_msg = (
                f"Actual cost {actual_cost} is {ratio:.1f}x higher than "
                f"estimate {estimated_cost} for {provider}/{task}"
            )
            logger.error(error_msg)
            return False, error_msg
            
        elif ratio > warning_threshold:
            logger.warning(
                f"Actual cost {actual_cost} is {ratio:.1f}x higher than "
                f"estimate {estimated_cost} for {provider}/{task}"
            )
        
        return True, None
