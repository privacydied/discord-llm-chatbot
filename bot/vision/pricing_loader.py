"""
Pricing table loader and cost estimator [CA][PA]

Loads canonical pricing from JSON and provides deterministic cost estimation
for vision generation tasks across all providers.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from decimal import Decimal
import json
import os

from bot.vision.money import Money
from bot.vision.types import VisionTask, VisionProvider
from bot.util.logging import get_logger

logger = get_logger(__name__)


class PricingTable:
    """
    Canonical pricing table for vision generation costs.

    Provides deterministic cost estimation based on provider, task type,
    model, and generation parameters.
    """

    def __init__(self, pricing_file: Optional[Path] = None):
        """
        Initialize pricing table from JSON file.

        Args:
            pricing_file: Path to pricing JSON file (defaults to vision_pricing.json)
        """
        if pricing_file is None:
            pricing_file = Path(__file__).parent / "pricing" / "vision_pricing.json"

        self.pricing_file = pricing_file
        self.pricing_data: Dict[str, Any] = {}
        self.load_pricing()

    def load_pricing(self) -> None:
        """Load pricing data from JSON file [REH]"""
        try:
            with open(self.pricing_file, "r") as f:
                self.pricing_data = json.load(f)
            providers = list(self.pricing_data.get("providers", {}).keys())
            logger.info(
                f"Loaded pricing table v{self.pricing_data.get('version', 'unknown')}",
                extra={
                    "subsys": "vision",
                    "event": "pricing.loaded",
                    "detail": f"providers={','.join(providers)}",
                },
            )
        except FileNotFoundError:
            logger.error(f"Pricing file not found: {self.pricing_file}")
            self.pricing_data = self._get_fallback_pricing()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid pricing JSON: {e}")
            self.pricing_data = self._get_fallback_pricing()

    def _get_fallback_pricing(self) -> Dict[str, Any]:
        """Get fallback pricing if file cannot be loaded [REH]"""
        logger.warning("Using fallback pricing data")
        return {
            "version": "fallback",
            "currency": "USD",
            "providers": {},
            "default_estimates": {
                "text_to_image": 0.02,
                "image_to_image": 0.025,
                "text_to_video": 0.30,
                "image_to_video": 0.36,
                "unknown": 0.05,
            },
            "estimation_config": {
                "safety_factor": 1.2,
                "max_discrepancy_ratio": 5.0,
                "warning_discrepancy_ratio": 2.0,
            },
        }

    def estimate_cost(
        self,
        provider: VisionProvider,
        task: VisionTask,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        duration_seconds: float = 4.0,
        model: Optional[str] = None,
    ) -> Money:
        """
        Estimate cost for a vision generation task.

        Args:
            provider: Vision provider to use
            task: Type of vision task
            width: Image width in pixels
            height: Image height in pixels
            num_images: Number of images to generate
            duration_seconds: Video duration in seconds
            model: Specific model name (optional)

        Returns:
            Estimated cost as Money object
        """
        # Check for environment variable overrides first
        provider_name = provider.value.lower()
        task_name = self._task_to_pricing_key(task)

        env_key = f"PRICING_{provider_name.upper()}_{task_name.upper()}_PER_IMAGE_USD"
        env_override = os.getenv(env_key)
        if env_override:
            try:
                per_image_cost = Money(env_override)
                return per_image_cost * num_images
            except Exception as e:
                logger.warning(f"Invalid env override {env_key}={env_override}: {e}")

        provider_config = self.pricing_data.get("providers", {}).get(provider_name)

        if not provider_config:
            # Use default estimates if provider not configured
            logger.warning(f"No pricing config for provider: {provider_name}")
            return self._get_default_estimate(task, num_images, duration_seconds)

        task_pricing = provider_config.get("base_prices", {}).get(task_name)

        if not task_pricing:
            # Use default estimates if task not configured for provider
            logger.warning(f"No pricing for {task_name} on {provider_name}")
            return self._get_default_estimate(task, num_images, duration_seconds)

        # Calculate base cost
        if task in [
            VisionTask.TEXT_TO_VIDEO,
            VisionTask.IMAGE_TO_VIDEO,
            VisionTask.VIDEO_GENERATION,
        ]:
            # Video tasks: base + per_second * duration
            base = Money(task_pricing.get("base_cost", 0.10))
            per_second = Money(task_pricing.get("per_second", 0.05))
            cost = base + (per_second * duration_seconds)
        else:
            # Image tasks: per_image * num_images * size_multiplier * model_multiplier
            per_image = Money(task_pricing.get("per_image", 0.02))

            # Apply size multiplier
            size_key = self._get_size_key(width, height)
            size_multipliers = task_pricing.get("size_multipliers", {})
            size_multiplier = Decimal(str(size_multipliers.get(size_key, 1.5)))

            # Apply model multiplier if specified
            model_multiplier = Decimal("1.0")
            if model and "model_multipliers" in task_pricing:
                model_key = self._normalize_model_name(model)
                model_multipliers = task_pricing.get("model_multipliers", {})
                model_multiplier = Decimal(str(model_multipliers.get(model_key, 1.0)))

            cost = per_image * num_images * size_multiplier * model_multiplier

        # Apply provider safety factor
        safety_factor = Decimal(str(provider_config.get("safety_factor", 1.1)))
        cost = cost * safety_factor

        # Apply global safety factor for estimation
        global_safety = Decimal(
            str(
                self.pricing_data.get("estimation_config", {}).get("safety_factor", 1.2)
            )
        )
        cost = cost * global_safety

        logger.debug(
            f"Estimated cost for {provider_name}/{task_name}",
            extra={
                "subsys": "vision",
                "event": "pricing.estimate",
                "detail": f"cost={str(cost)}, width={width}, height={height}, num_images={num_images}, duration_seconds={duration_seconds}, model={model}",
            },
        )

        return cost

    def _task_to_pricing_key(self, task: VisionTask) -> str:
        """Convert VisionTask enum to pricing key"""
        # Map all task types to their pricing categories
        mapping = {
            VisionTask.TEXT_TO_IMAGE: "text_to_image",
            VisionTask.IMAGE_TO_IMAGE: "image_to_image",
            VisionTask.TEXT_TO_VIDEO: "text_to_video",
        }

        # Handle additional task types that may exist
        if hasattr(VisionTask, "IMAGE_VARIATION"):
            mapping[VisionTask.IMAGE_VARIATION] = "image_to_image"
        if hasattr(VisionTask, "IMAGE_TO_VIDEO"):
            mapping[VisionTask.IMAGE_TO_VIDEO] = "image_to_video"
        if hasattr(VisionTask, "VIDEO_GENERATION"):
            mapping[VisionTask.VIDEO_GENERATION] = "text_to_video"

        return mapping.get(task, "text_to_image")

    def _get_size_key(self, width: int, height: int) -> str:
        """Get size key for multiplier lookup"""
        # Find closest standard size
        size = max(width, height)
        if size <= 512:
            return "512x512"
        elif size <= 768:
            return "768x768"
        elif size <= 1024:
            return "1024x1024"
        elif size <= 1536:
            return "1536x1536"
        else:
            return "2048x2048"

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for multiplier lookup"""
        model_lower = model.lower()
        if "flux" in model_lower:
            if "schnell" in model_lower:
                return "flux-schnell"
            elif "pro" in model_lower:
                return "flux-pro"
            return "flux"
        elif "sdxl" in model_lower:
            return "sdxl"
        elif "playground" in model_lower:
            return "playground"
        return "default"

    def _get_default_estimate(
        self, task: VisionTask, num_images: int = 1, duration_seconds: float = 4.0
    ) -> Money:
        """Get default estimate when provider/task pricing not available"""
        defaults = self.pricing_data.get("default_estimates", {})
        task_key = self._task_to_pricing_key(task)
        base_cost = defaults.get(task_key, defaults.get("unknown", 0.05))

        if task in [
            VisionTask.TEXT_TO_VIDEO,
            VisionTask.IMAGE_TO_VIDEO,
            VisionTask.VIDEO_GENERATION,
        ]:
            # Scale by duration for video
            return Money(base_cost) * (duration_seconds / 4.0)
        else:
            # Scale by number of images
            return Money(base_cost) * num_images

    def normalize_provider_usage(
        self, provider: VisionProvider, usage_data: Dict[str, Any]
    ) -> Money:
        """
        Normalize provider usage data to USD.

        Handles different provider formats:
        - Credits-based (Novita, Chutes)
        - GPU seconds (Together)
        - Direct USD amounts

        Args:
            provider: Vision provider
            usage_data: Raw usage data from provider

        Returns:
            Normalized cost as Money object
        """
        provider_name = provider.value.lower()
        provider_config = self.pricing_data.get("providers", {}).get(provider_name, {})

        # Check for direct USD amount
        if "cost_usd" in usage_data:
            return Money(usage_data["cost_usd"])
        elif "cost" in usage_data and "currency" in usage_data:
            if usage_data["currency"].upper() == "USD":
                return Money(usage_data["cost"])

        # Check for credits
        if "credits" in usage_data or "credits_used" in usage_data:
            credits = usage_data.get("credits", usage_data.get("credits_used", 0))
            credits_per_dollar = provider_config.get("credits_per_dollar", 100)
            return Money.from_credits(credits, credits_per_dollar)

        # Check for GPU seconds (Together AI)
        if "gpu_seconds" in usage_data:
            gpu_seconds = usage_data["gpu_seconds"]
            gpu_seconds_per_dollar = provider_config.get("gpu_seconds_per_dollar", 1000)
            return Money(
                Decimal(str(gpu_seconds)) / Decimal(str(gpu_seconds_per_dollar))
            )

        # Check for cents
        if "cents" in usage_data or "cost_cents" in usage_data:
            cents = usage_data.get("cents", usage_data.get("cost_cents", 0))
            return Money.from_cents(cents)

        # Log warning and return zero
        logger.warning(
            f"Could not normalize usage data for {provider_name}",
            extra={
                "subsys": "vision",
                "event": "pricing.normalize.warning",
                "detail": str(usage_data),
            },
        )
        return Money.zero()

    def get_max_discrepancy_ratio(self) -> Decimal:
        """Get maximum allowed discrepancy ratio between estimate and actual"""
        return Decimal(
            str(
                self.pricing_data.get("estimation_config", {}).get(
                    "max_discrepancy_ratio", 5.0
                )
            )
        )

    def get_warning_discrepancy_ratio(self) -> Decimal:
        """Get discrepancy ratio threshold for warnings"""
        return Decimal(
            str(
                self.pricing_data.get("estimation_config", {}).get(
                    "warning_discrepancy_ratio", 2.0
                )
            )
        )


# Global pricing table instance
_pricing_table: Optional[PricingTable] = None


def get_pricing_table() -> PricingTable:
    """Get or create global pricing table instance [PA]"""
    global _pricing_table
    if _pricing_table is None:
        _pricing_table = PricingTable()
    return _pricing_table
