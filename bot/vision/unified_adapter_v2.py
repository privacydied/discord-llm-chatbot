"""
Unified Vision Adapter with Money type and pricing table [CA][REH]

Provides a unified interface to multiple vision providers with:
- Type-safe Money calculations
- Canonical pricing table
- Provider usage normalization
"""

from typing import Dict, Any, Optional, List
import asyncio

from bot.vision.types import (
    VisionRequest,
    VisionResponse,
    VisionTask,
    VisionProvider,
    VisionError,
    VisionErrorType,
    NormalizedRequest,
)
from bot.vision.money import Money
from bot.vision.pricing_loader import get_pricing_table
from bot.vision.providers.base import ProviderPlugin
from bot.util.logging import get_logger

logger = get_logger(__name__)


class UnifiedVisionAdapter:
    """
    Unified adapter for vision providers with Money-based pricing [CA]

    Features:
    - Deterministic cost estimation from pricing table
    - Provider usage normalization to USD
    - Type-safe Money calculations
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize adapter with config and pricing table"""
        self.config = config
        self.provider_config = config
        self.pricing_table = get_pricing_table()

        # Provider plugins will be loaded lazily
        self._providers: Dict[str, ProviderPlugin] = {}
        self._provider_lock = asyncio.Lock()

        logger.info("UnifiedVisionAdapter initialized with pricing table")

    async def _get_provider(self, name: str) -> Optional[ProviderPlugin]:
        """Get or load provider plugin [REH]"""
        async with self._provider_lock:
            if name not in self._providers:
                # Load provider dynamically
                try:
                    if name == "novita":
                        from bot.vision.providers.novita_adapter import NovitaAdapter

                        self._providers[name] = NovitaAdapter(self.config)
                    elif name == "together":
                        from bot.vision.providers.together_adapter import (
                            TogetherAdapter,
                        )

                        self._providers[name] = TogetherAdapter(self.config)
                    elif name == "chutes":
                        from bot.vision.providers.chutes_adapter import ChutesAdapter

                        self._providers[name] = ChutesAdapter(self.config)
                    else:
                        logger.warning(f"Unknown provider: {name}")
                        return None

                    logger.info(f"Loaded provider: {name}")
                except Exception as e:
                    logger.error(f"Failed to load provider {name}: {e}")
                    return None

            return self._providers.get(name)

    def normalize_request(self, request: VisionRequest) -> NormalizedRequest:
        """Normalize request parameters [CMV]"""
        # Create normalized request with defaults
        normalized = NormalizedRequest(
            task=request.task,
            prompt=request.prompt or "",
            negative_prompt=request.negative_prompt or "",
            width=request.width or 1024,
            height=request.height or 1024,
            num_images=getattr(request, "batch_size", 1),
            steps=request.steps or 20,
            guidance_scale=request.guidance_scale or 7.5,
            seed=request.seed,
            model=request.model,
            style=request.style,
            quality=request.quality or "standard",
            input_image=request.input_image,
            mask_image=request.mask_image,
            strength=request.strength or 0.8,
            video_seconds=getattr(request, "duration_seconds", 4.0),
        )

        # Ensure dimensions are valid
        normalized.width = min(max(normalized.width, 256), 2048)
        normalized.height = min(max(normalized.height, 256), 2048)

        # Round to nearest 64 pixels for better provider compatibility
        normalized.width = (normalized.width // 64) * 64
        normalized.height = (normalized.height // 64) * 64

        return normalized

    def estimate_cost(self, provider: VisionProvider, request: VisionRequest) -> Money:
        """
        Estimate cost using pricing table [PA][CMV]

        Returns Money object with deterministic estimate.
        """
        normalized = self.normalize_request(request)

        return self.pricing_table.estimate_cost(
            provider=provider,
            task=request.task,
            width=normalized.width,
            height=normalized.height,
            num_images=normalized.num_images,
            duration_seconds=normalized.video_seconds,
            model=request.model,
        )

    def normalize_provider_cost(
        self, provider: VisionProvider, response: VisionResponse
    ) -> Money:
        """
        Normalize provider's actual cost to USD [REH]

        Handles different provider formats and ensures Money type.
        """
        # If response already has normalized cost, use it
        if hasattr(response, "actual_cost_money"):
            return response.actual_cost_money

        # Check if actual_cost is already set and valid
        if response.actual_cost and response.actual_cost > 0:
            # Assume it's in USD if not specified otherwise
            return Money(response.actual_cost)

        # Try to extract from provider-specific fields
        usage_data = {}

        if hasattr(response, "provider_metadata"):
            usage_data = response.provider_metadata
        elif hasattr(response, "usage"):
            usage_data = response.usage

        # Use pricing table to normalize
        normalized_cost = self.pricing_table.normalize_provider_usage(
            provider=provider, usage_data=usage_data
        )

        # If still zero, estimate based on task
        if normalized_cost.is_zero() and response.success:
            logger.warning(
                f"No cost data from {provider.value}, using estimate",
                job_id=response.job_id[:8] if response.job_id else "unknown",
            )
            # Create a minimal request for estimation
            if hasattr(response, "dimensions") and response.dimensions:
                width, height = response.dimensions
            else:
                width, height = 1024, 1024

            # Determine task from response
            task = VisionTask.TEXT_TO_IMAGE  # Default
            if response.duration_seconds:
                task = VisionTask.TEXT_TO_VIDEO

            normalized_cost = self.pricing_table.estimate_cost(
                provider=provider,
                task=task,
                width=width,
                height=height,
                num_images=len(response.artifacts) if response.artifacts else 1,
                duration_seconds=response.duration_seconds or 4.0,
                model=response.model_used,
            )
            # Remove safety factor for actual cost
            normalized_cost = normalized_cost / 1.2

        return normalized_cost

    async def submit(self, request: VisionRequest) -> VisionResponse:
        """
        Submit request with cost estimation and normalization [REH]

        Ensures all costs use Money type. Enforces capability and pricing gates.
        """
        # Normalize request
        normalized = self.normalize_request(request)

        # Get provider order with credential/health filtering
        provider_order = self._get_provider_order_with_health_check(request)

        selected_provider = None
        selection_reason = "no_providers"
        last_error = None

        for provider_name in provider_order:
            try:
                # Parse provider:endpoint format
                if ":" in provider_name:
                    provider_name = provider_name.split(":")[0]

                # Get provider plugin
                provider = await self._get_provider(provider_name)
                if not provider:
                    logger.warning(f"Provider {provider_name} not available")
                    continue

                provider_enum = VisionProvider(provider_name.upper())

                # Strict capability gating - check if provider supports the task
                if hasattr(provider, "get_supported_tasks"):
                    supported_tasks = provider.get_supported_tasks()
                elif hasattr(provider, "capabilities"):
                    capabilities = provider.capabilities()
                    supported_tasks = capabilities.get("modes", [])
                else:
                    # Fallback: assume basic support
                    supported_tasks = [
                        VisionTask.TEXT_TO_IMAGE,
                        VisionTask.IMAGE_TO_IMAGE,
                    ]

                if request.task not in supported_tasks:
                    logger.debug(
                        f"Provider {provider_name} does not support {request.task.value}"
                    )
                    continue

                # Strict pricing gating - fail fast if no pricing exists
                try:
                    estimated_cost = self.estimate_cost(provider_enum, request)
                    if estimated_cost.is_zero():
                        logger.info(
                            f"provider.select | task={request.task.value} selected=none reason=no_pricing provider={provider_name}"
                        )
                        selection_reason = "no_pricing"
                        continue
                except Exception:
                    logger.info(
                        f"provider.select | task={request.task.value} selected=none reason=no_pricing provider={provider_name}"
                    )
                    selection_reason = "no_pricing"
                    continue

                # Provider selected successfully
                selected_provider = provider_name
                selection_reason = "supports_capability"
                logger.info(
                    f"provider.select | task={request.task.value} selected={provider_name} reason=supports_capability"
                )

                # Update request with estimate (budget reserve only when estimate exists)
                request.estimated_cost = estimated_cost.to_float()  # For compatibility

                logger.info(
                    f"Submitting to {provider_name}",
                    task=request.task.value,
                    estimated_cost=str(estimated_cost),
                    width=normalized.width,
                    height=normalized.height,
                )

                # Submit to provider
                response = await provider.submit(request)

                if response.success:
                    # Normalize actual cost
                    normalized_cost = self.normalize_provider_cost(
                        provider_enum, response
                    )

                    # Update response with normalized cost
                    response.actual_cost = (
                        normalized_cost.to_float()
                    )  # For compatibility
                    response.actual_cost_money = normalized_cost  # New Money field

                    # Check discrepancy
                    discrepancy_ratio = normalized_cost.ratio_to(estimated_cost)
                    warning_threshold = (
                        self.pricing_table.get_warning_discrepancy_ratio()
                    )

                    if discrepancy_ratio > warning_threshold:
                        logger.warning(
                            "Cost discrepancy detected",
                            provider=provider_name,
                            estimated=str(estimated_cost),
                            actual=str(normalized_cost),
                            ratio=float(discrepancy_ratio),
                        )

                    logger.info(
                        "Request completed successfully",
                        provider=provider_name,
                        estimated_cost=str(estimated_cost),
                        actual_cost=str(normalized_cost),
                        discrepancy_ratio=float(discrepancy_ratio),
                    )

                    return response
                else:
                    last_error = response.error
                    logger.warning(
                        f"Provider {provider_name} failed", error=str(response.error)
                    )

            except Exception as e:
                logger.error(f"Error with provider {provider_name}: {e}")
                last_error = VisionError(
                    error_type=VisionErrorType.PROVIDER_ERROR,
                    message=str(e),
                    provider=VisionProvider(provider_name.upper()),
                )

        # Log final selection result if no provider was selected
        if not selected_provider:
            logger.info(
                f"provider.select | task={request.task.value} selected=none reason={selection_reason}"
            )

        # Early fail with clean user message
        if selection_reason == "no_capability":
            return VisionResponse(
                success=False,
                job_id=request.idempotency_key,
                provider=VisionProvider.UNKNOWN,
                model_used="unknown",
                error=VisionError(
                    error_type=VisionErrorType.VALIDATION_ERROR,
                    message=f"No provider supports {request.task.value}",
                    user_message=f"Sorry, {request.task.value.replace('_', ' ')} is not currently supported.",
                ),
                actual_cost=0.0,
            )
        elif selection_reason == "no_pricing":
            return VisionResponse(
                success=False,
                job_id=request.idempotency_key,
                provider=VisionProvider.UNKNOWN,
                model_used="unknown",
                error=VisionError(
                    error_type=VisionErrorType.VALIDATION_ERROR,
                    message=f"No pricing available for {request.task.value}",
                    user_message="Pricing unavailable for this provider/task. Please try another provider or lower spec.",
                ),
                actual_cost=0.0,
            )

        # All providers failed
        if last_error:
            return VisionResponse(
                success=False,
                job_id=request.idempotency_key,
                provider=VisionProvider.UNKNOWN,
                model_used="unknown",
                error=last_error,
                actual_cost=0.0,
            )
        else:
            return VisionResponse(
                success=False,
                job_id=request.idempotency_key,
                provider=VisionProvider.UNKNOWN,
                model_used="unknown",
                error=VisionError(
                    error_type=VisionErrorType.PROVIDER_ERROR,
                    message="No providers available",
                ),
                actual_cost=0.0,
            )

    def _get_provider_order(self, request: VisionRequest) -> List[str]:
        """Get provider order based on request and config [PA]"""
        # Check for forced provider
        if getattr(request, "preferred_provider", None):
            return [request.preferred_provider.value.lower()]

        # Check for model override
        if hasattr(self, "resolve_model_selection"):
            model_selection = self.resolve_model_selection(
                self.normalize_request(request)
            )
            if model_selection:
                return [model_selection.provider]

        # Use default policy
        policy = self.provider_config.get("vision", {}).get("default_policy", {})
        provider_order = policy.get("provider_order", ["novita", "together", "chutes"])

        # Parse provider:endpoint format
        resolved_order = []
        for entry in provider_order:
            if ":" in entry:
                provider_name = entry.split(":")[0]
                resolved_order.append(provider_name)
            else:
                resolved_order.append(entry)

        return resolved_order

    def _get_provider_order_with_health_check(
        self, request: VisionRequest
    ) -> List[str]:
        """Get provider order with credential and health filtering [REH][IV]"""
        # Get base provider order
        base_order = self._get_provider_order(request)
        filtered_order = []

        for provider_name in base_order:
            try:
                # Parse provider:endpoint format
                if ":" in provider_name:
                    provider_name = provider_name.split(":")[0]

                # Check if provider has valid credentials
                if not self._has_valid_credentials(provider_name):
                    self.logger.debug(
                        f"Provider {provider_name} filtered: missing/invalid credentials"
                    )
                    continue

                # Check if provider is healthy (basic health check)
                if not self._is_provider_healthy(provider_name):
                    self.logger.debug(f"Provider {provider_name} filtered: unhealthy")
                    continue

                # Provider passed all checks
                filtered_order.append(provider_name)
                self.logger.debug(
                    f"Provider {provider_name} passed credential/health checks"
                )

            except Exception as e:
                self.logger.warning(f"Error checking provider {provider_name}: {e}")
                continue

        if not filtered_order:
            self.logger.warning("No providers passed credential/health checks")

        return filtered_order

    def _has_valid_credentials(self, provider_name: str) -> bool:
        """Check if provider has valid API credentials [IV]"""
        try:
            if provider_name == "novita":
                api_key = self.config.get("VISION_API_KEY", "")
                return bool(api_key and api_key.strip())

            elif provider_name == "together":
                api_key = self.config.get("VISION_API_KEY_TOGETHER", "")
                return bool(api_key and api_key.strip())

            elif provider_name == "chutes":
                api_key = self.config.get("VISION_API_KEY_CHUTES", "")
                return bool(api_key and api_key.strip())

            else:
                # Unknown provider - assume no credentials needed for now
                return True

        except Exception as e:
            self.logger.warning(f"Error checking credentials for {provider_name}: {e}")
            return False

    def _is_provider_healthy(self, provider_name: str) -> bool:
        """Check if provider is in a healthy state [PA]"""
        try:
            # For now, implement basic health checks
            # In production, this could check rate limits, service status, etc.

            if provider_name == "novita":
                # Check if API key exists and basic config is valid
                api_key = self.config.get("VISION_API_KEY", "")
                return bool(api_key and len(api_key) > 10)  # Basic length check

            elif provider_name == "together":
                # Check if API key exists
                api_key = self.config.get("VISION_API_KEY_TOGETHER", "")
                return bool(api_key and len(api_key) > 10)

            elif provider_name == "chutes":
                # Check if API key exists
                api_key = self.config.get("VISION_API_KEY_CHUTES", "")
                return bool(api_key and len(api_key) > 10)

            else:
                # Unknown provider - assume healthy
                return True

        except Exception as e:
            self.logger.warning(f"Error checking health for {provider_name}: {e}")
            return False

    def resolve_model_selection(self, request: NormalizedRequest) -> Optional[Any]:
        """Resolve model selection from config [PA]"""
        # This would check VISION_MODEL env var or config
        # Simplified for now
        return None


# ---- Compatibility Shim -------------------------------------------------------
# Re-export canonical adapter to preserve imports from unified_adapter_v2.
try:
    from .unified_adapter import UnifiedVisionAdapter as _UnifiedVisionAdapterCanonical

    UnifiedVisionAdapter = _UnifiedVisionAdapterCanonical  # type: ignore
except Exception:
    # If canonical import fails (during partial installs/tests), do nothing.
    pass
