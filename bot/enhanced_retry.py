"""
Enhanced retry system with provider/model fallback ladder and circuit breaker.
Implements per-item wall-clock budget and configurable timeouts.
"""

import asyncio
import time
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import logging
import os
from .config import load_config, get_vl_model_ladder

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ProviderConfig:
    """Configuration for a provider/model combination."""

    name: str
    model: str
    timeout: float = 8.0  # Faster per-attempt timeout
    max_attempts: int = 2  # Reduce attempts for speed
    base_delay: float = 1.0  # Much faster base delay
    max_delay: float = 8.0  # Reduce max delay significantly
    exponential_base: float = 1.5  # Less aggressive backoff
    jitter: bool = True


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a provider."""

    status: ProviderStatus = ProviderStatus.HEALTHY
    failure_count: int = 0
    last_failure_time: float = 0.0
    cooldown_duration: float = 2.0  # Much faster cooldown recovery
    failure_threshold: int = 2  # Trigger circuit breaker faster


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_time: float = 0.0
    provider_used: Optional[str] = None
    fallback_occurred: bool = False


class EnhancedRetryManager:
    """Enhanced retry manager with provider fallback and circuit breaker."""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.provider_configs: Dict[str, List[ProviderConfig]] = {}
        self._load_default_configs()

    def _load_default_configs(self):
        """Load provider configurations from .env/config with sensible fallbacks."""
        load_config()

        def _parse_ladder(
            models_env: str | None,
            timeouts_env: str | None,
            default: List[ProviderConfig],
        ) -> List[ProviderConfig]:
            # models entries may be either "provider|model" or just "model" (defaults to openrouter)
            if not models_env:
                return default
            models = [m.strip() for m in models_env.split(",") if m.strip()]
            timeouts: List[float] = []
            if timeouts_env:
                try:
                    timeouts = [
                        float(t.strip()) for t in timeouts_env.split(",") if t.strip()
                    ]
                except Exception:
                    timeouts = []
            # Allow env overrides to inherit tuned defaults (timeout, attempts, backoff)
            # when the provider+model pair matches an entry in the default ladder. This
            # keeps TEXT_FALLBACK_MODELS aligned with default_text/default_vision even
            # when TEXT_FALLBACK_TIMEOUTS is omitted or partially specified.
            default_by_key: Dict[tuple[str, str], ProviderConfig] = {
                (pc.name, pc.model): pc for pc in default
            }
            ladder: List[ProviderConfig] = []
            for idx, entry in enumerate(models):
                if "|" in entry:
                    provider, model = entry.split("|", 1)
                    provider = provider.strip() or "openrouter"
                    model = model.strip()
                else:
                    provider, model = "openrouter", entry
                base_cfg = default_by_key.get((provider, model))

                # Determine timeout priority: explicit env timeout → default ladder timeout
                # for this provider+model (if any) → legacy generic fallback (15/20s).
                if idx < len(timeouts):
                    timeout = timeouts[idx]
                elif base_cfg is not None:
                    timeout = base_cfg.timeout
                else:
                    timeout = 15.0 if provider == "openrouter" else 20.0

                if base_cfg is not None:
                    # Preserve tuned retry/backoff parameters from the default ladder
                    ladder.append(
                        ProviderConfig(
                            provider,
                            model,
                            timeout=timeout,
                            max_attempts=base_cfg.max_attempts,
                            base_delay=base_cfg.base_delay,
                            max_delay=base_cfg.max_delay,
                            exponential_base=base_cfg.exponential_base,
                            jitter=base_cfg.jitter,
                        )
                    )
                else:
                    ladder.append(ProviderConfig(provider, model, timeout=timeout))
            return ladder

        # Optimized defaults with faster timeouts
        default_vision = [
            ProviderConfig(
                "openrouter", "moonshotai/kimi-vl-a3b-thinking:free", timeout=6.0
            ),
            ProviderConfig(
                "openrouter",
                "mistralai/mistral-small-3.2-24b-instruct:free",
                timeout=8.0,
            ),
            ProviderConfig(
                "openrouter", "qwen/qwen2.5-vl-32b-instruct:free", timeout=10.0
            ),
        ]
        default_text = [
            # Prefer chat-style model first for latency and robustness
            ProviderConfig(
                "openrouter",
                "deepseek/deepseek-chat-v3-0324:free",
                timeout=25.0,
                max_attempts=2,
            ),
            ProviderConfig(
                "openrouter",
                "deepseek/deepseek-r1-0528:free",
                timeout=25.0,
                max_attempts=1,
            ),
            ProviderConfig(
                "openrouter",
                "z-ai/glm-4.5-air:free",
                timeout=30.0,
                max_attempts=1,
            ),
        ]
        # Media tasks (e.g., video/audio downloads) are not LLM calls; they often need longer timeouts
        # and fewer attempts. Use an internal single-step "provider" to reuse the retry harness.
        # Timeout policy:
        # - MEDIA_PROVIDER_TIMEOUT env, if set, wins (float seconds)
        # - Else derive from MEDIA_PER_ITEM_BUDGET (minus small margin) if present
        # - Else fallback to sensible default 100s
        try:
            media_timeout_env = os.getenv("MEDIA_PROVIDER_TIMEOUT")
            if media_timeout_env is not None:
                media_timeout = float(media_timeout_env)
            else:
                budget_env = os.getenv("MEDIA_PER_ITEM_BUDGET")
                if budget_env is not None:
                    budget_val = float(budget_env)
                    # keep a 5s safety margin within the overall budget
                    media_timeout = max(30.0, budget_val - 5.0)
                else:
                    media_timeout = 100.0
        except Exception:
            media_timeout = 100.0

        default_media = [
            ProviderConfig(
                "internal", "media-handler", timeout=media_timeout, max_attempts=1
            ),
        ]

        vision_models = os.getenv("VISION_FALLBACK_MODELS")
        vision_timeouts = os.getenv("VISION_FALLBACK_TIMEOUTS")
        text_models = os.getenv("TEXT_FALLBACK_MODELS")
        text_timeouts = os.getenv("TEXT_FALLBACK_TIMEOUTS")

        self.provider_configs["vision"] = _parse_ladder(
            vision_models, vision_timeouts, default_vision
        )
        self.provider_configs["text"] = _parse_ladder(
            text_models, text_timeouts, default_text
        )
        # Media ladder can be overridden via env; if not provided, use defaults above
        media_models = os.getenv("MEDIA_FALLBACK_MODELS")
        media_timeouts = os.getenv("MEDIA_FALLBACK_TIMEOUTS")
        self.provider_configs["media"] = _parse_ladder(
            media_models, media_timeouts, default_media
        )

        self._apply_vl_override()

        # Log parsed ladders
        try:
            v = ", ".join(
                [
                    f"{pc.name}|{pc.model}(t={pc.timeout}s,a={pc.max_attempts})"
                    for pc in self.provider_configs["vision"]
                ]
            )
            t = ", ".join(
                [
                    f"{pc.name}|{pc.model}(t={pc.timeout}s,a={pc.max_attempts})"
                    for pc in self.provider_configs["text"]
                ]
            )
            m = ", ".join(
                [
                    f"{pc.name}|{pc.model}(t={pc.timeout}s,a={pc.max_attempts})"
                    for pc in self.provider_configs["media"]
                ]
            )
            logger.info(
                f"🔧 Fallback ladders loaded → vision: [{v}] | text: [{t}] | media: [{m}]"
            )
        except Exception:
            pass

    def _apply_vl_override(self) -> None:
        """Ensure the configured VL ladder head (VL_MODEL) is reflected in the vision provider order."""
        try:
            override_entries = get_vl_model_ladder()
        except Exception:
            override_entries = []

        if not override_entries:
            return

        vision_providers = self.provider_configs.get("vision", [])
        if not vision_providers and not override_entries:
            return

        # Helper to parse optional provider prefix (provider|model)
        def _parse_entry(entry: str) -> tuple[str, str]:
            if "|" in entry:
                provider, model = entry.split("|", 1)
                return (provider.strip() or "openrouter", model.strip())
            return "openrouter", entry.strip()

        base_by_model: Dict[str, ProviderConfig] = {
            provider.model: provider for provider in vision_providers
        }
        ordered: List[ProviderConfig] = []
        seen_models: set[str] = set()

        for idx, raw_entry in enumerate(override_entries):
            provider_name, model_name = _parse_entry(raw_entry)
            if not model_name or model_name in seen_models:
                continue

            existing = base_by_model.get(model_name)
            if existing:
                ordered.append(existing)
            else:
                # Derive timeout from existing ladder when possible, otherwise fallback to sane defaults
                if idx < len(vision_providers):
                    timeout = vision_providers[idx].timeout
                elif vision_providers:
                    timeout = vision_providers[-1].timeout
                else:
                    timeout = 8.0 + (idx * 2.0)
                ordered.append(
                    ProviderConfig(
                        provider_name,
                        model_name,
                        timeout=timeout,
                    )
                )
            seen_models.add(model_name)

        for provider in vision_providers:
            if provider.model not in seen_models:
                ordered.append(provider)
                seen_models.add(provider.model)

        if ordered:
            self.provider_configs["vision"] = ordered

    def refresh_from_env(self) -> Dict[str, List[str]]:
        """Rebuild provider ladders from the current environment while preserving breaker state."""
        old_breakers = dict(self.circuit_breakers)
        self._load_default_configs()

        new_breakers: Dict[str, CircuitBreakerState] = {}
        all_provider_keys: set[str] = set()
        for providers in self.provider_configs.values():
            for provider in providers:
                key = f"{provider.name}:{provider.model}"
                all_provider_keys.add(key)
                if key in old_breakers:
                    new_breakers[key] = old_breakers[key]
                else:
                    new_breakers[key] = CircuitBreakerState()

        self.circuit_breakers = new_breakers

        summary: Dict[str, List[str]] = {}
        for modality, providers in self.provider_configs.items():
            summary[modality] = [provider.model for provider in providers]

        return summary

    def _get_circuit_breaker(self, provider_key: str) -> CircuitBreakerState:
        """Get or create circuit breaker for provider."""
        if provider_key not in self.circuit_breakers:
            self.circuit_breakers[provider_key] = CircuitBreakerState()
        return self.circuit_breakers[provider_key]

    def _is_provider_available(self, provider_key: str) -> bool:
        """Check if provider is available (circuit breaker check)."""
        breaker = self._get_circuit_breaker(provider_key)

        if breaker.status == ProviderStatus.CIRCUIT_OPEN:
            # Check if cooldown period has passed
            if time.time() - breaker.last_failure_time > breaker.cooldown_duration:
                breaker.status = ProviderStatus.DEGRADED
                breaker.failure_count = 0
                logger.info(f"🔄 Circuit breaker reset for {provider_key}")
                return True
            return False

        return True

    def _record_success(self, provider_key: str):
        """Record successful operation for provider."""
        breaker = self._get_circuit_breaker(provider_key)
        breaker.status = ProviderStatus.HEALTHY
        breaker.failure_count = 0

    def _record_failure(self, provider_key: str):
        """Record failed operation for provider."""
        breaker = self._get_circuit_breaker(provider_key)
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()

        if breaker.failure_count >= breaker.failure_threshold:
            breaker.status = ProviderStatus.CIRCUIT_OPEN
            logger.warning(
                f"⚡ Circuit breaker opened for {provider_key} (failures: {breaker.failure_count})"
            )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        retryable_patterns = [
            "502",
            "503",
            "504",
            "500",
            "429",  # HTTP server errors and rate limits
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "provider returned error",
            "provider error",
            "timeout",
            "connection",
            "network",
            "cancelled",
            "canceled",
            "rate limit",
            "too many requests",
            "client error",
        ]

        # Hard non-retryable 404 / no-endpoints patterns (provider permanently unavailable)
        if "404" in error_str and "no endpoints found" in error_str:
            return False
        if "404" in error_str and "no endpoint" in error_str and "openrouter" in error_str:
            return False
        if "404" in error_str and "model not found" in error_str:
            return False

        # Check both error message and error type name
        return any(pattern in error_str for pattern in retryable_patterns) or any(
            pattern in error_type for pattern in retryable_patterns
        )

    async def run_with_fallback(
        self,
        modality: str,
        coro_factory: Callable[[ProviderConfig], Callable[[], Any]],
        per_item_budget: float = 45.0,
    ) -> RetryResult:
        """
        Run operation with provider fallback ladder and circuit breaker.

        Args:
            modality: Type of operation (vision, text, etc.)
            coro_factory: Factory function that takes ProviderConfig and returns coroutine
            per_item_budget: Maximum wall-clock time for this item
        """
        start_time = time.time()
        providers = self.provider_configs.get(modality, [])

        if not providers:
            return RetryResult(
                success=False,
                error=ValueError(f"No providers configured for modality: {modality}"),
                total_time=time.time() - start_time,
            )

        total_attempts = 0
        fallback_occurred = False
        last_exception: Optional[Exception] = None

        for provider_idx, provider_config in enumerate(providers):
            provider_key = f"{provider_config.name}:{provider_config.model}"
            has_next_provider = provider_idx < len(providers) - 1

            # Check circuit breaker
            if not self._is_provider_available(provider_key):
                logger.info(f"⚡ Skipping {provider_key} (circuit open)")
                fallback_occurred = True
                continue

            # Check remaining budget
            elapsed = time.time() - start_time
            if elapsed >= per_item_budget:
                logger.warning(
                    f"⏱️ Per-item budget ({per_item_budget}s) exceeded, aborting"
                )
                break

            # Try this provider with retries
            provider_attempts = 0
            for attempt in range(provider_config.max_attempts):
                # Check budget before each attempt
                elapsed = time.time() - start_time
                remaining_budget = per_item_budget - elapsed
                if remaining_budget <= 0:
                    logger.warning(f"⏱️ No budget remaining for attempt {attempt + 1}")
                    break

                # Use smaller of provider timeout and remaining budget
                attempt_timeout = min(provider_config.timeout, remaining_budget)

                try:
                    total_attempts += 1
                    provider_attempts += 1

                    logger.info(
                        f"🔄 Attempt {attempt + 1}/{provider_config.max_attempts} with {provider_key} (timeout: {attempt_timeout:.1f}s)"
                    )

                    # Create and run the coroutine with timeout
                    coro = coro_factory(provider_config)()
                    result = await asyncio.wait_for(coro, timeout=attempt_timeout)

                    # Success!
                    self._record_success(provider_key)
                    total_time = time.time() - start_time

                    logger.info(
                        f"✅ Success with {provider_key} after {provider_attempts} attempts ({total_time:.2f}s)"
                    )

                    return RetryResult(
                        success=True,
                        result=result,
                        attempts=total_attempts,
                        total_time=total_time,
                        provider_used=provider_key,
                        fallback_occurred=fallback_occurred or provider_idx > 0,
                    )

                except Exception as e:
                    # Normalize timeouts to a descriptive message [REH]
                    try:
                        if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                            desc = f"LLM attempt timed out after {attempt_timeout:.1f}s for {provider_key}"
                            te = TimeoutError(desc)
                            try:
                                setattr(te, "provider_key", provider_key)
                                setattr(te, "timeout_seconds", attempt_timeout)
                            except Exception:
                                pass
                            e = te
                    except Exception:
                        pass
                    # Treat OpenRouter 404 / no-endpoints as permanent provider unavailability [REH]
                    msg_lower = f"{type(e).__name__}: {e}".lower()
                    if "404" in msg_lower and "no endpoints found" in msg_lower:
                        logger.error(
                            f"❌ Provider unavailable (404 no endpoints) for {provider_key}: {type(e).__name__}: {e}"
                        )
                        last_exception = e
                        self._record_failure(provider_key)
                        breaker = self._get_circuit_breaker(provider_key)
                        breaker.status = ProviderStatus.CIRCUIT_OPEN
                        break

                    logger.warning(
                        f"⚠️ Attempt {attempt + 1} failed with {provider_key}: {type(e).__name__}: {e}"
                    )
                    # Keep track of the last exception so callers can inspect specifics (e.g., Retry-After)
                    last_exception = e

                    if not self._is_retryable_error(e):
                        logger.error(
                            f"❌ Non-retryable error, skipping remaining attempts for {provider_key}: {type(e).__name__}: {e}"
                        )
                        break

                    # Record failure for circuit breaker
                    self._record_failure(provider_key)

                    # Calculate delay for next attempt (if not last attempt)
                    if attempt < provider_config.max_attempts - 1:
                        use_retry_after = False
                        delay = min(
                            provider_config.base_delay
                            * (provider_config.exponential_base**attempt),
                            provider_config.max_delay,
                        )

                        # Respect server-provided Retry-After hint if present on the exception [REH]
                        extra_note = ""
                        try:
                            retry_after_hint = getattr(e, "retry_after_seconds", None)
                            if retry_after_hint is not None:
                                ra = float(retry_after_hint)
                                if ra > 0:
                                    delay = ra  # Follow server hint exactly; budget check below
                                    use_retry_after = True
                                    extra_note = f" (Retry-After={ra:.2f}s)"
                        except Exception:
                            use_retry_after = False

                        # Only apply jitter to backoff-based delays (not Retry-After)
                        if (not use_retry_after) and provider_config.jitter:
                            delay *= 0.8 + random.random() * 0.4

                        # Check if we have budget for the delay
                        elapsed = time.time() - start_time
                        if elapsed + delay >= per_item_budget:
                            logger.warning(
                                f"⏱️ No budget for {delay:.1f}s delay, skipping to next provider"
                            )
                            break

                        logger.info(f"⏳ Retrying in {delay:.2f}s...{extra_note}")
                        await asyncio.sleep(delay)

            # If we get here, all attempts for this provider failed
            fallback_occurred = True
            if has_next_provider:
                logger.warning(
                    f"❌ All attempts failed for {provider_key}, trying next provider"
                )
            else:
                logger.warning(
                    f"❌ All attempts failed for {provider_key}, last provider in ladder"
                )

        # All providers exhausted
        total_time = time.time() - start_time
        return RetryResult(
            success=False,
            error=last_exception or Exception("All providers exhausted"),
            attempts=total_attempts,
            total_time=total_time,
            fallback_occurred=fallback_occurred,
        )


# Global instance
_retry_manager = EnhancedRetryManager()


def get_retry_manager() -> EnhancedRetryManager:
    """Get the global retry manager instance."""
    return _retry_manager
