"""
Enhanced retry system with provider/model fallback ladder and circuit breaker.
Implements per-item wall-clock budget and configurable timeouts.
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import logging
import os
from .config import load_config

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
    timeout: float = 12.0  # Per-attempt timeout
    max_attempts: int = 3
    base_delay: float = 5.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True

@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a provider."""
    status: ProviderStatus = ProviderStatus.HEALTHY
    failure_count: int = 0
    last_failure_time: float = 0.0
    cooldown_duration: float = 60.0  # 60s cooldown
    failure_threshold: int = 3

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
        cfg = load_config()

        def _parse_ladder(models_env: str | None, timeouts_env: str | None, default: List[ProviderConfig]) -> List[ProviderConfig]:
            # models entries may be either "provider|model" or just "model" (defaults to openrouter)
            if not models_env:
                return default
            models = [m.strip() for m in models_env.split(',') if m.strip()]
            timeouts: List[float] = []
            if timeouts_env:
                try:
                    timeouts = [float(t.strip()) for t in timeouts_env.split(',') if t.strip()]
                except Exception:
                    timeouts = []
            ladder: List[ProviderConfig] = []
            for idx, entry in enumerate(models):
                if '|' in entry:
                    provider, model = entry.split('|', 1)
                    provider = provider.strip() or 'openrouter'
                    model = model.strip()
                else:
                    provider, model = 'openrouter', entry
                timeout = timeouts[idx] if idx < len(timeouts) else (15.0 if provider == 'openrouter' else 20.0)
                ladder.append(ProviderConfig(provider, model, timeout=timeout))
            return ladder

        # Defaults if .env not provided
        default_vision = [
            ProviderConfig("openrouter", "moonshotai/kimi-vl-a3b-thinking:free", timeout=12.0),
            ProviderConfig("openrouter", "openai/gpt-4o-mini", timeout=15.0),
            ProviderConfig("openrouter", "meta-llama/llama-3.2-11b-vision-instruct:free", timeout=18.0),
        ]
        default_text = [
            ProviderConfig("openrouter", "openai/gpt-4o-mini", timeout=20.0),
            ProviderConfig("openrouter", "meta-llama/llama-3.1-8b-instruct:free", timeout=25.0),
            ProviderConfig("ollama", "llama3.1", timeout=30.0),
        ]

        vision_models = os.getenv('VISION_FALLBACK_MODELS')
        vision_timeouts = os.getenv('VISION_FALLBACK_TIMEOUTS')
        text_models = os.getenv('TEXT_FALLBACK_MODELS')
        text_timeouts = os.getenv('TEXT_FALLBACK_TIMEOUTS')

        self.provider_configs["vision"] = _parse_ladder(vision_models, vision_timeouts, default_vision)
        self.provider_configs["text"] = _parse_ladder(text_models, text_timeouts, default_text)

        # Log parsed ladders
        try:
            v = ", ".join([f"{pc.name}|{pc.model}(t={pc.timeout}s)" for pc in self.provider_configs["vision"]])
            t = ", ".join([f"{pc.name}|{pc.model}(t={pc.timeout}s)" for pc in self.provider_configs["text"]])
            logger.info(f"🔧 Fallback ladders loaded → vision: [{v}] | text: [{t}]")
        except Exception:
            pass
    
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
            logger.warning(f"⚡ Circuit breaker opened for {provider_key} (failures: {breaker.failure_count})")
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        error_str = str(error).lower()
        retryable_patterns = [
            "502", "503", "504", "500",  # HTTP server errors
            "bad gateway", "service unavailable", "gateway timeout",
            "provider returned error", "provider error",
            "timeout", "connection", "network",
            "rate limit", "too many requests"
        ]
        
        return any(pattern in error_str for pattern in retryable_patterns)
    
    async def run_with_fallback(
        self,
        modality: str,
        coro_factory: Callable[[ProviderConfig], Callable[[], Any]],
        per_item_budget: float = 45.0
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
                total_time=time.time() - start_time
            )
        
        total_attempts = 0
        fallback_occurred = False
        
        for provider_idx, provider_config in enumerate(providers):
            provider_key = f"{provider_config.name}:{provider_config.model}"
            
            # Check circuit breaker
            if not self._is_provider_available(provider_key):
                logger.info(f"⚡ Skipping {provider_key} (circuit open)")
                fallback_occurred = True
                continue
            
            # Check remaining budget
            elapsed = time.time() - start_time
            if elapsed >= per_item_budget:
                logger.warning(f"⏱️ Per-item budget ({per_item_budget}s) exceeded, aborting")
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
                    
                    logger.info(f"🔄 Attempt {attempt + 1}/{provider_config.max_attempts} with {provider_key} (timeout: {attempt_timeout:.1f}s)")
                    
                    # Create and run the coroutine with timeout
                    coro = coro_factory(provider_config)()
                    result = await asyncio.wait_for(coro, timeout=attempt_timeout)
                    
                    # Success!
                    self._record_success(provider_key)
                    total_time = time.time() - start_time
                    
                    logger.info(f"✅ Success with {provider_key} after {provider_attempts} attempts ({total_time:.2f}s)")
                    
                    return RetryResult(
                        success=True,
                        result=result,
                        attempts=total_attempts,
                        total_time=total_time,
                        provider_used=provider_key,
                        fallback_occurred=fallback_occurred or provider_idx > 0
                    )
                
                except Exception as e:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed with {provider_key}: {e}")
                    
                    if not self._is_retryable_error(e):
                        logger.info(f"❌ Non-retryable error, skipping remaining attempts for {provider_key}")
                        break
                    
                    # Record failure for circuit breaker
                    self._record_failure(provider_key)
                    
                    # Calculate delay for next attempt (if not last attempt)
                    if attempt < provider_config.max_attempts - 1:
                        delay = min(
                            provider_config.base_delay * (provider_config.exponential_base ** attempt),
                            provider_config.max_delay
                        )
                        
                        if provider_config.jitter:
                            delay *= (0.5 + random.random() * 0.5)  # ±25% jitter
                        
                        # Check if we have budget for the delay
                        elapsed = time.time() - start_time
                        if elapsed + delay >= per_item_budget:
                            logger.warning(f"⏱️ No budget for {delay:.1f}s delay, skipping to next provider")
                            break
                        
                        logger.info(f"⏳ Retrying in {delay:.2f}s...")
                        await asyncio.sleep(delay)
            
            # If we get here, all attempts for this provider failed
            fallback_occurred = True
            logger.warning(f"❌ All attempts failed for {provider_key}, trying next provider")
        
        # All providers exhausted
        total_time = time.time() - start_time
        return RetryResult(
            success=False,
            error=Exception("All providers exhausted"),
            attempts=total_attempts,
            total_time=total_time,
            fallback_occurred=fallback_occurred
        )

# Global instance
_retry_manager = EnhancedRetryManager()

def get_retry_manager() -> EnhancedRetryManager:
    """Get the global retry manager instance."""
    return _retry_manager
