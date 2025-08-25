"""
Unified Vision Provider Adapter

Single adapter with pluggable provider system for vision generation.
Normalizes requests/responses and handles provider selection, fallback, and error mapping.
"""

import asyncio
import aiohttp
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import base64
import io
from enum import Enum

from bot.util.logging import get_logger
from .types import (
    VisionRequest, VisionResponse, VisionProvider, VisionTask,
    VisionError, VisionErrorType
)

logger = get_logger(__name__)


# Unified status mapping for all providers
class UnifiedStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    UPSCALING = "upscaling"
    SAFETY_REVIEW = "safety_review"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NormalizedRequest:
    """Normalized request format for all providers"""
    task: VisionTask
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    input_image_data: Optional[bytes] = None
    input_image_url: Optional[str] = None
    video_seconds: Optional[int] = None
    fps: Optional[int] = None
    batch_size: int = 1
    safety_mode: str = "strict"
    preferred_model: Optional[str] = None


@dataclass 
class UnifiedJobStatus:
    """Unified job status across all providers"""
    status: UnifiedStatus
    progress_percent: int = 0
    phase: str = ""
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    provider_raw: Optional[Dict] = None


@dataclass
class UnifiedResult:
    """Unified result format"""
    assets: List[str] = field(default_factory=list)  # URLs or file paths
    metadata: Dict[str, Any] = field(default_factory=dict)
    final_cost: float = 0.0
    provider_used: str = ""
    warnings: List[str] = field(default_factory=list)


class ProviderPlugin(ABC):
    """Abstract base class for provider plugins"""
    
    def __init__(self, name: str, config: Dict[str, Any], api_key: str):
        self.name = name
        self.config = config
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
    
    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities and limits"""
        pass
    
    @abstractmethod
    async def submit(self, request: NormalizedRequest) -> str:
        """Submit job and return job_id"""
        pass
    
    @abstractmethod
    async def poll(self, job_id: str) -> UnifiedJobStatus:
        """Poll job status"""
        pass
    
    @abstractmethod
    async def fetch_result(self, job_id: str) -> UnifiedResult:
        """Fetch final result"""
        pass
    
    async def cancel(self, job_id: str) -> bool:
        """Cancel job (optional)"""
        return False
    
    async def startup(self):
        """Initialize provider connection"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=300)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def shutdown(self):
        """Cleanup provider connection"""
        if self.session:
            await self.session.close()
            self.session = None


class TogetherPlugin(ProviderPlugin):
    """Together.ai provider plugin"""
    
    def __init__(self, name: str, config: Dict[str, Any], api_key: str):
        super().__init__(name, config, api_key)
        self.base_url = config.get("base_url", "https://api.together.xyz")
        self.model_map = {
            VisionTask.TEXT_TO_IMAGE: "black-forest-labs/FLUX.1-schnell-Free",
            VisionTask.IMAGE_TO_IMAGE: "black-forest-labs/FLUX.1-schnell-Free",
            VisionTask.VIDEO_GENERATION: "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        }
    
    def capabilities(self) -> Dict[str, Any]:
        return {
            "modes": [VisionTask.TEXT_TO_IMAGE, VisionTask.IMAGE_TO_IMAGE, VisionTask.VIDEO_GENERATION],
            "max_size": (1536, 1536),
            "max_steps": 50,
            "supports_negative_prompt": True,
            "supports_batch": False,
            "nsfw_policy": "blocked",
            "video_max_seconds": 4
        }
    
    async def submit(self, request: NormalizedRequest) -> str:
        """Submit to Together.ai API with unified error handling [REH]"""
        model = self.model_map.get(request.task)
        if not model:
            raise VisionError(
                message=f"Task {request.task.value} not supported by Together.ai",
                error_type=VisionErrorType.UNSUPPORTED_TASK,
                user_message=f"Sorry, {request.task.value} is not supported by this provider."
            )
        
        # Build request payload
        payload = {
            "model": model,
            "prompt": request.prompt,
            "width": min(request.width, 1536),
            "height": min(request.height, 1536),
            "steps": min(request.steps, 50),
            "n": 1
        }
        
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        
        if request.seed:
            payload["seed"] = request.seed
            
        if request.input_image_data and request.task == VisionTask.IMAGE_TO_IMAGE:
            # Convert to base64
            b64_image = base64.b64encode(request.input_image_data).decode()
            payload["image"] = f"data:image/png;base64,{b64_image}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/images/generations",
                json=payload,
                headers=headers
            ) as response:
                error_text = await response.text()
                
                # Unified error handling with proper taxonomy [REH]
                if response.status == 400:
                    if "content policy" in error_text.lower() or "safety" in error_text.lower():
                        raise VisionError(
                            message=f"Content filtered: {error_text}",
                            error_type=VisionErrorType.CONTENT_FILTERED,
                            user_message="Your request was blocked by content safety filters. Please modify your prompt."
                        )
                    elif "invalid" in error_text.lower() or "malformed" in error_text.lower():
                        raise VisionError(
                            message=f"Invalid request: {error_text}",
                            error_type=VisionErrorType.VALIDATION_ERROR,
                            user_message="There was an issue with your request parameters. Please check and try again."
                        )
                elif response.status == 401:
                    raise VisionError(
                        message="Together.ai authentication failed",
                        error_type=VisionErrorType.AUTHENTICATION_ERROR,
                        user_message="Vision service authentication failed. Please contact support."
                    )
                elif response.status == 429:
                    raise VisionError(
                        message="Together.ai rate limit exceeded",
                        error_type=VisionErrorType.RATE_LIMITED,
                        user_message="Too many requests. Please wait a moment and try again."
                    )
                elif response.status == 402 or "quota" in error_text.lower():
                    raise VisionError(
                        message="Together.ai quota exceeded",
                        error_type=VisionErrorType.QUOTA_EXCEEDED,
                        user_message="Service quota exceeded. Please try again later."
                    )
                elif response.status >= 500:
                    raise VisionError(
                        message=f"Together.ai server error: {error_text}",
                        error_type=VisionErrorType.SERVER_ERROR,
                        user_message="The vision service is temporarily unavailable. Please try again."
                    )
                elif response.status != 200:
                    raise VisionError(
                        message=f"Together.ai error ({response.status}): {error_text}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="Vision generation failed. Please try again."
                    )
                
                result = await response.json()
                
                # Together.ai returns results immediately for images
                job_id = f"together_{int(time.time() * 1000)}"
                
                # Store result for polling interface compatibility
                self._results = getattr(self, '_results', {})
                self._results[job_id] = {
                    "status": "completed",
                    "data": result.get("data", []),
                    "cost": self._calculate_cost(request)
                }
                
                return job_id
                
        except VisionError:
            raise
        except Exception as e:
            raise VisionError(
                message=f"Together.ai connection error: {str(e)}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Unable to connect to vision service. Please try again."
            )
    
    async def poll(self, job_id: str) -> UnifiedJobStatus:
        """Poll job status (Together.ai is typically immediate)"""
        results = getattr(self, '_results', {})
        
        if job_id not in results:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percent=0,
                phase="Job not found"
            )
        
        job_data = results[job_id]
        
        if job_data["status"] == "completed":
            return UnifiedJobStatus(
                status=UnifiedStatus.COMPLETED,
                progress_percent=100,
                phase="Generation complete",
                actual_cost=job_data["cost"]
            )
        else:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percent=0,
                phase="Generation failed"
            )
    
    async def fetch_result(self, job_id: str) -> UnifiedResult:
        """Fetch final result from Together.ai"""
        results = getattr(self, '_results', {})
        job_data = results.get(job_id, {})
        
        assets = []
        for item in job_data.get("data", []):
            if "url" in item:
                assets.append(item["url"])
        
        return UnifiedResult(
            assets=assets,
            final_cost=job_data.get("cost", 0.0),
            provider_used="together",
            metadata={"model": "FLUX.1-schnell"}
        )
    
    def _calculate_cost(self, request: NormalizedRequest) -> float:
        """Calculate estimated cost for Together.ai"""
        # Together.ai pricing (example rates)
        base_cost = 0.02  # per image
        pixel_cost = (request.width * request.height) / (1024 * 1024) * 0.005
        return base_cost + pixel_cost


class NovitaPlugin(ProviderPlugin):
    """Novita.ai provider plugin"""
    
    def __init__(self, name: str, config: Dict[str, Any], api_key: str):
        super().__init__(name, config, api_key)
        self.base_url = config.get("base_url", "https://api.novita.ai")
        self.model_map = {
            VisionTask.TEXT_TO_IMAGE: "sd_xl_base_1.0.safetensors",
            VisionTask.IMAGE_TO_IMAGE: "sd_xl_base_1.0.safetensors",
            VisionTask.VIDEO_GENERATION: "stable-video-diffusion-img2vid-xt"
        }
    
    def capabilities(self) -> Dict[str, Any]:
        return {
            "modes": [VisionTask.TEXT_TO_IMAGE, VisionTask.IMAGE_TO_IMAGE, VisionTask.VIDEO_GENERATION],
            "max_size": (2048, 2048),
            "max_steps": 60,
            "supports_negative_prompt": True,
            "supports_batch": True,
            "nsfw_policy": "filtered",
            "video_max_seconds": 6
        }
    
    async def submit(self, request: NormalizedRequest) -> str:
        """Submit to Novita.ai API with unified error handling [REH]"""
        model = self.model_map.get(request.task)
        if not model:
            raise VisionError(
                message=f"Task {request.task.value} not supported by Novita.ai", 
                error_type=VisionErrorType.UNSUPPORTED_TASK,
                user_message=f"Sorry, {request.task.value} is not supported by this provider."
            )
        
        # Build request payload for Novita.ai format
        payload = {
            "model_name": model,
            "prompt": request.prompt,
            "width": min(request.width, 2048),
            "height": min(request.height, 2048),
            "steps": min(request.steps, 60),
            "guidance_scale": request.guidance_scale,
            "batch_size": 1
        }
        
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        
        if request.seed:
            payload["seed"] = request.seed
            
        if request.input_image_data and request.task == VisionTask.IMAGE_TO_IMAGE:
            # Novita.ai expects base64 encoded images
            b64_image = base64.b64encode(request.input_image_data).decode()
            payload["init_image"] = b64_image
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/v3/async/txt2img",  # Novita.ai async endpoint
                json=payload,
                headers=headers
            ) as response:
                error_text = await response.text()
                
                # Unified error handling with proper taxonomy [REH]
                if response.status == 400:
                    if "nsfw" in error_text.lower() or "content" in error_text.lower():
                        raise VisionError(
                            message=f"Content filtered: {error_text}",
                            error_type=VisionErrorType.CONTENT_FILTERED,
                            user_message="Your request was blocked by content safety filters. Please modify your prompt."
                        )
                    elif "invalid" in error_text.lower() or "parameter" in error_text.lower():
                        raise VisionError(
                            message=f"Invalid request: {error_text}",
                            error_type=VisionErrorType.VALIDATION_ERROR,
                            user_message="There was an issue with your request parameters. Please check and try again."
                        )
                elif response.status == 401:
                    raise VisionError(
                        message="Novita.ai authentication failed",
                        error_type=VisionErrorType.AUTHENTICATION_ERROR,
                        user_message="Vision service authentication failed. Please contact support."
                    )
                elif response.status == 429:
                    raise VisionError(
                        message="Novita.ai rate limit exceeded",
                        error_type=VisionErrorType.RATE_LIMITED,
                        user_message="Too many requests. Please wait a moment and try again."
                    )
                elif response.status == 402 or "credit" in error_text.lower():
                    raise VisionError(
                        message="Novita.ai quota exceeded",
                        error_type=VisionErrorType.QUOTA_EXCEEDED,
                        user_message="Service quota exceeded. Please try again later."
                    )
                elif response.status >= 500:
                    raise VisionError(
                        message=f"Novita.ai server error: {error_text}",
                        error_type=VisionErrorType.SERVER_ERROR,
                        user_message="The vision service is temporarily unavailable. Please try again."
                    )
                elif response.status != 200:
                    raise VisionError(
                        message=f"Novita.ai error ({response.status}): {error_text}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="Vision generation failed. Please try again."
                    )
                
                result = await response.json()
                job_id = result.get("task_id", f"novita_{int(time.time() * 1000)}")
                
                # Store job for tracking
                self._jobs = getattr(self, '_jobs', {})
                self._jobs[job_id] = {
                    "status": "running",
                    "progress": 0,
                    "start_time": time.time(),
                    "cost": self._calculate_cost(request),
                    "result": None
                }
                
                return job_id
                
        except VisionError:
            raise
        except Exception as e:
            raise VisionError(
                message=f"Novita.ai connection error: {str(e)}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Unable to connect to vision service. Please try again."
            )
    
    async def poll(self, job_id: str) -> UnifiedJobStatus:
        """Poll Novita.ai job status (mock implementation)"""
        jobs = getattr(self, '_jobs', {})
        
        if job_id not in jobs:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percent=0,
                phase="Job not found"
            )
        
        job_data = jobs[job_id]
        elapsed = time.time() - job_data["start_time"]
        
        # Simulate progress
        if elapsed < 10:
            progress = min(int(elapsed * 10), 90)
            return UnifiedJobStatus(
                status=UnifiedStatus.RUNNING,
                progress_percent=progress,
                phase="Generating image",
                estimated_cost=job_data["cost"]
            )
        else:
            job_data["status"] = "completed"
            return UnifiedJobStatus(
                status=UnifiedStatus.COMPLETED,
                progress_percent=100,
                phase="Generation complete",
                actual_cost=job_data["cost"]
            )
    
    async def fetch_result(self, job_id: str) -> UnifiedResult:
        """Fetch result from Novita.ai (mock implementation)"""
        jobs = getattr(self, '_jobs', {})
        job_data = jobs.get(job_id, {})
        
        # Mock result
        return UnifiedResult(
            assets=["https://example.com/generated_image.png"],
            final_cost=job_data.get("cost", 0.0),
            provider_used="novita",
            metadata={"model": "SDXL"}
        )
    
    def _calculate_cost(self, request: NormalizedRequest) -> float:
        """Calculate estimated cost for Novita.ai"""
        base_cost = 0.018  # per image
        pixel_cost = (request.width * request.height) / (1024 * 1024) * 0.004
        return base_cost + pixel_cost


class UnifiedVisionAdapter:
    """
    Unified Vision adapter with pluggable provider system [CA][REH][SFT]
    
    Features:
    - Automatic provider selection and fallback
    - Parameter normalization and validation
    - Unified error handling and status mapping
    - Cost estimation and budget enforcement
    - JSON-based configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.providers: Dict[str, ProviderPlugin] = {}
        self.provider_config = {}
        
        # Load provider configuration
        self._load_provider_config()
        self._initialize_providers()
    
    def _load_provider_config(self):
        """Load provider configuration from JSON or defaults [IV]"""
        config_path = self.config.get("VISION_PROVIDER_CONFIG_PATH")
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    self.provider_config = json.load(f)
                self.logger.info(f"Loaded provider config from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load provider config: {e}")
                self.provider_config = self._default_provider_config()
        else:
            self.provider_config = self._default_provider_config()
    
    def _default_provider_config(self) -> Dict[str, Any]:
        """Default provider configuration [CMV]"""
        return {
            "vision": {
                "default_policy": {
                    "provider_order": ["together", "novita"],
                    "budget_per_job_usd": 0.25,
                    "prefer_model": {"image": "flux.1-pro", "video": "svd-xt"},
                    "nsfw_mode": "block",
                    "auto_fallback": True,
                    "max_retries_per_provider": 2
                },
                "providers": [
                    {
                        "name": "together",
                        "base_url": "https://api.together.xyz",
                        "api_key_env": "VISION_API_KEY",
                        "enabled": True,
                        "priority": 1,
                        "price": {"image_base": 0.02, "image_per_px": 0.000005, "video_per_s": 0.06},
                        "limits": {"max_size": "1536x1536", "max_steps": 50}
                    },
                    {
                        "name": "novita",
                        "base_url": "https://api.novita.ai",
                        "api_key_env": "VISION_API_KEY",
                        "enabled": True,
                        "priority": 2,
                        "price": {"image_base": 0.018, "image_per_px": 0.000004, "video_per_s": 0.05},
                        "limits": {"max_size": "2048x2048", "max_steps": 60}
                    }
                ]
            }
        }
    
    def _initialize_providers(self):
        """Initialize provider plugins [CA]"""
        api_key = self.config.get("VISION_API_KEY", "")
        
        for provider_config in self.provider_config["vision"]["providers"]:
            if not provider_config.get("enabled", True):
                continue
                
            name = provider_config["name"]
            
            # Get API key from environment
            key_env = provider_config.get("api_key_env", "VISION_API_KEY")
            provider_key = self.config.get(key_env, api_key)
            
            if not provider_key:
                self.logger.warning(f"No API key found for provider {name}")
                continue
            
            try:
                if name == "together":
                    plugin = TogetherPlugin(name, provider_config, provider_key)
                elif name == "novita":
                    plugin = NovitaPlugin(name, provider_config, provider_key)
                else:
                    self.logger.warning(f"Unknown provider: {name}")
                    continue
                
                self.providers[name] = plugin
                self.logger.info(f"Initialized provider: {name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {name}: {e}")
    
    async def startup(self):
        """Initialize all provider connections [REH]"""
        for name, provider in self.providers.items():
            try:
                await provider.startup()
                self.logger.info(f"Started provider: {name}")
            except Exception as e:
                self.logger.error(f"Failed to start provider {name}: {e}")
    
    async def shutdown(self):
        """Cleanup all provider connections [RM]"""
        for name, provider in self.providers.items():
            try:
                await provider.shutdown()
                self.logger.info(f"Shutdown provider: {name}")
            except Exception as e:
                self.logger.error(f"Failed to shutdown provider {name}: {e}")
    
    def normalize_request(self, request: VisionRequest) -> NormalizedRequest:
        """Normalize request parameters across providers [IV]"""
        return NormalizedRequest(
            task=request.task,
            prompt=request.prompt or "",
            negative_prompt=request.negative_prompt,
            width=request.width or 1024,
            height=request.height or 1024,
            steps=request.steps or 20,
            guidance_scale=request.guidance_scale or 7.5,
            seed=request.seed,
            input_image_data=request.input_image_data,
            input_image_url=request.input_image_url,
            batch_size=1,  # Start with single images
            safety_mode="strict"
        )
    
    def select_provider(self, request: NormalizedRequest) -> Optional[ProviderPlugin]:
        """Select best provider for request based on capabilities and policy [CA]"""
        policy = self.provider_config["vision"]["default_policy"]
        provider_order = policy.get("provider_order", ["together", "novita"])
        
        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            capabilities = provider.capabilities()
            
            # Check if provider supports the task
            if request.task not in capabilities.get("modes", []):
                continue
            
            # Check size limits
            max_w, max_h = capabilities.get("max_size", (1024, 1024))
            if request.width > max_w or request.height > max_h:
                continue
            
            # Estimate cost and check budget
            try:
                estimated_cost = self._estimate_cost(provider, request)
                budget_limit = policy.get("budget_per_job_usd", 0.25)
                
                if estimated_cost > budget_limit:
                    self.logger.warning(
                        f"Provider {provider_name} cost ${estimated_cost:.3f} exceeds budget ${budget_limit:.3f}"
                    )
                    continue
                
                self.logger.info(f"Selected provider: {provider_name} (cost: ${estimated_cost:.3f})")
                return provider
                
            except Exception as e:
                self.logger.warning(f"Cost estimation failed for {provider_name}: {e}")
                continue
        
        return None
    
    def _estimate_cost(self, provider: ProviderPlugin, request: NormalizedRequest) -> float:
        """Estimate cost for provider and request [PA]"""
        config = next(
            (p for p in self.provider_config["vision"]["providers"] if p["name"] == provider.name),
            {}
        )
        
        pricing = config.get("price", {})
        base_cost = pricing.get("image_base", 0.02)
        
        if request.task == VisionTask.VIDEO_GENERATION:
            video_cost = pricing.get("video_per_s", 0.05)
            duration = request.video_seconds or 4
            return base_cost + (video_cost * duration)
        else:
            pixel_cost = pricing.get("image_per_px", 0.000005)
            pixels = request.width * request.height
            return base_cost + (pixels * pixel_cost)
    
    async def submit(self, request: VisionRequest) -> Tuple[str, str]:
        """
        Submit job to best available provider with automatic fallback [REH]
        
        Returns:
            Tuple of (job_id, provider_name)
        """
        normalized = self.normalize_request(request)
        policy = self.provider_config["vision"]["default_policy"]
        provider_order = policy.get("provider_order", ["together", "novita"])
        max_retries = policy.get("max_retries_per_provider", 2)
        
        last_error = None
        
        # Try each provider in order with fallback
        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
                
            # Check if provider supports the task
            capabilities = provider.capabilities()
            if normalized.task not in capabilities.get("modes", []):
                self.logger.debug(f"Provider {provider_name} doesn't support {normalized.task}")
                continue
            
            # Check size constraints
            max_w, max_h = capabilities.get("max_size", (1024, 1024))
            if normalized.width > max_w or normalized.height > max_h:
                self.logger.debug(f"Provider {provider_name} size limit exceeded")
                continue
            
            # Try with retries for transient errors
            for attempt in range(max_retries + 1):
                try:
                    job_id = await provider.submit(normalized)
                    # Prefix with provider name for routing  
                    full_job_id = f"{provider.name}:{job_id}"
                    
                    self.logger.info(f"✅ Submitted job {full_job_id} to provider {provider.name}")
                    return full_job_id, provider.name
                    
                except VisionError as e:
                    last_error = e
                    
                    # Don't retry certain error types
                    if e.error_type in [
                        VisionErrorType.CONTENT_FILTERED,
                        VisionErrorType.QUOTA_EXCEEDED,
                        VisionErrorType.AUTHENTICATION_ERROR,
                        VisionErrorType.VALIDATION_ERROR,
                        VisionErrorType.UNSUPPORTED_TASK
                    ]:
                        self.logger.warning(f"❌ Non-retryable error from {provider_name}: {e.error_type.value}")
                        break  # Try next provider
                    
                    # Retry transient errors
                    if attempt < max_retries:
                        delay = 1.0 * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(
                            f"⚠️ Provider {provider_name} attempt {attempt + 1} failed, retrying in {delay}s: {e.message}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        self.logger.error(f"❌ Provider {provider_name} failed after {max_retries + 1} attempts")
                        break  # Try next provider
                        
                except Exception as e:
                    last_error = VisionError(
                        message=f"Unexpected error from {provider_name}: {str(e)}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="Vision generation failed. Please try again."
                    )
                    
                    if attempt < max_retries:
                        delay = 1.0 * (2 ** attempt)
                        self.logger.warning(f"⚠️ Unexpected error from {provider_name}, retrying in {delay}s")
                        await asyncio.sleep(delay)
                    else:
                        self.logger.error(f"❌ Provider {provider_name} failed with unexpected error")
                        break
        
        # All providers exhausted
        if last_error:
            raise last_error
        else:
            raise VisionError(
                message="No suitable provider available for request",
                error_type=VisionErrorType.NO_PROVIDER_AVAILABLE,
                user_message="Vision generation is currently unavailable. Please try again later."
            )
    
    async def poll(self, full_job_id: str) -> UnifiedJobStatus:
        """Poll job status across providers"""
        try:
            provider_name, job_id = full_job_id.split(":", 1)
            provider = self.providers.get(provider_name)
            
            if not provider:
                return UnifiedJobStatus(
                    status=UnifiedStatus.FAILED,
                    progress_percent=0,
                    phase=f"Provider {provider_name} not available"
                )
            
            return await provider.poll(job_id)
            
        except ValueError:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percent=0,
                phase="Invalid job ID format"
            )
        except Exception as e:
            self.logger.error(f"Failed to poll job {full_job_id}: {e}")
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percent=0,
                phase=f"Polling error: {e}"
            )
    
    async def fetch_result(self, full_job_id: str) -> UnifiedResult:
        """Fetch final result from provider"""
        try:
            provider_name, job_id = full_job_id.split(":", 1)
            provider = self.providers.get(provider_name)
            
            if not provider:
                raise VisionError(f"Provider {provider_name} not available", VisionErrorType.PROVIDER_ERROR)
            
            return await provider.fetch_result(job_id)
            
        except ValueError:
            raise VisionError("Invalid job ID format", VisionErrorType.INVALID_REQUEST)
        except Exception as e:
            self.logger.error(f"Failed to fetch result for {full_job_id}: {e}")
            raise VisionError(f"Result fetch failed: {e}", VisionErrorType.PROVIDER_ERROR)
    
    async def cancel(self, full_job_id: str) -> bool:
        """Cancel job if provider supports it"""
        try:
            provider_name, job_id = full_job_id.split(":", 1)
            provider = self.providers.get(provider_name)
            
            if provider:
                return await provider.cancel(job_id)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job {full_job_id}: {e}")
            return False
    
    def get_supported_tasks(self) -> List[VisionTask]:
        """Get list of all supported tasks across providers [CA]"""
        tasks = set()
        for provider in self.providers.values():
            capabilities = provider.capabilities()
            tasks.update(capabilities.get("modes", []))
        return list(tasks)
    
    def get_providers_for_task(self, task: VisionTask) -> List[VisionProvider]:
        """Get available providers that support specific task [CA]"""
        providers = []
        for provider_name, provider in self.providers.items():
            capabilities = provider.capabilities()
            if task in capabilities.get("modes", []):
                # Convert provider name to VisionProvider enum
                try:
                    providers.append(VisionProvider(provider_name.upper()))
                except ValueError:
                    self.logger.warning(f"Unknown provider enum for {provider_name}")
        return providers
    
    def get_models_for_task(self, task: VisionTask, provider: Optional[VisionProvider] = None) -> List[str]:
        """Get available models for task, optionally filtered by provider [CA]"""
        models = []
        
        for provider_name, provider_plugin in self.providers.items():
            # Filter by provider if specified
            if provider and provider_name != provider.value.lower():
                continue
            
            capabilities = provider_plugin.capabilities()
            if task in capabilities.get("modes", []):
                # Get model from provider's model mapping
                if hasattr(provider_plugin, 'model_map'):
                    model = provider_plugin.model_map.get(task)
                    if model and model not in models:
                        models.append(model)
        
        return models
    
    def get_provider_capabilities(self, provider_name: str) -> Dict[str, Any]:
        """Get detailed capabilities for specific provider [PA]"""
        provider = self.providers.get(provider_name)
        if provider:
            return provider.capabilities()
        return {}
    
    def estimate_cost_for_request(self, request: VisionRequest) -> Dict[str, float]:
        """Get cost estimates from all providers for comparison [CMV]"""
        normalized = self.normalize_request(request)
        estimates = {}
        
        for provider_name, provider in self.providers.items():
            try:
                cost = self._estimate_cost(provider, normalized)
                estimates[provider_name] = cost
            except Exception as e:
                self.logger.warning(f"Cost estimation failed for {provider_name}: {e}")
                estimates[provider_name] = float('inf')
        
        return estimates


# Factory function for backward compatibility  
def create_vision_adapter(config: Dict[str, Any]) -> UnifiedVisionAdapter:
    """Create unified vision adapter with configuration [CA]"""
    return UnifiedVisionAdapter(config)
