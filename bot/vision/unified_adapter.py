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
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
import base64
from enum import Enum

from bot.util.logging import get_logger
from .types import (
    VisionRequest,
    VisionResponse,
    VisionProvider,
    VisionTask,
    VisionError,
    VisionErrorType,
)
from .money import Money
from .pricing_loader import get_pricing_table

logger = get_logger(__name__)


class ModelSelection(NamedTuple):
    """Model selection result from VISION_MODEL resolution [CMV]"""

    provider: str
    endpoint: str
    model_hint: str
    supports_advanced: bool  # steps, guidance, etc.


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
    progress_percentage: int = 0
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
            VisionTask.VIDEO_GENERATION: "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modes": [
                VisionTask.TEXT_TO_IMAGE,
                VisionTask.IMAGE_TO_IMAGE,
                VisionTask.VIDEO_GENERATION,
            ],
            "max_size": (1536, 1536),
            "max_steps": 50,
            "supports_negative_prompt": True,
            "supports_batch": False,
            "nsfw_policy": "blocked",
            "video_max_seconds": 4,
        }

    async def submit(self, request: NormalizedRequest) -> str:
        """Submit to Together.ai API with unified error handling [REH]"""
        model = self.model_map.get(request.task)
        if not model:
            raise VisionError(
                message=f"Task {request.task.value} not supported by Together.ai",
                error_type=VisionErrorType.UNSUPPORTED_TASK,
                user_message=f"Sorry, {request.task.value} is not supported by this provider.",
            )

        # Build request payload
        payload = {
            "model": model,
            "prompt": request.prompt,
            "width": min(request.width, 1536),
            "height": min(request.height, 1536),
            "steps": min(request.steps, 50),
            "n": 1,
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
            "Content-Type": "application/json",
        }

        try:
            async with self.session.post(
                f"{self.base_url}/v1/images/generations", json=payload, headers=headers
            ) as response:
                error_text = await response.text()

                # Unified error handling with proper taxonomy [REH]
                if response.status == 400:
                    if (
                        "content policy" in error_text.lower()
                        or "safety" in error_text.lower()
                    ):
                        raise VisionError(
                            message=f"Content filtered: {error_text}",
                            error_type=VisionErrorType.CONTENT_FILTERED,
                            user_message="Your request was blocked by content safety filters. Please modify your prompt.",
                        )
                    elif (
                        "invalid" in error_text.lower()
                        or "malformed" in error_text.lower()
                    ):
                        raise VisionError(
                            message=f"Invalid request: {error_text}",
                            error_type=VisionErrorType.VALIDATION_ERROR,
                            user_message="There was an issue with your request parameters. Please check and try again.",
                        )
                elif response.status == 401:
                    raise VisionError(
                        message="Together.ai authentication failed",
                        error_type=VisionErrorType.AUTHENTICATION_ERROR,
                        user_message="Vision service authentication failed. Please contact support.",
                    )
                elif response.status == 429:
                    raise VisionError(
                        message="Together.ai rate limit exceeded",
                        error_type=VisionErrorType.RATE_LIMITED,
                        user_message="Too many requests. Please wait a moment and try again.",
                    )
                elif response.status == 402 or "quota" in error_text.lower():
                    raise VisionError(
                        message="Together.ai quota exceeded",
                        error_type=VisionErrorType.QUOTA_EXCEEDED,
                        user_message="Service quota exceeded. Please try again later.",
                    )
                elif response.status >= 500:
                    raise VisionError(
                        message=f"Together.ai server error: {error_text}",
                        error_type=VisionErrorType.SERVER_ERROR,
                        user_message="The vision service is temporarily unavailable. Please try again.",
                    )
                elif response.status != 200:
                    raise VisionError(
                        message=f"Together.ai error ({response.status}): {error_text}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="Vision generation failed. Please try again.",
                    )

                result = await response.json()

                # Together.ai returns results immediately for images
                job_id = f"together_{int(time.time() * 1000)}"

                # Store result for polling interface compatibility
                self._results = getattr(self, "_results", {})
                self._results[job_id] = {
                    "status": "completed",
                    "data": result.get("data", []),
                    "cost": self._calculate_cost(request),
                }

                return job_id

        except VisionError:
            raise
        except Exception as e:
            raise VisionError(
                message=f"Together.ai connection error: {str(e)}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Unable to connect to vision service. Please try again.",
            )

    async def poll(self, job_id: str) -> UnifiedJobStatus:
        """Poll job status (Together.ai is typically immediate)"""
        results = getattr(self, "_results", {})

        if job_id not in results:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percentage=0,
                phase="Job not found",
            )

        job_data = results[job_id]

        if job_data["status"] == "completed":
            return UnifiedJobStatus(
                status=UnifiedStatus.COMPLETED,
                progress_percentage=100,
                phase="Generation complete",
                actual_cost=job_data["cost"],
            )
        else:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percentage=0,
                phase="Generation failed",
            )

    async def fetch_result(self, job_id: str) -> UnifiedResult:
        """Fetch final result from Together.ai"""
        results = getattr(self, "_results", {})
        job_data = results.get(job_id, {})

        assets = []
        for item in job_data.get("data", []):
            if "url" in item:
                assets.append(item["url"])

        return UnifiedResult(
            assets=assets,
            final_cost=job_data.get("cost", 0.0),
            provider_used="together",
            metadata={"model": "FLUX.1-schnell"},
        )

    def _calculate_cost(self, request: NormalizedRequest) -> float:
        """Calculate estimated cost for Together.ai"""
        # Together.ai pricing (example rates)
        base_cost = 0.02  # per image
        pixel_cost = (request.width * request.height) / (1024 * 1024) * 0.005
        return base_cost + pixel_cost


class NovitaPlugin(ProviderPlugin):
    """Novita.ai provider plugin with Qwen-Image and SDXL support"""

    def __init__(self, name: str, config: Dict[str, Any], api_key: str):
        super().__init__(name, config, api_key)
        self.base_url = config.get("base_url", "https://api.novita.ai")
        self.model_map = {
            VisionTask.TEXT_TO_IMAGE: "sd_xl_base_1.0.safetensors",
            VisionTask.IMAGE_TO_IMAGE: "sd_xl_base_1.0.safetensors",
            VisionTask.VIDEO_GENERATION: "stable-video-diffusion-img2vid-xt",
        }

        # Endpoint-specific configurations [CMV]
        self.endpoints = {
            "qwen-image-txt2img": {
                "path": "/v3/async/qwen-image-txt2img",
                "supports_advanced": False,
                "max_size": (1536, 1536),
                "size_format": "WxH",  # "1024*1024" format
            },
            "txt2img": {
                "path": "/v3/async/txt2img",
                "supports_advanced": True,
                "max_size": (2048, 2048),
                "size_format": "separate",  # width/height fields
            },
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modes": [
                VisionTask.TEXT_TO_IMAGE,
                VisionTask.IMAGE_TO_IMAGE,
                VisionTask.VIDEO_GENERATION,
            ],
            "max_size": (2048, 2048),
            "max_steps": 60,
            "supports_negative_prompt": True,
            "supports_batch": True,
            "nsfw_policy": "filtered",
            "video_max_seconds": 6,
        }

    def normalize_size_for_endpoint(
        self, endpoint: str, width: int, height: int
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Normalize size parameters for specific endpoint [IV]"""
        endpoint_config = self.endpoints[endpoint]
        max_w, max_h = endpoint_config["max_size"]
        warnings = []

        # Clamp to endpoint limits
        original_w, original_h = width, height
        if width > max_w or height > max_h:
            # Proportional downscale
            scale = min(max_w / width, max_h / height)
            width = int(width * scale)
            height = int(height * scale)
            warnings.append(
                f"Size downscaled from {original_w}x{original_h} to {width}x{height} for {endpoint} limits"
            )

        # Ensure minimum sizes
        width = max(256, width)
        height = max(256, height)

        # Align to multiples of 8 and clamp to endpoint maximums
        width = min(max_w, (width // 8) * 8)
        height = min(max_h, (height // 8) * 8)

        # Format according to endpoint requirements
        if endpoint_config["size_format"] == "WxH":
            return {"size": f"{width}*{height}"}, warnings
        else:
            return {"width": width, "height": height}, warnings

    def build_payload_for_endpoint(
        self, endpoint: str, request: NormalizedRequest
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Build request payload for specific Novita endpoint [CA]"""
        self.endpoints[endpoint]
        warnings = []

        # Base payload with prompt
        payload = {"prompt": request.prompt.strip() if request.prompt else ""}

        # Add size parameters
        size_params, size_warnings = self.normalize_size_for_endpoint(
            endpoint, request.width, request.height
        )
        payload.update(size_params)
        warnings.extend(size_warnings)

        if endpoint == "qwen-image-txt2img":
            # Qwen endpoint: minimal parameters [CMV]
            if request.negative_prompt:
                warnings.append(
                    "Negative prompt not supported by Qwen-Image endpoint, ignoring"
                )
            if request.steps != 20:
                warnings.append(
                    "Custom steps not supported by Qwen-Image endpoint, using default"
                )
            if request.guidance_scale != 7.5:
                warnings.append(
                    "Custom guidance scale not supported by Qwen-Image endpoint, using default"
                )
            if request.seed:
                warnings.append(
                    "Custom seed not supported by Qwen-Image endpoint, ignoring"
                )
        elif endpoint == "txt2img":
            # SDXL endpoint: full parameter support [REH]
            payload.update(
                {
                    "model_name": request.preferred_model
                    or "sd_xl_base_1.0.safetensors",
                    "steps": max(1, min(100, request.steps)),
                    "guidance_scale": max(1.0, min(30.0, request.guidance_scale)),
                    "batch_size": 1,
                }
            )

            if request.negative_prompt:
                payload["negative_prompt"] = request.negative_prompt

            if request.seed:
                payload["seed"] = request.seed

            # NSFW detection (optional, billable)
            if request.safety_mode == "detect":
                payload["extra"] = {
                    "enable_nsfw_detection": True,
                    "nsfw_detection_level": 2,
                }

            if request.input_image_data:
                b64_image = base64.b64encode(request.input_image_data).decode()
                payload["init_image"] = b64_image

        return payload, warnings

    def _normalize_payload_for_novita(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sanitize/normalize payload before sending to Novita [IV][REH].
        - Drops None/empty strings
        - Clamps steps 1..100, guidance 1..20
        - Coerces seed to int
        - Rounds sizes to multiples of 8 (>=256) and within endpoint max
        - Converts list negative_prompt to comma-joined string
        - Removes keys outside the endpoint allowlist
        """
        allowed_qwen = {"prompt", "size"}
        allowed_txt2img = {
            "prompt",
            "model_name",
            "steps",
            "guidance_scale",
            "width",
            "height",
            "negative_prompt",
            "seed",
            "batch_size",
            "extra",
            "init_image",
        }
        allowed = allowed_qwen if endpoint == "qwen-image-txt2img" else allowed_txt2img

        def _clean_ok(v: Any) -> bool:
            return v is not None and not (isinstance(v, str) and v.strip() == "")

        norm = {k: v for k, v in payload.items() if _clean_ok(v)}

        # Prompt whitespace normalization and 2000-rune clamp [IV]
        if "prompt" in norm:
            p = norm.get("prompt")
            # Coerce non-str to str defensively
            prompt_clean = " ".join(str(p if p is not None else "").split()).strip()
            orig_len = len(prompt_clean)
            if orig_len == 0:
                raise VisionError(
                    message="Prompt is empty after normalization",
                    error_type=VisionErrorType.VALIDATION_ERROR,
                    user_message="A prompt is required.",
                )
            if orig_len > 2000:
                logger.debug(f"Novita prompt auto-truncated from {orig_len} to 2000")
                prompt_clean = prompt_clean[:2000]
            norm["prompt"] = prompt_clean

        # negative_prompt as CSV
        if isinstance(norm.get("negative_prompt"), list):
            norm["negative_prompt"] = ", ".join(
                str(x).strip() for x in norm["negative_prompt"] if str(x).strip()
            )

        # steps clamp
        if "steps" in norm:
            try:
                norm["steps"] = max(1, min(int(norm["steps"]), 100))
            except Exception:
                norm.pop("steps", None)

        # guidance_scale clamp
        if "guidance_scale" in norm:
            try:
                norm["guidance_scale"] = max(
                    1.0, min(float(norm["guidance_scale"]), 20.0)
                )
            except Exception:
                norm.pop("guidance_scale", None)

        # seed coercion
        if "seed" in norm:
            try:
                norm["seed"] = int(norm["seed"])
            except Exception:
                norm.pop("seed", None)

        # Size rounding depending on endpoint format
        try:
            max_w, max_h = self.endpoints[endpoint]["max_size"]
            if (
                endpoint == "qwen-image-txt2img"
                and isinstance(norm.get("size"), str)
                and "*" in norm["size"]
            ):
                w_str, h_str = norm["size"].split("*", 1)
                w = min(max_w, (max(256, int(w_str)) // 8) * 8)
                h = min(max_h, (max(256, int(h_str)) // 8) * 8)
                norm["size"] = f"{w}*{h}"
            else:
                if "width" in norm:
                    w = min(max_w, (max(256, int(norm["width"])) // 8) * 8)
                    norm["width"] = w
                if "height" in norm:
                    h = min(max_h, (max(256, int(norm["height"])) // 8) * 8)
                    norm["height"] = h
        except Exception:
            # Best-effort; leave as-is if parsing fails
            pass

        # Strict allowlist
        norm = {k: v for k, v in norm.items() if k in allowed}

        # Debug without leaking prompt text
        debug_view = {
            k: ("…" if k in ("prompt", "negative_prompt") else v)
            for k, v in norm.items()
        }
        logger.debug(f"Novita normalized payload for {endpoint}: {debug_view}")
        return norm

    async def submit(
        self, request: NormalizedRequest, endpoint: str = "qwen-image-txt2img"
    ) -> str:
        """Submit to Novita.ai API with endpoint selection and unified error handling [REH]"""
        if endpoint not in self.endpoints:
            raise VisionError(
                message=f"Endpoint {endpoint} not supported by Novita.ai",
                error_type=VisionErrorType.UNSUPPORTED_TASK,
                user_message="Sorry, the requested model is not supported.",
            )

        endpoint_config = self.endpoints[endpoint]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build and sanitize payload
        payload, warnings = self.build_payload_for_endpoint(endpoint, request)
        payload = self._normalize_payload_for_novita(endpoint, payload)

        try:
            async with self.session.post(
                f"{self.base_url}{endpoint_config['path']}",
                json=payload,
                headers=headers,
            ) as response:
                error_text = await response.text()

                # Unified error handling with proper taxonomy [REH]
                if response.status == 400:
                    if "nsfw" in error_text.lower() or "content" in error_text.lower():
                        raise VisionError(
                            message=f"Content filtered: {error_text}",
                            error_type=VisionErrorType.CONTENT_FILTERED,
                            user_message="Your request was blocked by content safety filters. Please modify your prompt.",
                        )
                    elif (
                        "invalid" in error_text.lower()
                        or "parameter" in error_text.lower()
                    ):
                        raise VisionError(
                            message=f"Invalid request: {error_text}",
                            error_type=VisionErrorType.VALIDATION_ERROR,
                            user_message="There was an issue with your request parameters. Please check and try again.",
                        )
                    elif (
                        "prompt: value length must be between 1 and 2000"
                        in error_text.lower()
                    ):
                        raise VisionError(
                            message=f"Invalid request: {error_text}",
                            error_type=VisionErrorType.VALIDATION_ERROR,
                            user_message="Your prompt is too long for this provider. I've trimmed it and will retry.",
                        )
                elif response.status == 401:
                    raise VisionError(
                        message="Novita.ai authentication failed",
                        error_type=VisionErrorType.AUTHENTICATION_ERROR,
                        user_message="Vision service authentication failed. Please contact support.",
                    )
                elif response.status == 429:
                    raise VisionError(
                        message="Novita.ai rate limit exceeded",
                        error_type=VisionErrorType.RATE_LIMITED,
                        user_message="Too many requests. Please wait a moment and try again.",
                    )
                elif response.status == 402 or "credit" in error_text.lower():
                    raise VisionError(
                        message="Novita.ai quota exceeded",
                        error_type=VisionErrorType.QUOTA_EXCEEDED,
                        user_message="Service quota exceeded. Please try again later.",
                    )
                elif response.status >= 500:
                    raise VisionError(
                        message=f"Novita.ai server error: {error_text}",
                        error_type=VisionErrorType.SERVER_ERROR,
                        user_message="The vision service is temporarily unavailable. Please try again.",
                    )
                elif response.status != 200:
                    raise VisionError(
                        message=f"Novita.ai error ({response.status}): {error_text}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="Vision generation failed. Please try again.",
                    )

                result = await response.json()
                job_id = result.get("task_id", f"novita_{int(time.time() * 1000)}")

                # Store job for tracking
                self._jobs = getattr(self, "_jobs", {})
                self._jobs[job_id] = {
                    "status": "running",
                    "progress": 0,
                    "start_time": time.time(),
                    "cost": self._calculate_cost(request, endpoint),
                    "result": None,
                    "endpoint": endpoint,
                }

                return job_id

        except VisionError:
            raise
        except Exception as e:
            raise VisionError(
                message=f"Novita.ai connection error: {str(e)}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Unable to connect to vision service. Please try again.",
            )

    async def poll(self, job_id: str) -> UnifiedJobStatus:
        """Poll Novita.ai job status via async task-result endpoint [PA][REH]"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with self.session.get(
                f"{self.base_url}/v3/async/task-result",
                params={"task_id": job_id},
                headers=headers,
            ) as response:
                # Return unified errors on HTTP failure
                if response.status == 404:
                    return UnifiedJobStatus(
                        status=UnifiedStatus.FAILED,
                        progress_percentage=0,
                        phase="Job not found (404)",
                    )
                if response.status >= 500:
                    return UnifiedJobStatus(
                        status=UnifiedStatus.RUNNING,
                        progress_percentage=0,
                        phase=f"Provider error {response.status}",
                    )
                if response.status != 200:
                    text = await response.text()
                    return UnifiedJobStatus(
                        status=UnifiedStatus.RUNNING,
                        progress_percentage=0,
                        phase=f"Polling error {response.status}: {text[:120]}",
                    )
                data = await response.json()
                task = data.get("task", {})
                status = task.get("status", "TASK_STATUS_PENDING")
                progress = int(task.get("progress", 0) or 0)

                # Map provider status → unified
                if status == "TASK_STATUS_SUCCEED":
                    unified = UnifiedStatus.COMPLETED
                elif status in ("TASK_STATUS_FAILED", "TASK_STATUS_CANCELED"):
                    unified = (
                        UnifiedStatus.FAILED
                        if status == "TASK_STATUS_FAILED"
                        else UnifiedStatus.CANCELLED
                    )
                elif status in (
                    "TASK_STATUS_PROCESSING",
                    "TASK_STATUS_QUEUED",
                    "TASK_STATUS_RUNNING",
                    "TASK_STATUS_PENDING",
                ):
                    unified = UnifiedStatus.RUNNING
                else:
                    unified = UnifiedStatus.RUNNING

                # Update local cache if present
                jobs = getattr(self, "_jobs", {})
                if job_id in jobs:
                    jobs[job_id]["status"] = status
                    jobs[job_id]["progress"] = progress
                    if unified in (
                        UnifiedStatus.COMPLETED,
                        UnifiedStatus.FAILED,
                        UnifiedStatus.CANCELLED,
                    ):
                        jobs[job_id]["result"] = data

                return UnifiedJobStatus(
                    status=unified,
                    progress_percentage=progress,
                    phase=f"Novita: {status.replace('TASK_STATUS_', '').lower()}",
                    provider_raw=data,
                )
        except Exception as e:
            return UnifiedJobStatus(
                status=UnifiedStatus.RUNNING,
                progress_percentage=0,
                phase=f"Polling exception: {e}",
            )

    async def fetch_result(self, job_id: str) -> UnifiedResult:
        """Fetch final result from Novita.ai async task-result API [REH]"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with self.session.get(
            f"{self.base_url}/v3/async/task-result",
            params={"task_id": job_id},
            headers=headers,
        ) as response:
            text = await response.text()
            if response.status != 200:
                raise VisionError(
                    message=f"Novita result fetch failed ({response.status}): {text}",
                    error_type=VisionErrorType.PROVIDER_ERROR,
                    user_message="Failed to retrieve generated image. Please try again.",
                )
            data = json.loads(text)
            task = data.get("task", {})
            status = task.get("status")
            if status != "TASK_STATUS_SUCCEED":
                # If not yet complete, surface as retryable via caller's loop
                raise VisionError(
                    message=f"Result not ready: {status}",
                    error_type=VisionErrorType.TIMEOUT_ERROR,
                    user_message="The image is still being generated.",
                )
            # Extract asset URLs
            assets: List[str] = []
            images = data.get("images", []) or data.get("data", [])
            for item in images:
                url = (
                    item.get("image_url") or item.get("url") or item.get("download_url")
                )
                if url:
                    assets.append(url)
            if not assets:
                raise VisionError(
                    message="No assets found in Novita response",
                    error_type=VisionErrorType.PROVIDER_ERROR,
                    user_message="No images were returned by the provider.",
                )
            # Cost from local cache if available
            jobs = getattr(self, "_jobs", {})
            cost = jobs.get(job_id, {}).get(
                "cost",
                self._calculate_cost(
                    NormalizedRequest(
                        task=VisionTask.TEXT_TO_IMAGE,
                        prompt="",
                    ),
                    jobs.get(job_id, {}).get("endpoint", "qwen-image-txt2img"),
                ),
            )
            # Augment cache with final data
            if job_id in jobs:
                jobs[job_id]["result"] = data
                jobs[job_id]["status"] = status
                jobs[job_id]["progress"] = 100
            return UnifiedResult(
                assets=assets,
                final_cost=cost,
                provider_used="novita",
                metadata={"task": task},
            )

    def _calculate_cost(
        self, request: NormalizedRequest, endpoint: str = "qwen-image-txt2img"
    ) -> float:
        """Calculate estimated cost for Novita.ai using endpoint-specific pricing [PA][CMV]"""
        # Defaults if config missing
        base_cost = 0.018
        per_px = 0.000004
        # Provider config may have per-endpoint pricing
        provider_cfg = self.config or {}
        endpoints_cfg = provider_cfg.get("endpoints", {})
        # Map unified endpoint key used in code to config alias
        cfg_key = "qwen-image" if endpoint == "qwen-image-txt2img" else "txt2img"
        pricing = endpoints_cfg.get(cfg_key, {}).get("price", {})
        base_cost = pricing.get("image_base", base_cost)
        per_px = pricing.get("image_per_px", per_px)
        pixels = max(1, request.width * request.height)
        return float(base_cost + (pixels * per_px))


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

        # Model override from environment [CMV]
        self.vision_model_override = config.get("VISION_MODEL", "").strip()
        if self.vision_model_override:
            self.logger.info(
                f"🎯 VISION_MODEL override active: {self.vision_model_override}"
            )

        # Load provider configuration
        self._load_provider_config()
        self._initialize_providers()

        # Initialize model resolver
        self._model_aliases = self._build_model_aliases()

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
                    "provider_order": [
                        "novita:qwen-image",
                        "novita:txt2img",
                        "together",
                    ],
                    "budget_per_job_usd": 0.25,
                    "prefer_model": {"image": "qwen-image", "video": "svd-xt"},
                    "nsfw_mode": "block",
                    "auto_fallback": True,
                    "max_retries_per_provider": 2,
                },
                "providers": [
                    {
                        "name": "together",
                        "base_url": "https://api.together.xyz",
                        "api_key_env": "VISION_API_KEY",
                        "enabled": True,
                        "priority": 1,
                        "price": {
                            "image_base": 0.02,
                            "image_per_px": 0.000005,
                            "video_per_s": 0.06,
                        },
                        "limits": {"max_size": "1536x1536", "max_steps": 50},
                    },
                    {
                        "name": "novita",
                        "base_url": "https://api.novita.ai",
                        "api_key_env": "VISION_API_KEY",
                        "enabled": True,
                        "priority": 2,
                        "price": {
                            "image_base": 0.018,
                            "image_per_px": 0.000004,
                            "video_per_s": 0.05,
                        },
                        "limits": {"max_size": "2048x2048", "max_steps": 60},
                        "models": {
                            "qwen-image": {
                                "endpoint": "qwen-image-txt2img",
                                "price": {
                                    "image_base": 0.015,
                                    "image_per_px": 0.000003,
                                },
                                "limits": {
                                    "max_size": "1536x1536",
                                    "supports_advanced": False,
                                },
                            },
                            "txt2img": {
                                "endpoint": "txt2img",
                                "price": {
                                    "image_base": 0.018,
                                    "image_per_px": 0.000004,
                                },
                                "limits": {
                                    "max_size": "2048x2048",
                                    "supports_advanced": True,
                                },
                            },
                        },
                    },
                ],
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

    def _has_valid_credentials(self, provider_name: str) -> bool:
        """Best-effort credential presence check per provider [SFT].
        Conservative: Together requires a provider-specific key; Novita can use NOVITA_API_KEY or fallback VISION_API_KEY.
        """
        try:
            name = provider_name.split(":")[0].lower()
            if name == "novita":
                key = (
                    self.config.get("NOVITA_API_KEY")
                    or self.config.get("VISION_API_KEY")
                    or ""
                )
            elif name == "together":
                key = (
                    self.config.get("TOGETHER_API_KEY")
                    or self.config.get("VISION_API_KEY_TOGETHER")
                    or ""
                )
            else:
                key = self.config.get("VISION_API_KEY", "")
            return isinstance(key, str) and len(key.strip()) > 10
        except Exception:
            return False

    def _is_provider_healthy(self, provider_name: str) -> bool:
        """Lightweight health gate. Currently mirrors credential presence/shape [PA]."""
        return self._has_valid_credentials(provider_name)

    def _estimate_cost_money(
        self, provider_name: str, request: NormalizedRequest
    ) -> Optional[Money]:
        """Best-effort Money estimate using pricing table [PA][CMV].
        Returns None if pricing table does not cover this provider/task.
        """
        try:
            table = get_pricing_table()
            # Normalize provider string and strip endpoint suffix if present [IV]
            normalized_provider = provider_name.split(":")[0].lower()
            provider_enum = VisionProvider(normalized_provider)
            # Derive model if available on normalized request
            model = getattr(request, "preferred_model", None)
            money_est = table.estimate_cost(
                provider=provider_enum,
                task=request.task,
                width=request.width,
                height=request.height,
                num_images=getattr(request, "batch_size", 1) or 1,
                duration_seconds=getattr(request, "video_seconds", None) or 4.0,
                model=model,
            )
            # Ensure Money type
            return money_est if isinstance(money_est, Money) else Money(money_est)
        except Exception:
            return None

    def _build_model_aliases(self) -> Dict[str, ModelSelection]:
        """Build model alias mapping for VISION_MODEL resolution [CMV]"""
        aliases = {}

        # Novita Qwen-Image aliases
        for alias in [
            "novita:qwen-image",
            "qwen-image",
            "novita-qwen-image",
            "qwen_image",
        ]:
            aliases[alias] = ModelSelection(
                provider="novita",
                endpoint="qwen-image-txt2img",
                model_hint="qwen-image",
                supports_advanced=False,
            )

        # Novita SDXL aliases
        for alias in ["novita:txt2img:sdxl", "novita:sdxl", "sdxl"]:
            aliases[alias] = ModelSelection(
                provider="novita",
                endpoint="txt2img",
                model_hint="sd_xl_base_1.0.safetensors",
                supports_advanced=True,
            )

        # Together.ai aliases
        for alias in ["together:flux.1-pro", "flux.1-pro", "together"]:
            aliases[alias] = ModelSelection(
                provider="together",
                endpoint="images/generations",
                model_hint="black-forest-labs/FLUX.1-schnell-Free",
                supports_advanced=True,
            )

        return aliases

    def resolve_model_selection(
        self, request: NormalizedRequest
    ) -> Optional[ModelSelection]:
        """Resolve VISION_MODEL override to specific provider/endpoint [CA]"""
        if not self.vision_model_override:
            return None

        # Direct alias lookup
        selection = self._model_aliases.get(self.vision_model_override.lower())
        if selection:
            self.logger.info(
                f" Resolved VISION_MODEL '{self.vision_model_override}' → {selection.provider}:{selection.endpoint}"
            )
            return selection

        # Parse provider:model format
        if ":" in self.vision_model_override:
            parts = self.vision_model_override.split(":", 1)
            provider_name = parts[0].lower()

            if provider_name in self.providers:
                # Default to provider's primary endpoint
                if provider_name == "novita":
                    return ModelSelection(
                        provider="novita",
                        endpoint="qwen-image-txt2img",  # Default to Qwen
                        model_hint="qwen-image",
                        supports_advanced=False,
                    )
                elif provider_name == "together":
                    return ModelSelection(
                        provider="together",
                        endpoint="images/generations",
                        model_hint="black-forest-labs/FLUX.1-schnell-Free",
                        supports_advanced=True,
                    )

        self.logger.warning(
            f" Unrecognized VISION_MODEL '{self.vision_model_override}', falling back to policy"
        )
        return None

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
            safety_mode="strict",
        )

    def select_provider(self, request: NormalizedRequest) -> Optional[ProviderPlugin]:
        """Select best provider for request based on capabilities and policy [CA]"""
        policy = self.provider_config["vision"]["default_policy"]
        provider_order = policy.get("provider_order", ["together", "novita"])

        for provider_name in provider_order:
            # Support provider entries that include endpoint alias e.g. "novita:qwen-image"
            provider_key = provider_name.split(":")[0].lower()
            if provider_key not in self.providers:
                continue

            provider = self.providers[provider_key]

            # Check if provider supports the task
            capabilities = provider.capabilities()
            if request.task not in capabilities.get("modes", []):
                continue

            # Check size limits
            max_w, max_h = capabilities.get("max_size", (1024, 1024))
            if request.width > max_w or request.height > max_h:
                continue

            # Estimate cost and check budget
            try:
                estimated_cost = self._estimate_cost(provider, request)
                money_est = self._estimate_cost_money(provider_name, request)
                budget_limit = policy.get("budget_per_job_usd", 0.25)

                if estimated_cost > budget_limit:
                    self.logger.warning(
                        f"Provider {provider_key} cost ${estimated_cost:.3f} exceeds budget ${budget_limit:.3f}"
                    )
                    continue

                if money_est is not None:
                    self.logger.info(
                        f"Selected provider: {provider_key} (cost: ${estimated_cost:.3f}, money={money_est})"
                    )
                else:
                    self.logger.info(
                        f"Selected provider: {provider_key} (cost: ${estimated_cost:.3f})"
                    )
                return provider

            except Exception as e:
                self.logger.warning(f"Cost estimation failed for {provider_key}: {e}")
                continue

        return None

    def _estimate_cost(
        self, provider: ProviderPlugin, request: NormalizedRequest
    ) -> float:
        """Estimate cost for provider and request [PA]"""
        config = next(
            (
                p
                for p in self.provider_config["vision"]["providers"]
                if p["name"] == provider.name
            ),
            {},
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

    async def submit(self, request: VisionRequest) -> VisionResponse:
        """Submit vision request with VISION_MODEL override and automatic provider selection [REH]"""
        policy = self.provider_config.get("vision", {}).get("default_policy", {})
        normalized_request = self.normalize_request(request)

        # Check for VISION_MODEL override
        model_selection = self.resolve_model_selection(normalized_request)

        if model_selection:
            # Use pinned model/provider/endpoint
            provider_order = [model_selection.provider]
            forced_endpoint = model_selection.endpoint
            self.logger.info(
                f"🎯 Using VISION_MODEL override: {model_selection.provider}:{model_selection.endpoint}"
            )
        else:
            # Use policy-driven selection
            provider_order = policy.get(
                "provider_order", ["novita:qwen-image", "novita:txt2img", "together"]
            )
            forced_endpoint = None

            # Parse provider:endpoint format in policy
            resolved_order = []
            for entry in provider_order:
                if ":" in entry:
                    provider_name = entry.split(":")[0]
                    resolved_order.append(provider_name)
                else:
                    resolved_order.append(entry)
            provider_order = resolved_order

        # Filter by configured/healthy providers, preserving order [SFT]
        filtered_order: List[str] = []
        for name in provider_order:
            base = name.split(":")[0]
            if (
                self._has_valid_credentials(base)
                and self._is_provider_healthy(base)
                and base in self.providers
            ):
                if base not in filtered_order:
                    filtered_order.append(base)
        if not filtered_order:
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message="No configured/healthy vision providers available",
                user_message="Vision generation is not configured.",
            )
        self.logger.debug(f"Provider order (filtered): {filtered_order}")
        provider_order = filtered_order

        last_error = None

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]
            # Belt-and-braces: skip if not configured/healthy
            if not (
                self._has_valid_credentials(provider_name)
                and self._is_provider_healthy(provider_name)
            ):
                self.logger.debug(
                    f"Skipping provider {provider_name}: not configured/unhealthy"
                )
                continue

            # Check if provider supports the task
            capabilities = provider.capabilities()
            if normalized_request.task not in capabilities.get("modes", []):
                continue

            # Determine endpoint for this provider
            endpoint = None
            if (
                forced_endpoint
                and model_selection
                and provider_name == model_selection.provider
            ):
                endpoint = forced_endpoint
            elif provider_name == "novita":
                # Default endpoint selection for Novita based on policy order
                original_entry = next(
                    (
                        e
                        for e in policy.get("provider_order", [])
                        if e.startswith("novita")
                    ),
                    "novita:qwen-image",
                )
                if "qwen-image" in original_entry:
                    endpoint = "qwen-image-txt2img"
                else:
                    endpoint = "txt2img"

            # Attempt submission with retries
            did_prompt_retry = False  # Single-shot retry for Novita prompt length
            for attempt in range(policy.get("max_retries_per_provider", 2)):
                try:
                    # Submit with endpoint if supported (Novita), otherwise default
                    if (
                        hasattr(provider, "submit")
                        and endpoint
                        and provider_name == "novita"
                    ):
                        task_id = await provider.submit(normalized_request, endpoint)
                    else:
                        task_id = await provider.submit(normalized_request)

                    return VisionResponse(
                        success=True,
                        job_id=f"{provider_name}:{task_id}",
                        provider=VisionProvider(provider_name.lower()),
                        model_used=getattr(provider, "model_map", {}).get(
                            normalized_request.task, "unknown"
                        ),
                        provider_job_id=task_id,
                    )

                except VisionError as e:
                    last_error = e
                    # Special-case: Novita 400 prompt length error → one clean retry with 2000-char prompt [REH]
                    if (
                        provider_name == "novita"
                        and e.error_type == VisionErrorType.VALIDATION_ERROR
                        and "prompt: value length must be between 1 and 2000"
                        in (e.message or "").lower()
                    ):
                        if not did_prompt_retry:
                            # Normalize whitespace and clamp to 2000
                            clamped = " ".join(
                                (normalized_request.prompt or "").split()
                            ).strip()
                            if len(clamped) > 2000:
                                clamped = clamped[:2000]
                            if len(clamped) == 0:
                                # Nothing to retry with
                                self.logger.debug(
                                    "Novita prompt length retry skipped: prompt empty after normalization"
                                )
                            else:
                                normalized_request.prompt = clamped
                                self.logger.debug("Novita retry with 2000-char prompt")
                                did_prompt_retry = True
                                # Immediate retry without backoff
                                continue
                        # Already retried once; fall through to existing handling
                    if e.error_type in [
                        VisionErrorType.RATE_LIMITED,
                        VisionErrorType.PROVIDER_ERROR,
                    ]:
                        # Exponential backoff for retryable errors
                        wait_time = (2**attempt) * 1.0  # 1s, 2s, 4s...
                        self.logger.warning(
                            f"Retryable error from {provider_name} (attempt {attempt + 1}): {e.message}, retrying in {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Non-retryable error, try next provider
                        self.logger.warning(
                            f"Non-retryable error from {provider_name}: {e.message}"
                        )
                        break

                except Exception as e:
                    last_error = VisionError(
                        message=f"Unexpected error from {provider_name}: {str(e)}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="An unexpected error occurred during image generation.",
                    )
                    self.logger.error(
                        f"Unexpected error from {provider_name}: {e}", exc_info=True
                    )
                    break

        # All providers failed
        raise last_error or VisionError(
            message="All vision providers exhausted",
            error_type=VisionErrorType.PROVIDER_ERROR,
            user_message="Image generation is temporarily unavailable. Please try again later.",
        )

    async def poll(self, full_job_id: str) -> UnifiedJobStatus:
        """Poll job status across providers"""
        try:
            provider_name, job_id = full_job_id.split(":", 1)
            provider = self.providers.get(provider_name)

            if not provider:
                return UnifiedJobStatus(
                    status=UnifiedStatus.FAILED,
                    progress_percentage=0,
                    phase=f"Provider {provider_name} not available",
                )

            return await provider.poll(job_id)

        except ValueError:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percentage=0,
                phase="Invalid job ID format",
            )
        except Exception as e:
            self.logger.error(f"Failed to poll job {full_job_id}: {e}")
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percentage=0,
                phase=f"Polling error: {e}",
            )

    async def fetch_result(self, full_job_id: str) -> UnifiedResult:
        """Fetch final result from provider"""
        try:
            provider_name, job_id = full_job_id.split(":", 1)
            provider = self.providers.get(provider_name)

            if not provider:
                raise VisionError(
                    f"Provider {provider_name} not available",
                    VisionErrorType.PROVIDER_ERROR,
                )

            return await provider.fetch_result(job_id)

        except ValueError:
            raise VisionError("Invalid job ID format", VisionErrorType.VALIDATION_ERROR)
        except Exception as e:
            self.logger.error(f"Failed to fetch result for {full_job_id}: {e}")
            raise VisionError(
                f"Result fetch failed: {e}", VisionErrorType.PROVIDER_ERROR
            )

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
                    providers.append(VisionProvider(provider_name.lower()))
                except ValueError:
                    self.logger.warning(f"Unknown provider enum for {provider_name}")
        return providers

    def get_models_for_task(
        self, task: VisionTask, provider: Optional[VisionProvider] = None
    ) -> List[str]:
        """Get available models for task, optionally filtered by provider [CA]"""
        models = []

        for provider_name, provider_plugin in self.providers.items():
            # Filter by provider if specified
            if provider and provider_name != provider.value.lower():
                continue

            capabilities = provider_plugin.capabilities()
            if task in capabilities.get("modes", []):
                # Get model from provider's model mapping
                if hasattr(provider_plugin, "model_map"):
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
                estimates[provider_name] = float("inf")

        return estimates


# Factory function for backward compatibility
def create_vision_adapter(config: Dict[str, Any]) -> UnifiedVisionAdapter:
    """Create unified vision adapter with configuration [CA]"""
    return UnifiedVisionAdapter(config)
