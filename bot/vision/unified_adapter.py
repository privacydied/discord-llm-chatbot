"""Unified Vision Provider Adapter.

Single adapter with pluggable provider system for vision generation.
Normalizes requests/responses and handles provider selection, fallback, and error mapping.
"""

import asyncio
import base64
import contextlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import aiohttp

from bot.utils.logging import get_logger

from .money import Money
from .pricing_loader import get_pricing_table
from .types import (
    VisionError,
    VisionErrorType,
    VisionProvider,
    VisionRequest,
    VisionResponse,
    VisionTask,
)

logger = get_logger(__name__)


class ModelSelection(NamedTuple):
    """Model selection result from VISION_MODEL resolution [CMV]."""

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
    """Normalized request format for all providers."""

    task: VisionTask
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance_scale: float = 7.5
    seed: int | None = None
    input_image_data: bytes | None = None
    input_image_url: str | None = None
    video_seconds: int | None = None
    fps: int | None = None
    batch_size: int = 1
    safety_mode: str = "strict"
    preferred_model: str | None = None


@dataclass
class UnifiedJobStatus:
    """Unified job status across all providers."""

    status: UnifiedStatus
    progress_percentage: int = 0
    phase: str = ""
    estimated_cost: float | None = None
    actual_cost: float | None = None
    warnings: list[str] = field(default_factory=list)
    provider_raw: dict | None = None


@dataclass
class UnifiedResult:
    """Unified result format."""

    assets: list[str] = field(default_factory=list)  # URLs or file paths
    metadata: dict[str, Any] = field(default_factory=dict)
    final_cost: float = 0.0
    provider_used: str = ""
    warnings: list[str] = field(default_factory=list)


# Provider phrases that mean "this account cannot pay for the job". Novita answers
# 403 NOT_ENOUGH_BALANCE (not 402, and the word is "balance", not "credit"), which
# used to be filed as a generic PROVIDER_ERROR -- indistinguishable from a real
# outage, so the user was told to "try again" forever. [CMV][REH]
_BALANCE_ERROR_MARKERS: tuple[str, ...] = (
    "not_enough_balance",
    "not enough balance",
    "insufficient balance",
    "insufficient_balance",
    "insufficient funds",
    "out of credit",
    "credit",
    "quota",
    "billing",
)
# Statuses a provider uses to report a payment problem rather than a rejection.
_BALANCE_ERROR_STATUSES: frozenset[int] = frozenset({402, 403})


# How long a provider stays benched after it reports an empty account. Re-asking a
# provider with no balance costs a budget reservation, a round trip and a user-facing
# failure on every single job. Env: VISION_PROVIDER_QUOTA_COOLDOWN_S. [CMV][PA]
DEFAULT_QUOTA_COOLDOWN_S = 900.0

# Novita endpoint that accepts an init_image. The qwen-image endpoint only allows
# {prompt, size}, so it can never serve an edit. [CMV]
NOVITA_IMAGE_TO_IMAGE_ENDPOINT = "txt2img"

# Per-request ceiling for provider HTTP calls. VISION_PROVIDER_TIMEOUT_MS used to be
# parsed from .env and then ignored here, so every provider session ran with a 300s
# total timeout: a model that hangs (NVIDIA's flux.1-schnell answers nothing at all)
# held a job for five minutes instead of failing over. [CMV][REH][PA]
DEFAULT_PROVIDER_TIMEOUT_S = 300.0
MIN_PROVIDER_TIMEOUT_S = 10.0


def _is_balance_error(status: int, error_text: str) -> bool:
    """True when this response means "top up the account", not "retry later". [IV]"""
    if status == 402:
        return True
    lowered = (error_text or "").lower()
    return status in _BALANCE_ERROR_STATUSES and any(marker in lowered for marker in _BALANCE_ERROR_MARKERS)


def _resolve_openrouter_api_key(config: dict[str, Any]) -> str:
    """Best-effort OpenRouter key resolution over the adapter's own config dict.

    A dedicated ``OPENROUTER_API_KEY`` first; otherwise ``OPENAI_API_KEY`` or
    ``VISION_API_KEY`` when their paired ``_BASE`` var points at
    openrouter.ai -- the common shape where OpenRouter doubles as the text
    backend, so its real key lives under ``OPENAI_API_KEY`` rather than a
    provider-specific var. A key belonging to a different provider is never
    returned. [SFT]

    Mirrors ``bot.vision.free_model_probe.resolve_openrouter_key()``'s
    priority order but reads the passed-in ``config`` dict (falling back to
    ``os.environ``) rather than a fresh global ``load_config()`` call, so
    callers that construct ``UnifiedVisionAdapter`` with an explicit config
    dict (e.g. tests) aren't silently overridden by unrelated env state.
    """
    direct = str(config.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "").strip()
    if direct:
        return direct
    for key_name, base_name in (("OPENAI_API_KEY", "OPENAI_API_BASE"), ("VISION_API_KEY", "VISION_API_BASE")):
        base = str(config.get(base_name) or os.getenv(base_name) or "").lower()
        candidate = str(config.get(key_name) or os.getenv(key_name) or "").strip()
        if candidate and "openrouter" in base:
            return candidate
    return ""


class ProviderPlugin(ABC):
    """Abstract base class for provider plugins."""

    def __init__(self, name: str, config: dict[str, Any], api_key: str) -> None:
        self.name = name
        self.config = config
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return provider capabilities and limits."""

    @abstractmethod
    async def submit(self, request: NormalizedRequest) -> str:
        """Submit job and return job_id."""

    @abstractmethod
    async def poll(self, job_id: str) -> UnifiedJobStatus:
        """Poll job status."""

    @abstractmethod
    async def fetch_result(self, job_id: str) -> UnifiedResult:
        """Fetch final result."""

    async def cancel(self, job_id: str) -> bool:
        """Cancel job (optional)."""
        return False

    def session_timeout_s(self) -> float:
        """Total per-request timeout, from VISION_PROVIDER_TIMEOUT_MS. [CMV][REH]"""
        raw = self.config.get("timeout_ms")
        if raw in (None, ""):
            return DEFAULT_PROVIDER_TIMEOUT_S
        try:
            seconds = float(raw) / 1000.0
        except (TypeError, ValueError):
            return DEFAULT_PROVIDER_TIMEOUT_S
        if seconds <= 0:
            return DEFAULT_PROVIDER_TIMEOUT_S
        return max(MIN_PROVIDER_TIMEOUT_S, seconds)

    async def startup(self) -> None:
        """Initialize provider connection."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.session_timeout_s())
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def shutdown(self) -> None:
        """Cleanup provider connection."""
        if self.session:
            await self.session.close()
            self.session = None


class TogetherPlugin(ProviderPlugin):
    """Together.ai provider plugin."""

    def __init__(self, name: str, config: dict[str, Any], api_key: str) -> None:
        super().__init__(name, config, api_key)
        self.base_url = config.get("base_url", "https://api.together.xyz")
        self.model_map = {
            VisionTask.TEXT_TO_IMAGE: "black-forest-labs/FLUX.1-schnell-Free",
            VisionTask.IMAGE_TO_IMAGE: "black-forest-labs/FLUX.1-schnell-Free",
            VisionTask.VIDEO_GENERATION: "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        }

    def capabilities(self) -> dict[str, Any]:
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
        """Submit to Together.ai API with unified error handling [REH]."""
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
            async with self.session.post(f"{self.base_url}/v1/images/generations", json=payload, headers=headers) as response:
                error_text = await response.text()

                # Unified error handling with proper taxonomy [REH]
                if response.status == 400:
                    if "content policy" in error_text.lower() or "safety" in error_text.lower():
                        raise VisionError(
                            message=f"Content filtered: {error_text}",
                            error_type=VisionErrorType.CONTENT_FILTERED,
                            user_message="Your request was blocked by content safety filters. Please modify your prompt.",
                        )
                    if "invalid" in error_text.lower() or "malformed" in error_text.lower():
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
                elif _is_balance_error(response.status, error_text):
                    raise VisionError(
                        message=f"Together.ai balance/quota exhausted ({response.status}): {error_text}",
                        error_type=VisionErrorType.QUOTA_EXCEEDED,
                        user_message="The image provider's account is out of credit, so image generation is unavailable right now.",
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
                message=f"Together.ai connection error: {e!s}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Unable to connect to vision service. Please try again.",
            )

    async def poll(self, job_id: str) -> UnifiedJobStatus:
        """Poll job status (Together.ai is typically immediate)."""
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
        return UnifiedJobStatus(
            status=UnifiedStatus.FAILED,
            progress_percentage=0,
            phase="Generation failed",
        )

    async def fetch_result(self, job_id: str) -> UnifiedResult:
        """Fetch final result from Together.ai."""
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
        """Calculate estimated cost for Together.ai."""
        # Together.ai pricing (example rates)
        base_cost = 0.02  # per image
        pixel_cost = (request.width * request.height) / (1024 * 1024) * 0.005
        return base_cost + pixel_cost


class NovitaPlugin(ProviderPlugin):
    """Novita.ai provider plugin with Qwen-Image and SDXL support."""

    def __init__(self, name: str, config: dict[str, Any], api_key: str) -> None:
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

    def capabilities(self) -> dict[str, Any]:
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

    def normalize_size_for_endpoint(self, endpoint: str, width: int, height: int) -> tuple[dict[str, Any], list[str]]:
        """Normalize size parameters for specific endpoint [IV]."""
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
            warnings.append(f"Size downscaled from {original_w}x{original_h} to {width}x{height} for {endpoint} limits")

        # Ensure minimum sizes
        width = max(256, width)
        height = max(256, height)

        # Align to multiples of 8 and clamp to endpoint maximums
        width = min(max_w, (width // 8) * 8)
        height = min(max_h, (height // 8) * 8)

        # Format according to endpoint requirements
        if endpoint_config["size_format"] == "WxH":
            return {"size": f"{width}*{height}"}, warnings
        return {"width": width, "height": height}, warnings

    def build_payload_for_endpoint(self, endpoint: str, request: NormalizedRequest) -> tuple[dict[str, Any], list[str]]:
        """Build request payload for specific Novita endpoint [CA]."""
        self.endpoints[endpoint]
        warnings = []

        # Base payload with prompt
        payload = {"prompt": request.prompt.strip() if request.prompt else ""}

        # Add size parameters
        size_params, size_warnings = self.normalize_size_for_endpoint(endpoint, request.width, request.height)
        payload.update(size_params)
        warnings.extend(size_warnings)

        if endpoint == "qwen-image-txt2img":
            # Qwen endpoint: minimal parameters [CMV]
            if request.negative_prompt:
                warnings.append("Negative prompt not supported by Qwen-Image endpoint, ignoring")
            if request.steps != 20:
                warnings.append("Custom steps not supported by Qwen-Image endpoint, using default")
            if request.guidance_scale != 7.5:
                warnings.append("Custom guidance scale not supported by Qwen-Image endpoint, using default")
            if request.seed:
                warnings.append("Custom seed not supported by Qwen-Image endpoint, ignoring")
        elif endpoint == "txt2img":
            # SDXL endpoint: full parameter support [REH]
            payload.update(
                {
                    "model_name": request.preferred_model or "sd_xl_base_1.0.safetensors",
                    "steps": max(1, min(100, request.steps)),
                    "guidance_scale": max(1.0, min(30.0, request.guidance_scale)),
                    "batch_size": 1,
                },
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

    def _normalize_payload_for_novita(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Sanitize/normalize payload before sending to Novita [IV][REH].
        - Drops None/empty strings
        - Clamps steps 1..100, guidance 1..20
        - Coerces seed to int
        - Rounds sizes to multiples of 8 (>=256) and within endpoint max
        - Converts list negative_prompt to comma-joined string
        - Removes keys outside the endpoint allowlist.
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
            norm["negative_prompt"] = ", ".join(str(x).strip() for x in norm["negative_prompt"] if str(x).strip())

        # steps clamp
        if "steps" in norm:
            try:
                norm["steps"] = max(1, min(int(norm["steps"]), 100))
            except Exception as exc:
                logger.debug(f"Steps normalization failed: {exc}")
                norm.pop("steps", None)

        # guidance_scale clamp
        if "guidance_scale" in norm:
            try:
                norm["guidance_scale"] = max(1.0, min(float(norm["guidance_scale"]), 20.0))
            except Exception as exc:
                logger.debug(f"Guidance scale normalization failed: {exc}")
                norm.pop("guidance_scale", None)

        # seed coercion
        if "seed" in norm:
            try:
                norm["seed"] = int(norm["seed"])
            except Exception as exc:
                logger.debug(f"Seed normalization failed: {exc}")
                norm.pop("seed", None)

        # Size rounding depending on endpoint format
        try:
            max_w, max_h = self.endpoints[endpoint]["max_size"]
            if endpoint == "qwen-image-txt2img" and isinstance(norm.get("size"), str) and "*" in norm["size"]:
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
        except Exception as exc:
            # Best-effort; leave as-is if parsing fails
            logger.debug(f"Novita size normalization failed: {exc}")

        # Strict allowlist
        norm = {k: v for k, v in norm.items() if k in allowed}

        # Debug without leaking prompt text
        debug_view = {k: ("…" if k in ("prompt", "negative_prompt") else v) for k, v in norm.items()}
        logger.debug(f"Novita normalized payload for {endpoint}: {debug_view}")
        return norm

    async def submit(self, request: NormalizedRequest, endpoint: str = "qwen-image-txt2img") -> str:
        """Submit to Novita.ai API with endpoint selection and unified error handling [REH]."""
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
        payload, _warnings = self.build_payload_for_endpoint(endpoint, request)
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
                    if "invalid" in error_text.lower() or "parameter" in error_text.lower():
                        raise VisionError(
                            message=f"Invalid request: {error_text}",
                            error_type=VisionErrorType.VALIDATION_ERROR,
                            user_message="There was an issue with your request parameters. Please check and try again.",
                        )
                    if "prompt: value length must be between 1 and 2000" in error_text.lower():
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
                elif _is_balance_error(response.status, error_text):
                    raise VisionError(
                        message=f"Novita.ai balance/quota exhausted ({response.status}): {error_text}",
                        error_type=VisionErrorType.QUOTA_EXCEEDED,
                        user_message="The image provider's account is out of credit, so image generation is unavailable right now.",
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
                message=f"Novita.ai connection error: {e!s}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Unable to connect to vision service. Please try again.",
            )

    async def poll(self, job_id: str) -> UnifiedJobStatus:
        """Poll Novita.ai job status via async task-result endpoint [PA][REH]."""
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
                    unified = UnifiedStatus.FAILED if status == "TASK_STATUS_FAILED" else UnifiedStatus.CANCELLED
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
        """Fetch final result from Novita.ai async task-result API [REH]."""
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
            assets: list[str] = []
            images = data.get("images", []) or data.get("data", [])
            for item in images:
                url = item.get("image_url") or item.get("url") or item.get("download_url")
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

    def _calculate_cost(self, request: NormalizedRequest, endpoint: str = "qwen-image-txt2img") -> float:
        """Calculate estimated cost for Novita.ai using endpoint-specific pricing [PA][CMV]."""
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


class OpenRouterPlugin(ProviderPlugin):
    def __init__(self, name: str, config: dict[str, Any], api_key: str) -> None:
        super().__init__(name, config, api_key)
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        self.model_map = {
            VisionTask.TEXT_TO_IMAGE: "black-forest-labs/flux.2-pro",
        }

    def capabilities(self) -> dict[str, Any]:
        # IMAGE_TO_IMAGE deliberately NOT advertised: verified against OpenRouter's
        # live catalogue (2026-08-27) that every model with "image" in its
        # output_modalities is paid -- there is currently no free model this
        # provider could default to. qwen2.5-vl-72b-instruct:free (previously
        # listed here) is image+text->text only: an understanding model, not a
        # generation/edit one -- it can never produce an edited image regardless
        # of API key validity. Re-add only alongside a specific, deliberately
        # chosen (likely paid) model in model_map -- never as a bare capability
        # claim with no model behind it, which just moves the "always fails"
        # bug from "wrong key" to "wrong model". [SFT][REH]
        return {
            "modes": [VisionTask.TEXT_TO_IMAGE],
            "max_size": (2048, 2048),
            "max_steps": 0,
            "supports_negative_prompt": False,
            "supports_batch": False,
            "nsfw_policy": "unknown",
        }

    def _is_gemini_model(self, model: str) -> bool:
        try:
            m = (model or "").lower()
            return m.startswith("google/") or "gemini" in m
        except Exception as exc:
            logger.debug(f"Gemini model check failed: {exc}")
            return False

    def _aspect_ratio_for_dims(self, width: int, height: int) -> str:
        ratios = [
            (1.0, "1:1"),
            (2 / 3, "2:3"),
            (3 / 2, "3:2"),
            (3 / 4, "3:4"),
            (4 / 3, "4:3"),
            (4 / 5, "4:5"),
            (5 / 4, "5:4"),
            (9 / 16, "9:16"),
            (16 / 9, "16:9"),
            (21 / 9, "21:9"),
        ]
        try:
            w = max(1, int(width))
            h = max(1, int(height))
            r = w / h
            best = min(ratios, key=lambda x: abs(x[0] - r))
            return best[1]
        except Exception as exc:
            logger.debug(f"Aspect ratio calc failed: {exc}")
            return "1:1"

    async def submit(self, request: NormalizedRequest) -> str:
        await self.startup()
        model = request.preferred_model or self.model_map.get(request.task)
        if not model:
            raise VisionError(
                message=f"Task {request.task.value} not supported by {self.name}",
                error_type=VisionErrorType.UNSUPPORTED_TASK,
                user_message=f"Sorry, {request.task.value} is not supported by this provider.",
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "modalities": ["image", "text"],
            "stream": False,
        }
        if self._is_gemini_model(model):
            payload["image_config"] = {"aspect_ratio": self._aspect_ratio_for_dims(request.width, request.height)}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                error_text = await response.text()

                if response.status == 400:
                    raise VisionError(
                        message=f"{self.name} invalid request: {error_text}",
                        error_type=VisionErrorType.VALIDATION_ERROR,
                        user_message="There was an issue with your request parameters. Please check and try again.",
                    )
                if response.status == 401:
                    raise VisionError(
                        message=f"{self.name} authentication failed",
                        error_type=VisionErrorType.AUTHENTICATION_ERROR,
                        user_message="Vision service authentication failed. Please contact support.",
                    )
                if response.status == 429:
                    raise VisionError(
                        message=f"{self.name} rate limit exceeded",
                        error_type=VisionErrorType.RATE_LIMITED,
                        user_message="Too many requests. Please wait a moment and try again.",
                    )
                if response.status >= 500:
                    raise VisionError(
                        message=f"{self.name} server error: {error_text}",
                        error_type=VisionErrorType.SERVER_ERROR,
                        user_message="The vision service is temporarily unavailable. Please try again.",
                    )
                if response.status != 200:
                    raise VisionError(
                        message=f"{self.name} error ({response.status}): {error_text}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="Vision generation failed. Please try again.",
                    )

                result = await response.json()

                # Debug: log response structure without raw data [REH]
                logger.debug(f"{self.name} response keys: {list(result.keys())}, choices count: {len(result.get('choices') or [])}")

                urls: list[str] = []

                # Method 0: Check response-level 'data' array (DALL-E style) [CA]
                response_data = result.get("data") or []
                for item in response_data:
                    if isinstance(item, dict):
                        # Check for url, b64_json, or revised_prompt with embedded data
                        if item.get("url"):
                            urls.append(item["url"])
                        elif item.get("b64_json"):
                            # Convert b64_json to data URL
                            urls.append(f"data:image/png;base64,{item['b64_json']}")

                choices = result.get("choices") or []
                message = (choices[0] or {}).get("message") if choices else None

                if isinstance(message, dict):
                    # Method 1: Check message.images array
                    images = message.get("images") or []
                    for item in images:
                        # Handle string items directly (some models return URLs as strings)
                        if isinstance(item, str) and item:
                            urls.append(item)
                            continue
                        if not isinstance(item, dict):
                            continue
                        # Check various field names for image URL
                        image_url = item.get("image_url") or item.get("imageUrl") or item.get("url") or item.get("data")
                        if isinstance(image_url, str) and image_url:
                            urls.append(image_url)
                            continue
                        if isinstance(image_url, dict):
                            u = image_url.get("url")
                            if isinstance(u, str) and u:
                                urls.append(u)
                        # Check for b64_json field
                        if item.get("b64_json"):
                            urls.append(f"data:image/png;base64,{item['b64_json']}")

                    # Method 2: Check message.content array for image blocks [CA]
                    content = message.get("content")
                    if isinstance(content, list):
                        for block in content:
                            # Handle string blocks that might be data URLs
                            if isinstance(block, str):
                                if block.startswith("data:image"):
                                    urls.append(block)
                                continue
                            if not isinstance(block, dict):
                                continue
                            block_type = block.get("type", "")
                            if block_type == "image_url":
                                img_url_obj = block.get("image_url")
                                if isinstance(img_url_obj, str) and img_url_obj:
                                    urls.append(img_url_obj)
                                elif isinstance(img_url_obj, dict):
                                    u = img_url_obj.get("url")
                                    if isinstance(u, str) and u:
                                        urls.append(u)
                            elif block_type == "image":
                                # Some models use type=image with url, data, or b64_json
                                u = block.get("url") or block.get("data")
                                if isinstance(u, str) and u:
                                    urls.append(u)
                                elif block.get("b64_json"):
                                    urls.append(f"data:image/png;base64,{block['b64_json']}")
                            # Check for b64_json at block level regardless of type
                            elif block.get("b64_json"):
                                urls.append(f"data:image/png;base64,{block['b64_json']}")

                    # Method 3: Check for raw base64 in message.content string [REH]
                    if isinstance(content, str):
                        if content.startswith("data:image"):
                            urls.append(content)
                        # Some models embed base64 directly without data: prefix
                        elif len(content) > 1000 and not content.startswith(("{", "[")):
                            # Likely raw base64, wrap it
                            try:
                                # Validate it looks like base64
                                import re

                                if re.match(r"^[A-Za-z0-9+/=]+$", content[:100]):
                                    urls.append(f"data:image/png;base64,{content}")
                                    logger.debug("Detected raw base64 in message.content string")
                            except Exception as exc:
                                logger.debug(f"base64 validation failed: {exc}")

                # Deduplicate while preserving order
                seen = set()
                unique_urls = []
                for u in urls:
                    if u not in seen:
                        seen.add(u)
                        unique_urls.append(u)
                urls = unique_urls

                if not urls:
                    # Log response structure for debugging (without sensitive data)
                    logger.warning(f"{self.name}: No images found. Response structure: keys={list(result.keys())}, choices={len(choices)}, message_keys={list(message.keys()) if isinstance(message, dict) else 'N/A'}")
                    raise VisionError(
                        message="No images returned from provider",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="No images were returned by the provider.",
                    )

                # Log extracted images without dumping base64 data [REH]
                url_summary = []
                for u in urls:
                    if isinstance(u, str) and u.startswith("data:"):
                        # Summarize data URL without logging payload
                        mime_part = u.split(",")[0] if "," in u else u[:50]
                        url_summary.append(f"<data-url:{mime_part}>")
                    else:
                        url_summary.append(u[:100] if len(u) > 100 else u)
                logger.debug(f"{self.name} extracted {len(urls)} image(s): {url_summary}")

                job_id = f"{self.name}_{int(time.time() * 1000)}"
                self._results = getattr(self, "_results", {})
                self._results[job_id] = {
                    "status": "completed",
                    "assets": urls,
                    "cost": 0.0,
                    "model": model,
                }
                return job_id

        except VisionError:
            raise
        except Exception as e:
            raise VisionError(
                message=f"{self.name} connection error: {e!s}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Unable to connect to vision service. Please try again.",
            )

    async def poll(self, job_id: str) -> UnifiedJobStatus:
        results = getattr(self, "_results", {})
        if job_id not in results:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percentage=0,
                phase="Job not found",
            )
        return UnifiedJobStatus(
            status=UnifiedStatus.COMPLETED,
            progress_percentage=100,
            phase="Generation complete",
            actual_cost=results[job_id].get("cost", 0.0),
        )

    async def fetch_result(self, job_id: str) -> UnifiedResult:
        results = getattr(self, "_results", {})
        job_data = results.get(job_id, {})
        assets = list(job_data.get("assets") or [])
        return UnifiedResult(
            assets=assets,
            final_cost=float(job_data.get("cost", 0.0) or 0.0),
            provider_used=self.name,
            metadata={"model": job_data.get("model", "unknown")},
        )


class NvidiaPlugin(OpenRouterPlugin):
    """NVIDIA image provider using the GenAI endpoint, not chat/completions."""

    def __init__(self, name: str, config: dict[str, Any], api_key: str) -> None:
        super().__init__(name, config, api_key)
        # NVIDIA chat/text models live on integrate.api.nvidia.com/v1, but image
        # generation NIMs are exposed on ai.api.nvidia.com/v1/genai/{model}.
        # Calling integrate /chat/completions with image models returns 404.
        self.base_url = config.get("base_url", "https://integrate.api.nvidia.com/v1")
        self.genai_base_url = (config.get("genai_base_url") or os.getenv("NVIDIA_IMAGE_API_BASE") or "https://ai.api.nvidia.com/v1").rstrip("/")
        self.model_map = {
            VisionTask.TEXT_TO_IMAGE: "black-forest-labs/flux.1-dev",
        }
        # NOTE: FLUX.1 Kontext [dev] (black-forest-labs/flux.1-kontext-dev) was
        # tried here for IMAGE_TO_IMAGE and reverted - confirmed via a live 422
        # that this hosted GenAI endpoint only accepts one of NVIDIA's 3 canned
        # demo images (`"image": "data:image/png;example_id,{0,1,2}"`), not an
        # arbitrary base64-encoded user image. NVIDIA does ship a real
        # self-hostable NIM microservice for Kontext editing, but that's a
        # different deployment (your own GPU/endpoint), not this public API
        # tier. Don't re-add IMAGE_TO_IMAGE here without a genai_base_url that
        # actually accepts uploaded images. [CMV][REH]
        #
        # Re-verified 2026-08-27 against a live key, every encoding the endpoint
        # could plausibly want:
        #   raw base64                        -> 422 "Image has been provided in the invalid form"
        #   data:image/png;base64,<b64>       -> 422 "Expected: example_id, got: base64"
        #   data:image/png;asset_id,<uuid>    -> 422 "Expected: example_id, got: asset_id"
        #     (the NVCF asset upload itself succeeds - POST /v2/nvcf/assets 200,
        #      PUT to the presigned URL 200 - the genai endpoint just won't take it)
        #   data:image/png;example_id,3       -> 422 "Not valid example_id, expected value 0, 1, 2"
        # That last one enumerates the whole allowed set: three demo images.
        # flux.2-klein-4b enforces the same gate, so this is platform-wide on the
        # hosted preview endpoints, not specific to kontext.

    def capabilities(self) -> dict[str, Any]:
        # Explicit override, NOT inherited from OpenRouterPlugin: this class
        # subclasses OpenRouterPlugin for its chat/completions plumbing, but
        # submit() above hard-rejects everything except TEXT_TO_IMAGE (see the
        # NOTE above -- NVIDIA's hosted GenAI endpoint only accepts one of its
        # 3 canned demo images for editing, not an arbitrary upload). Letting
        # this fall through to the parent's capabilities() would advertise
        # IMAGE_TO_IMAGE support NVIDIA cannot actually serve, so the fallback
        # chain would pick NVIDIA and then error instead of skipping it. [REH]
        parent = super().capabilities()
        return {**parent, "modes": [VisionTask.TEXT_TO_IMAGE]}

    async def submit(self, request: NormalizedRequest) -> str:
        await self.startup()
        model = request.preferred_model or self.model_map.get(request.task)
        if request.task != VisionTask.TEXT_TO_IMAGE or not model:
            raise VisionError(
                message=f"Task {request.task.value} not supported by {self.name}",
                error_type=VisionErrorType.UNSUPPORTED_TASK,
                user_message=f"Sorry, {request.task.value} is not supported by this provider.",
            )

        payload: dict[str, Any] = {"prompt": (request.prompt or "").strip()}
        if not payload["prompt"]:
            raise VisionError(
                message="Prompt is empty after normalization",
                error_type=VisionErrorType.VALIDATION_ERROR,
                user_message="A prompt is required.",
            )
        if request.seed is not None:
            payload["seed"] = int(request.seed)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{self.genai_base_url}/genai/{model}"

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                error_text = await response.text()
                if response.status in (400, 422):
                    raise VisionError(
                        message=f"{self.name} invalid request ({response.status}): {error_text}",
                        error_type=VisionErrorType.VALIDATION_ERROR,
                        user_message="There was an issue with your request parameters. Please check and try again.",
                    )
                if response.status == 401:
                    raise VisionError(
                        message=f"{self.name} authentication failed",
                        error_type=VisionErrorType.AUTHENTICATION_ERROR,
                        user_message="Vision service authentication failed. Please contact support.",
                    )
                if response.status == 403:
                    raise VisionError(
                        message=f"{self.name} forbidden/quota error: {error_text}",
                        error_type=VisionErrorType.QUOTA_EXCEEDED,
                        user_message="Vision service quota or access is unavailable. Please try another provider.",
                    )
                if response.status == 404:
                    raise VisionError(
                        message=f"{self.name} model endpoint not found: {model}",
                        error_type=VisionErrorType.UNSUPPORTED_TASK,
                        user_message="That NVIDIA image model is not available.",
                    )
                if response.status == 429:
                    raise VisionError(
                        message=f"{self.name} rate limit exceeded",
                        error_type=VisionErrorType.RATE_LIMITED,
                        user_message="Too many requests. Please wait a moment and try again.",
                    )
                if response.status >= 500:
                    raise VisionError(
                        message=f"{self.name} server error: {error_text}",
                        error_type=VisionErrorType.SERVER_ERROR,
                        user_message="The vision service is temporarily unavailable. Please try again.",
                    )
                if response.status != 200:
                    raise VisionError(
                        message=f"{self.name} error ({response.status}): {error_text}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="Vision generation failed. Please try again.",
                    )

                result = await response.json()
                assets: list[str] = []
                for artifact in result.get("artifacts") or []:
                    if not isinstance(artifact, dict):
                        continue
                    b64 = artifact.get("base64")
                    if isinstance(b64, str) and b64:
                        assets.append(f"data:image/jpeg;base64,{b64}")
                    url_value = artifact.get("url")
                    if isinstance(url_value, str) and url_value:
                        assets.append(url_value)

                if not assets:
                    raise VisionError(
                        message="No images returned from NVIDIA GenAI endpoint",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="No images were returned by the provider.",
                    )

                job_id = f"{self.name}_{int(time.time() * 1000)}"
                self._results = getattr(self, "_results", {})
                self._results[job_id] = {
                    "status": "completed",
                    "assets": assets,
                    "cost": 0.0,
                    "model": model,
                }
                return job_id

        except VisionError:
            raise
        except Exception as e:
            raise VisionError(
                message=f"{self.name} connection error: {e!s}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Unable to connect to vision service. Please try again.",
            )


class UnifiedVisionAdapter:
    """Unified Vision adapter with pluggable provider system [CA][REH][SFT].

    Features:
    - Automatic provider selection and fallback
    - Parameter normalization and validation
    - Unified error handling and status mapping
    - Cost estimation and budget enforcement
    - JSON-based configuration
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.providers: dict[str, ProviderPlugin] = {}
        self.provider_config = {}
        self._provider_lock = asyncio.Lock()  # Provider initialization lock

        self.allowed_providers = self._parse_allowed_providers(config)
        self.default_provider = self._parse_default_provider(config)
        # provider -> monotonic deadline while it is benched for an empty account [REH]
        self._quota_cooldowns: dict[str, float] = {}

        # Model override from environment [CMV]
        self.vision_model_override = self._parse_vision_model_override(config)
        if self.vision_model_override:
            self.logger.info(f"🎯 VISION_MODEL override active: {self.vision_model_override}")

        # Load provider configuration
        self._load_provider_config()
        self._initialize_providers()

        # Initialize model resolver
        self._model_aliases = self._build_model_aliases()
        self._log_task_coverage()

    @staticmethod
    def _parse_allowed_providers(config: dict[str, Any]) -> list[str]:
        raw = config.get("VISION_ALLOWED_PROVIDERS") or []
        if isinstance(raw, str):
            raw = raw.split(",")
        return [str(p).strip().lower() for p in raw if str(p).strip()]

    @staticmethod
    def _parse_default_provider(config: dict[str, Any]) -> str:
        return str(config.get("VISION_DEFAULT_PROVIDER") or "").strip().lower()

    @staticmethod
    def _parse_vision_model_override(config: dict[str, Any]) -> str:
        return str(config.get("VISION_MODEL") or "").strip()

    def update_config(self, config: dict[str, Any]) -> None:
        """Hot-reload mutable vision adapter config without recreating sessions."""
        self.config = config
        self.allowed_providers = self._parse_allowed_providers(config)
        self.default_provider = self._parse_default_provider(config)
        self.vision_model_override = self._parse_vision_model_override(config)
        # A config reload may have added credit/keys: give every provider a fresh start.
        self._quota_cooldowns = {}
        self._load_provider_config()
        self._initialize_providers()
        self._model_aliases = self._build_model_aliases()
        self._log_task_coverage()

        fallback_key = str(config.get("VISION_API_KEY") or "")
        for provider_config in self.provider_config.get("vision", {}).get("providers", []):
            name = str(provider_config.get("name") or "").lower()
            provider = self.providers.get(name)
            if provider is None:
                continue
            provider.config = provider_config
            if name == "openrouter":
                # Same special-case as _initialize_providers() -- otherwise a
                # hot-reload silently resets openrouter's key back to
                # VISION_API_KEY (Together/Novita's key) right after
                # _initialize_providers() resolved it correctly. [SFT][REH]
                provider.api_key = _resolve_openrouter_api_key(config) or fallback_key
            else:
                key_env = provider_config.get("api_key_env", "VISION_API_KEY")
                provider.api_key = str(config.get(key_env) or fallback_key or "")
            if hasattr(provider, "base_url"):
                provider.base_url = provider_config.get("base_url", provider.base_url)
            if hasattr(provider, "genai_base_url"):
                provider.genai_base_url = provider_config.get("genai_base_url", provider.genai_base_url)

        self.logger.info(
            "vision.adapter.config_updated default=%s allowed=%s model_override=%s",
            self.default_provider,
            self.allowed_providers,
            self.vision_model_override,
        )

    def _load_provider_config(self) -> None:
        """Load provider configuration from JSON or defaults [IV]."""
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

    def _default_provider_config(self) -> dict[str, Any]:
        """Default provider configuration [CMV]."""
        return {
            "vision": {
                "default_policy": {
                    # NOTE: this is the list actually used at runtime whenever
                    # VISION_PROVIDER_CONFIG_PATH is unset (the common case) --
                    # submit()'s `policy.get("provider_order", [...])` fallback
                    # is dead code once this key exists, so it must be kept in
                    # sync here too. [CA][REH]
                    #
                    # OpenRouter deliberately NOT listed: verified against its
                    # live catalogue (2026-08-27) that no free model can serve
                    # IMAGE_TO_IMAGE (every image-output-capable model is paid).
                    # See OpenRouterPlugin.capabilities() for the full note.
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
                    {
                        "name": "openrouter",
                        "base_url": "https://openrouter.ai/api/v1",
                        # Informational fallback only -- _initialize_providers()
                        # special-cases "openrouter" to use resolve_openrouter_key()
                        # instead, since a deployment's real OpenRouter key is
                        # often stored under OPENAI_API_KEY (text backend) rather
                        # than a dedicated OPENROUTER_API_KEY or VISION_API_KEY
                        # (which is typically a Together/Novita key). [SFT][REH]
                        "api_key_env": "OPENROUTER_API_KEY",
                        "enabled": True,
                        "priority": 3,
                        "price": {
                            "image_base": 0.02,
                            "image_per_px": 0.000005,
                        },
                        "limits": {"max_size": "2048x2048", "max_steps": 0},
                    },
                    {
                        "name": "nvidia",
                        "base_url": "https://integrate.api.nvidia.com/v1",
                        "genai_base_url": "https://ai.api.nvidia.com/v1",
                        "api_key_env": "NVIDIA_NIM_API_KEY",
                        "enabled": True,
                        "priority": 4,
                        "price": {
                            "image_base": 0.02,
                            "image_per_px": 0.000005,
                        },
                        "limits": {"max_size": "2048x2048", "max_steps": 0},
                    },
                ],
            },
        }

    def _initialize_providers(self) -> None:
        """Initialize provider plugins [CA]."""
        api_key = self.config.get("VISION_API_KEY", "")

        for provider_config in self.provider_config["vision"]["providers"]:
            if not provider_config.get("enabled", True):
                continue

            name = provider_config["name"]

            if self.allowed_providers and name.lower() not in self.allowed_providers:
                continue

            # Get API key from environment. OpenRouter is special-cased: its key
            # is commonly whatever OPENAI_API_KEY holds when OPENAI_API_BASE
            # points at openrouter.ai (the text-backend setup), not a dedicated
            # OPENROUTER_API_KEY or the Together/Novita-flavored VISION_API_KEY.
            # [SFT][REH]
            if name == "openrouter":
                provider_key = _resolve_openrouter_api_key(self.config)
            else:
                key_env = provider_config.get("api_key_env", "VISION_API_KEY")
                provider_key = self.config.get(key_env, api_key)

            # Plugins only ever see their own provider_config, so the global
            # per-request timeout has to be handed down explicitly. [CMV]
            provider_timeout_ms = self.config.get("VISION_PROVIDER_TIMEOUT_MS")
            if provider_timeout_ms not in (None, ""):
                provider_config.setdefault("timeout_ms", provider_timeout_ms)

            if not provider_key:
                self.logger.warning(f"No API key found for provider {name}")
                continue

            try:
                if name in self.providers:
                    plugin = self.providers[name]
                    plugin.config = provider_config
                    plugin.api_key = provider_key
                    if hasattr(plugin, "base_url"):
                        plugin.base_url = provider_config.get("base_url", plugin.base_url)
                    if hasattr(plugin, "genai_base_url"):
                        plugin.genai_base_url = provider_config.get("genai_base_url", plugin.genai_base_url)
                    continue

                if name == "together":
                    plugin = TogetherPlugin(name, provider_config, provider_key)
                elif name == "novita":
                    plugin = NovitaPlugin(name, provider_config, provider_key)
                elif name == "openrouter":
                    plugin = OpenRouterPlugin(name, provider_config, provider_key)
                elif name == "nvidia":
                    plugin = NvidiaPlugin(name, provider_config, provider_key)
                else:
                    self.logger.warning(f"Unknown provider: {name}")
                    continue

                self.providers[name] = plugin
                self.logger.info(f"Initialized provider: {name}")

            except Exception as e:
                self.logger.exception(f"Failed to initialize provider {name}: {e}")

    def _has_valid_credentials(self, provider_name: str) -> bool:
        """Best-effort credential presence check per provider [SFT].
        Each provider checks for its specific key first, then falls back to VISION_API_KEY.
        """
        try:
            name = provider_name.split(":", maxsplit=1)[0].lower()
            if name == "novita":
                key = self.config.get("NOVITA_API_KEY") or self.config.get("VISION_API_KEY") or ""
            elif name == "together":
                key = self.config.get("TOGETHER_API_KEY") or self.config.get("VISION_API_KEY_TOGETHER") or ""
            elif name == "openrouter":
                # VISION_API_KEY is typically a Together/Novita key, not a valid
                # OpenRouter one -- checking for its mere presence here used to
                # let this gate pass with the wrong key, so the failure only
                # surfaced later as an opaque "authentication failed" from the
                # actual API call instead of being skipped upfront. [SFT][REH]
                key = _resolve_openrouter_api_key(self.config)
            elif name == "nvidia":
                key = self.config.get("NVIDIA_NIM_API_KEY") or self.config.get("OPENAI_API_KEY") or self.config.get("VISION_API_KEY") or ""
            else:
                key = self.config.get("VISION_API_KEY", "")
            return isinstance(key, str) and len(key.strip()) > 10
        except Exception as exc:
            logger.debug(f"Credential check failed for {provider_name}: {exc}")
            return False

    def _quota_cooldown_s(self) -> float:
        """Bench duration after an out-of-balance response. [CMV]"""
        try:
            return max(0.0, float(self.config.get("VISION_PROVIDER_QUOTA_COOLDOWN_S", DEFAULT_QUOTA_COOLDOWN_S)))
        except (TypeError, ValueError):
            return DEFAULT_QUOTA_COOLDOWN_S

    def _bench_provider_for_quota(self, provider_name: str) -> None:
        """Bench a provider that just reported an empty account. [REH][PA]"""
        cooldown = self._quota_cooldown_s()
        if cooldown <= 0:
            return
        base = provider_name.split(":", maxsplit=1)[0].lower()
        self._quota_cooldowns[base] = time.monotonic() + cooldown
        self.logger.warning(
            "vision.provider.benched provider=%s reason=quota_exceeded cooldown_s=%.0f",
            base,
            cooldown,
        )

    def _quota_cooldown_remaining(self, provider_name: str) -> float:
        """Seconds left on this provider's quota bench (0 when it is available)."""
        base = provider_name.split(":", maxsplit=1)[0].lower()
        until = self._quota_cooldowns.get(base)
        if until is None:
            return 0.0
        remaining = until - time.monotonic()
        if remaining <= 0:
            self._quota_cooldowns.pop(base, None)
            return 0.0
        return remaining

    def audit_task_coverage(self) -> dict[str, list[str]]:
        """Which usable providers declare support for each vision task. [REH]

        "Usable" means: initialized, in VISION_ALLOWED_PROVIDERS and holding
        credentials -- the same gates the submit path applies -- so this answers
        "would a request for X find anyone to serve it?" rather than "is a plugin
        for X compiled in?".
        """
        coverage: dict[str, list[str]] = {}
        for task in VisionTask:
            capable: list[str] = []
            for name, plugin in self.providers.items():
                if self.allowed_providers and name.lower() not in self.allowed_providers:
                    continue
                if not self._has_valid_credentials(name):
                    continue
                try:
                    modes = plugin.capabilities().get("modes", []) or []
                except (AttributeError, TypeError, ValueError) as exc:
                    self.logger.debug(f"capabilities() failed for {name}: {exc}")
                    continue
                if task in modes:
                    capable.append(name)
            coverage[task.value] = capable
        return coverage

    def _log_task_coverage(self) -> None:
        """Say at startup which tasks nobody can serve. [REH]

        The operator's .env can name a default provider (VISION_DEFAULT_PROVIDER)
        or an image ladder for a task that provider cannot actually perform. That
        used to surface only as a confusing per-request failure; now the mismatch
        is stated once, at boot, with the reason.
        """
        with contextlib.suppress(Exception):
            coverage = self.audit_task_coverage()
            self.logger.info("vision.task_coverage %s", coverage)
            uncovered = [task for task, names in coverage.items() if not names]
            if uncovered:
                self.logger.warning(
                    "vision.task_coverage.uncovered tasks=%s providers_loaded=%s allowed=%s (requests for these tasks cannot succeed with the current config)",
                    uncovered,
                    sorted(self.providers),
                    self.allowed_providers or "all",
                )
            default = self.default_provider
            if default:
                default_tasks = [task for task, names in coverage.items() if default in names]
                self.logger.info("vision.default_provider provider=%s serves=%s", default, default_tasks)
                unservable = [task for task, names in coverage.items() if names and default not in names]
                if unservable:
                    self.logger.warning(
                        "vision.default_provider.cannot_serve provider=%s tasks=%s handled_by=%s",
                        default,
                        unservable,
                        {task: coverage[task] for task in unservable},
                    )

    def _filter_provider_order(self, provider_order: list[str]) -> tuple[list[str], dict[str, str]]:
        """Keep usable providers, and record WHY each of the others was dropped. [REH]

        The drop reasons used to be invisible, so a provider that vanished from the
        ladder (missing key, benched, never initialized) looked exactly like one that
        was never configured -- and the ladder silently shrank to a single rung.
        """
        filtered: list[str] = []
        dropped: dict[str, str] = {}
        for name in provider_order:
            base = name.split(":")[0].lower()
            if base in filtered or base in dropped:
                continue
            if self.allowed_providers and base not in self.allowed_providers:
                dropped[base] = "not_in_allowlist"
            elif base not in self.providers:
                dropped[base] = "not_initialized"
            elif not self._has_valid_credentials(base):
                dropped[base] = "no_credentials"
            elif self._quota_cooldown_remaining(base) > 0:
                dropped[base] = f"quota_benched_{self._quota_cooldown_remaining(base):.0f}s"
            elif not self._is_provider_healthy(base):
                dropped[base] = "unhealthy"
            else:
                filtered.append(base)
        return filtered, dropped

    def _is_provider_healthy(self, provider_name: str) -> bool:
        """Credential presence plus the out-of-balance bench. [PA][REH]"""
        if self._quota_cooldown_remaining(provider_name) > 0:
            return False
        return self._has_valid_credentials(provider_name)

    def _estimate_cost_money(self, provider_name: str, request: NormalizedRequest) -> Money | None:
        """Best-effort Money estimate using pricing table [PA][CMV].
        Returns None if pricing table does not cover this provider/task.
        """
        try:
            table = get_pricing_table()
            # Normalize provider string and strip endpoint suffix if present [IV]
            normalized_provider = provider_name.split(":", maxsplit=1)[0].lower()
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
        except Exception as exc:
            logger.debug(f"Cost estimate failed for {provider_name}: {exc}")
            return None

    def _build_model_aliases(self) -> dict[str, ModelSelection]:
        """Build model alias mapping for VISION_MODEL resolution [CMV]."""
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

        # OpenRouter aliases [CA]
        # Seedream 4.5
        for alias in [
            "openrouter:bytedance-seed/seedream-4.5",
            "openrouter:seedream-4.5",
            "seedream-4.5",
            "seedream",
        ]:
            aliases[alias] = ModelSelection(
                provider="openrouter",
                endpoint="chat/completions",
                model_hint="bytedance-seed/seedream-4.5",
                supports_advanced=False,
            )

        # FLUX models via OpenRouter
        for alias in [
            "openrouter:flux-pro",
            "openrouter:flux.2-pro",
            "openrouter:black-forest-labs/flux-pro",
        ]:
            aliases[alias] = ModelSelection(
                provider="openrouter",
                endpoint="chat/completions",
                model_hint="black-forest-labs/flux-pro",
                supports_advanced=False,
            )

        # Gemini image models via OpenRouter
        for alias in [
            "openrouter:gemini-2.0-flash-exp",
            "openrouter:google/gemini-2.0-flash-exp:free",
        ]:
            aliases[alias] = ModelSelection(
                provider="openrouter",
                endpoint="chat/completions",
                model_hint="google/gemini-2.0-flash-exp:free",
                supports_advanced=False,
            )

        # Generic OpenRouter alias (uses default model)
        aliases["openrouter"] = ModelSelection(
            provider="openrouter",
            endpoint="chat/completions",
            model_hint="black-forest-labs/flux-pro",
            supports_advanced=False,
        )

        # NVIDIA Build/NIM aliases [CA]
        for alias, model in {
            "nvidia:black-forest-labs/flux_2-klein-4b": "black-forest-labs/flux_2-klein-4b",
            "nvidia:flux_2-klein-4b": "black-forest-labs/flux_2-klein-4b",
            "nvidia:qwen/qwen-image": "qwen/qwen-image",
            "nvidia:qwen-image": "qwen/qwen-image",
        }.items():
            aliases[alias] = ModelSelection(
                provider="nvidia",
                endpoint="chat/completions",
                model_hint=model,
                supports_advanced=False,
            )
        aliases["nvidia"] = ModelSelection(
            provider="nvidia",
            endpoint="chat/completions",
            model_hint="black-forest-labs/flux_2-klein-4b",
            supports_advanced=False,
        )

        return aliases

    def resolve_model_selection(self, request: NormalizedRequest) -> ModelSelection | None:
        """Resolve VISION_MODEL override to specific provider/endpoint [CA]."""
        if not self.vision_model_override:
            return None

        # Direct alias lookup
        selection = self._model_aliases.get(self.vision_model_override.lower())
        if selection:
            self.logger.info(f" Resolved VISION_MODEL '{self.vision_model_override}' → {selection.provider}:{selection.endpoint}")
            return selection

        # Parse provider:model format
        if ":" in self.vision_model_override:
            parts = self.vision_model_override.split(":", 1)
            provider_name = parts[0].lower()
            model_part = parts[1] if len(parts) > 1 else None

            if provider_name in self.providers:
                # Default to provider's primary endpoint
                if provider_name == "novita":
                    return ModelSelection(
                        provider="novita",
                        endpoint="qwen-image-txt2img",  # Default to Qwen
                        model_hint="qwen-image",
                        supports_advanced=False,
                    )
                if provider_name == "together":
                    return ModelSelection(
                        provider="together",
                        endpoint="images/generations",
                        model_hint="black-forest-labs/FLUX.1-schnell-Free",
                        supports_advanced=True,
                    )
                if provider_name == "openrouter":
                    # OpenRouter: use the model part as-is if provided [CA]
                    model_hint = model_part or "black-forest-labs/flux-pro"
                    self.logger.info(f"Resolved VISION_MODEL '{self.vision_model_override}' → openrouter with model={model_hint}")
                    return ModelSelection(
                        provider="openrouter",
                        endpoint="chat/completions",
                        model_hint=model_hint,
                        supports_advanced=False,
                    )
                if provider_name == "nvidia":
                    model_hint = model_part or "black-forest-labs/flux.1-dev"
                    self.logger.info(f"Resolved VISION_MODEL '{self.vision_model_override}' → nvidia with model={model_hint}")
                    return ModelSelection(
                        provider="nvidia",
                        endpoint="genai",
                        model_hint=model_hint,
                        supports_advanced=False,
                    )

        self.logger.warning(f"Unrecognized VISION_MODEL '{self.vision_model_override}', falling back to policy")
        return None

    async def startup(self) -> None:
        """Initialize all provider connections [REH]."""
        for name, provider in self.providers.items():
            try:
                await provider.startup()
                self.logger.info(f"Started provider: {name}")
            except Exception as e:
                self.logger.exception(f"Failed to start provider {name}: {e}")

    async def shutdown(self) -> None:
        """Cleanup all provider connections [RM]."""
        for name, provider in self.providers.items():
            try:
                await provider.shutdown()
                self.logger.info(f"Shutdown provider: {name}")
            except Exception as e:
                self.logger.exception(f"Failed to shutdown provider {name}: {e}")

    def normalize_request(self, request: VisionRequest) -> NormalizedRequest:
        """Normalize request parameters across providers [IV]."""
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
            preferred_model=request.preferred_model,
        )

    def select_provider(self, request: NormalizedRequest) -> ProviderPlugin | None:
        """Select best provider for request based on capabilities and policy [CA]."""
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
                    self.logger.warning(f"Provider {provider_key} cost ${estimated_cost:.3f} exceeds budget ${budget_limit:.3f}")
                    continue

                if money_est is not None:
                    self.logger.info(f"Selected provider: {provider_key} (cost: ${estimated_cost:.3f}, money={money_est})")
                else:
                    self.logger.info(f"Selected provider: {provider_key} (cost: ${estimated_cost:.3f})")
                return provider

            except Exception as e:
                self.logger.warning(f"Cost estimation failed for {provider_key}: {e}")
                continue

        return None

    def _estimate_cost(self, provider: ProviderPlugin, request: NormalizedRequest) -> float:
        """Estimate cost for provider and request [PA]."""
        config = next(
            (p for p in self.provider_config["vision"]["providers"] if p["name"] == provider.name),
            {},
        )

        pricing = config.get("price", {})
        base_cost = pricing.get("image_base", 0.02)

        if request.task == VisionTask.VIDEO_GENERATION:
            video_cost = pricing.get("video_per_s", 0.05)
            duration = request.video_seconds or 4
            return base_cost + (video_cost * duration)
        pixel_cost = pricing.get("image_per_px", 0.000005)
        pixels = request.width * request.height
        return base_cost + (pixels * pixel_cost)

    def _parse_provider_model_ladder(self, raw: str) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for part in str(raw or "").split(","):
            item = part.strip()
            if not item:
                continue
            if "|" in item:
                provider, model = item.split("|", 1)
            else:
                provider, model = self.default_provider or "openrouter", item
            provider = provider.strip().lower() or "openrouter"
            model = model.strip()
            if model:
                entries.append((provider, model))
        return entries

    async def _submit_text_to_image_ladder(self, normalized_request: NormalizedRequest, raw_ladder: str) -> VisionResponse:
        last_error: VisionError | None = None
        for provider_name, model_name in self._parse_provider_model_ladder(raw_ladder):
            if self.allowed_providers and provider_name not in self.allowed_providers:
                continue
            if provider_name not in self.providers:
                continue
            if not (self._has_valid_credentials(provider_name) and self._is_provider_healthy(provider_name)):
                continue

            provider = self.providers[provider_name]
            capabilities = provider.capabilities()
            if normalized_request.task not in capabilities.get("modes", []):
                continue

            normalized_request.preferred_model = model_name
            try:
                self.logger.info(f"image.ladder.attempt provider={provider_name} model={model_name}")
                task_id = await provider.submit(normalized_request)
                return VisionResponse(
                    success=True,
                    job_id=f"{provider_name}:{task_id}",
                    provider=VisionProvider(provider_name.lower()),
                    model_used=model_name,
                    provider_job_id=task_id,
                )
            except VisionError as exc:
                last_error = exc
                self.logger.warning(f"image.ladder.fail provider={provider_name} model={model_name} error={exc.message}")
                continue
            except Exception as exc:
                last_error = VisionError(
                    message=f"Unexpected error from {provider_name}: {exc}",
                    error_type=VisionErrorType.PROVIDER_ERROR,
                    user_message="Image generation is temporarily unavailable. Please try again later.",
                )
                self.logger.warning(f"image.ladder.fail provider={provider_name} model={model_name} error={exc}")
                continue

        raise last_error or VisionError(
            message="All image generation providers exhausted",
            error_type=VisionErrorType.PROVIDER_ERROR,
            user_message="Image generation is temporarily unavailable. Please try again later.",
        )

    async def submit(self, request: VisionRequest) -> VisionResponse:
        """Submit vision request with VISION_MODEL override and automatic provider selection [REH]."""
        policy = self.provider_config.get("vision", {}).get("default_policy", {})
        normalized_request = self.normalize_request(request)

        image_ladder_raw = str(self.config.get("VISION_IMAGE_FALLBACK_MODELS") or "").strip()
        if normalized_request.task == VisionTask.TEXT_TO_IMAGE and image_ladder_raw:
            return await self._submit_text_to_image_ladder(normalized_request, image_ladder_raw)

        preferred_provider_obj = getattr(request, "preferred_provider", None)
        if preferred_provider_obj is not None:
            try:
                preferred_name = preferred_provider_obj.value.lower()
            except Exception as exc:
                logger.debug(f"Preferred provider value extraction failed: {exc}")
                preferred_name = str(preferred_provider_obj).strip().lower()

            if self.allowed_providers and preferred_name not in self.allowed_providers:
                raise VisionError(
                    message=f"Provider '{preferred_name}' not allowed",
                    error_type=VisionErrorType.VALIDATION_ERROR,
                    user_message="That vision provider is not enabled. Please choose a different provider.",
                    provider=None,
                )

            if preferred_name not in self.providers:
                raise VisionError(
                    message=f"Provider '{preferred_name}' is not configured",
                    error_type=VisionErrorType.SYSTEM_ERROR,
                    user_message="That vision provider is not available right now. Please try again later.",
                    provider=None,
                )

        # Check for VISION_MODEL override
        model_selection = self.resolve_model_selection(normalized_request)

        if model_selection:
            # Use pinned model/provider/endpoint
            provider_order = [model_selection.provider]
            forced_endpoint = model_selection.endpoint
            # Apply model_hint to normalized request so provider uses correct model [CA]
            if model_selection.model_hint:
                normalized_request.preferred_model = model_selection.model_hint
            self.logger.info(f"🎯 Using VISION_MODEL override: {model_selection.provider}:{model_selection.endpoint} (model={model_selection.model_hint})")
        else:
            # Use policy-driven selection
            # NOTE: this .get() default only fires when `policy` truly lacks the
            # key (e.g. a custom VISION_PROVIDER_CONFIG_PATH file without one) --
            # the common case (no override file) uses _default_provider_config()'s
            # own provider_order above, which must be kept in sync separately.
            provider_order = policy.get("provider_order", ["novita:qwen-image", "novita:txt2img", "together"])
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

        preferred_provider = preferred_provider_obj
        if preferred_provider is not None:
            try:
                preferred_name = preferred_provider.value.lower()
                provider_order = [preferred_name, *list(provider_order)]
            except Exception as exc:
                logger.debug(f"preferred provider ordering failed: {exc}")

        # Promote the configured default provider to the front of the order.
        # Previously this only special-cased "openrouter", so e.g.
        # VISION_DEFAULT_PROVIDER=nvidia was silently ignored for every other
        # provider name. Safe to generalize: providers that can't handle the
        # requested task are skipped later via the capabilities() check
        # (line ~1945), so promoting a provider with no matching capability
        # (e.g. nvidia for IMAGE_TO_IMAGE) just falls through to the next
        # candidate instead of erroring. [REH][CMV]
        if self.default_provider:
            with contextlib.suppress(Exception):
                provider_order = [self.default_provider] + [p for p in provider_order if p != self.default_provider]

        # Filter by configured/healthy providers, preserving order [SFT]
        filtered_order, dropped = self._filter_provider_order(provider_order)
        if not filtered_order:
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"No configured/healthy vision providers available (dropped: {dropped})",
                user_message="Image generation is not available right now — no provider is configured for it.",
            )
        self.logger.info("vision.provider_order task=%s order=%s dropped=%s", normalized_request.task.value, filtered_order, dropped)
        provider_order = filtered_order

        last_error = None

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            # Ensure provider is properly initialized with thread-safe access
            try:
                async with self._provider_lock:
                    provider = self.providers[provider_name]
                    # Double-check provider health after lock acquisition
                    if not self._is_provider_healthy(provider_name):
                        continue
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} initialization failed: {e}")
                continue
            # Belt-and-braces: skip if not configured/healthy
            if not (self._has_valid_credentials(provider_name) and self._is_provider_healthy(provider_name)):
                self.logger.debug(f"Skipping provider {provider_name}: not configured/unhealthy")
                continue

            # Check if provider supports the task
            capabilities = provider.capabilities()
            supported_modes = capabilities.get("modes", [])
            task_supported = normalized_request.task in supported_modes
            self.logger.info(
                f"vision.capability_check provider={provider_name} task={normalized_request.task.value} "
                f"supported={task_supported} modes={[m.value if hasattr(m, 'value') else m for m in supported_modes]} "
                f"plugin_class={type(provider).__name__}",
            )
            if not task_supported:
                continue

            # Determine endpoint for this provider
            endpoint = None
            if forced_endpoint and model_selection and provider_name == model_selection.provider:
                endpoint = forced_endpoint
            elif provider_name == "novita":
                # Default endpoint selection for Novita based on policy order
                original_entry = next(
                    (e for e in policy.get("provider_order", []) if e.startswith("novita")),
                    "novita:qwen-image",
                )
                endpoint = "qwen-image-txt2img" if "qwen-image" in original_entry else "txt2img"
            # Only the txt2img endpoint carries init_image; the qwen endpoint's payload
            # allowlist is {prompt, size}, so routing an edit there silently DROPPED the
            # user's image and generated a fresh one from the prompt alone. [IV][REH]
            if provider_name == "novita" and normalized_request.task == VisionTask.IMAGE_TO_IMAGE and endpoint != NOVITA_IMAGE_TO_IMAGE_ENDPOINT:
                self.logger.info(
                    "vision.novita.endpoint_override from=%s to=%s reason=image_to_image_needs_init_image",
                    endpoint,
                    NOVITA_IMAGE_TO_IMAGE_ENDPOINT,
                )
                endpoint = NOVITA_IMAGE_TO_IMAGE_ENDPOINT

            # Attempt submission with retries
            did_prompt_retry = False  # Single-shot retry for Novita prompt length
            for _attempt in range(policy.get("max_retries_per_provider", 2)):
                try:
                    # Submit with endpoint if supported (Novita), otherwise default
                    if hasattr(provider, "submit") and endpoint and provider_name == "novita":
                        task_id = await provider.submit(normalized_request, endpoint)
                    else:
                        task_id = await provider.submit(normalized_request)

                    return VisionResponse(
                        success=True,
                        job_id=f"{provider_name}:{task_id}",
                        provider=VisionProvider(provider_name.lower()),
                        model_used=getattr(provider, "model_map", {}).get(normalized_request.task, "unknown"),
                        provider_job_id=task_id,
                    )

                except VisionError as e:
                    last_error = e
                    # Special-case: Novita 400 prompt length error → one clean retry with 2000-char prompt [REH]
                    if provider_name == "novita" and e.error_type == VisionErrorType.VALIDATION_ERROR and "prompt: value length must be between 1 and 2000" in (e.message or "").lower():
                        if not did_prompt_retry:
                            # Normalize whitespace and clamp to 2000
                            clamped = " ".join((normalized_request.prompt or "").split()).strip()
                            if len(clamped) > 2000:
                                clamped = clamped[:2000]
                            if len(clamped) == 0:
                                # Nothing to retry with
                                self.logger.debug("Novita prompt length retry skipped: prompt empty after normalization")
                            else:
                                normalized_request.prompt = clamped
                                self.logger.debug("Novita retry with 2000-char prompt")
                                did_prompt_retry = True
                                # Immediate retry without backoff
                                continue
                        # Already retried once; fall through to existing handling
                    elif e.error_type == VisionErrorType.QUOTA_EXCEEDED:
                        # Payment/quota errors - NOT retryable, try next provider immediately,
                        # and bench this one so the next job does not re-pay the round trip. [REH][PA]
                        self.logger.warning(f"Quota/balance error from {provider_name}: {e.message}. Trying next provider...")
                        self._bench_provider_for_quota(provider_name)
                        break
                    else:
                        # Non-retryable error, try next provider
                        self.logger.warning(f"Non-retryable error from {provider_name}: {e.message}")
                        break

                except Exception as e:
                    last_error = VisionError(
                        message=f"Unexpected error from {provider_name}: {e!s}",
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        user_message="An unexpected error occurred during image generation.",
                    )
                    self.logger.error(f"Unexpected error from {provider_name}: {e}", exc_info=True)
                    break

        # All providers failed
        raise last_error or VisionError(
            message="All vision providers exhausted",
            error_type=VisionErrorType.PROVIDER_ERROR,
            user_message="Image generation is temporarily unavailable. Please try again later.",
        )

    async def poll(self, full_job_id: str) -> UnifiedJobStatus:
        """Poll job status across providers."""
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
            self.logger.exception(f"Failed to poll job {full_job_id}: {e}")
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percentage=0,
                phase=f"Polling error: {e}",
            )

    async def fetch_result(self, full_job_id: str) -> UnifiedResult:
        """Fetch final result from provider."""
        try:
            provider_name, job_id = full_job_id.split(":", 1)
            provider = self.providers.get(provider_name)

            if not provider:
                msg = f"Provider {provider_name} not available"
                raise VisionError(
                    msg,
                    VisionErrorType.PROVIDER_ERROR,
                )

            return await provider.fetch_result(job_id)

        except ValueError:
            msg = "Invalid job ID format"
            raise VisionError(msg, VisionErrorType.VALIDATION_ERROR)
        except Exception as e:
            self.logger.exception(f"Failed to fetch result for {full_job_id}: {e}")
            msg = f"Result fetch failed: {e}"
            raise VisionError(msg, VisionErrorType.PROVIDER_ERROR)

    async def cancel(self, full_job_id: str) -> bool:
        """Cancel job if provider supports it."""
        try:
            provider_name, job_id = full_job_id.split(":", 1)
            provider = self.providers.get(provider_name)

            if provider:
                return await provider.cancel(job_id)

            return False

        except Exception as e:
            self.logger.exception(f"Failed to cancel job {full_job_id}: {e}")
            return False

    def get_supported_tasks(self) -> list[VisionTask]:
        """Get list of all supported tasks across providers [CA]."""
        tasks = set()
        for provider in self.providers.values():
            capabilities = provider.capabilities()
            tasks.update(capabilities.get("modes", []))
        return list(tasks)

    def get_providers_for_task(self, task: VisionTask) -> list[VisionProvider]:
        """Get available providers that support specific task [CA]."""
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

    def get_models_for_task(self, task: VisionTask, provider: VisionProvider | None = None) -> list[str]:
        """Get available models for task, optionally filtered by provider [CA]."""
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

    def get_provider_capabilities(self, provider_name: str) -> dict[str, Any]:
        """Get detailed capabilities for specific provider [PA]."""
        provider = self.providers.get(provider_name)
        if provider:
            return provider.capabilities()
        return {}

    def estimate_cost_for_request(self, request: VisionRequest) -> dict[str, float]:
        """Get cost estimates from all providers for comparison [CMV]."""
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
def create_vision_adapter(config: dict[str, Any]) -> UnifiedVisionAdapter:
    """Create unified vision adapter with configuration [CA]."""
    return UnifiedVisionAdapter(config)
