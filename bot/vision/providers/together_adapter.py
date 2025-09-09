"""
Together.ai Vision Provider Adapter

Implements Together.ai API integration for vision generation tasks.
Supports text-to-image and image-to-image generation with Stable Diffusion models.
"""

import aiohttp
import time
from pathlib import Path
from typing import Dict, Any, Optional
import base64

from bot.utils.logging import get_logger
from ..types import (
    VisionRequest,
    VisionResponse,
    VisionProvider,
    VisionTask,
    VisionError,
    VisionErrorType,
)
from .base import BaseVisionProvider

logger = get_logger(__name__)


class TogetherAdapter(BaseVisionProvider):
    """
    Together.ai API adapter for vision generation

    Supports:
    - Text-to-image generation with SDXL and SD 1.5
    - Image-to-image editing and variations
    - Batch generation for compatible models
    - Proper error mapping and retry handling [REH]
    """

    def __init__(self, config: Dict[str, Any], policy: Dict[str, Any]):
        super().__init__(config, policy)

        self.api_key = config["VISION_API_KEY"]
        self.base_url = "https://api.together.xyz/v1"
        self.session: Optional[aiohttp.ClientSession] = None

        # Together-specific timeouts
        self.request_timeout = 60  # seconds
        self.long_timeout = 300  # for video generation

    def get_provider_name(self) -> VisionProvider:
        return VisionProvider.TOGETHER

    def _validate_config(self) -> None:
        """Validate Together.ai specific configuration [IV]"""
        if not self.config.get("VISION_API_KEY"):
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message="VISION_API_KEY not configured for Together.ai",
                user_message="Vision generation is not properly configured.",
            )

    def _initialize(self) -> None:
        """Initialize HTTP session and validate API access"""
        self.logger.info("Initializing Together.ai adapter")

        # HTTP session will be created lazily in _get_session()
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with proper headers [RM]"""
        if self.session is None or self.session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Discord-LLM-Chatbot/1.0",
            }

            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10),
            )

        return self.session

    async def generate(self, request: VisionRequest, model: str) -> VisionResponse:
        """Generate vision content using Together.ai API"""
        start_time = time.time()

        self._log_request_start(request, model)

        try:
            if request.task == VisionTask.TEXT_TO_IMAGE:
                response = await self._text_to_image(request, model)
            elif request.task == VisionTask.IMAGE_TO_IMAGE:
                response = await self._image_to_image(request, model)
            else:
                raise VisionError(
                    error_type=VisionErrorType.VALIDATION_ERROR,
                    message=f"Task {request.task.value} not supported by Together.ai",
                    user_message=f"Sorry, {request.task.value} is not yet available with Together.ai.",
                )

            processing_time = time.time() - start_time
            response.processing_time_seconds = processing_time

            if response.success:
                self._log_request_complete(request, response, processing_time)

            return response

        except VisionError as e:
            processing_time = time.time() - start_time
            self._log_request_error(request, e, processing_time)
            return self._create_error_response(
                request.idempotency_key, e, processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error = VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message=f"Unexpected Together.ai error: {str(e)}",
                user_message="An unexpected error occurred. Please try again.",
                provider=VisionProvider.TOGETHER,
            )
            self._log_request_error(request, error, processing_time)
            return self._create_error_response(
                request.idempotency_key, error, processing_time
            )

    async def _text_to_image(
        self, request: VisionRequest, model: str
    ) -> VisionResponse:
        """Generate image from text prompt"""
        session = await self._get_session()

        # Prepare request payload
        payload = {
            "model": model,
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "n": request.batch_size,
            "response_format": "b64_json",  # Get base64 encoded images
        }

        # Add optional parameters
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt

        if request.seed is not None:
            payload["seed"] = request.seed

        if request.guidance_scale != 7.0:  # Only send if non-default
            payload["guidance_scale"] = request.guidance_scale

        # Make API request
        try:
            async with session.post(
                f"{self.base_url}/images/generations", json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return await self._process_image_response(data, request)
                else:
                    error_data = (
                        await resp.json()
                        if resp.content_type == "application/json"
                        else {}
                    )
                    raise await self._map_api_error(resp.status, error_data)

        except aiohttp.ClientError as e:
            raise VisionError(
                error_type=VisionErrorType.NETWORK_ERROR,
                message=f"Network error: {str(e)}",
                user_message="Network connection failed. Please check your connection and try again.",
                provider=VisionProvider.TOGETHER,
            )

    async def _image_to_image(
        self, request: VisionRequest, model: str
    ) -> VisionResponse:
        """Edit image using image-to-image pipeline"""
        if not request.input_image or not request.input_image.exists():
            raise VisionError(
                error_type=VisionErrorType.VALIDATION_ERROR,
                message="Input image required for image-to-image generation",
                user_message="Please provide an input image for editing.",
            )

        # Validate image format and size
        self._validate_image_format(request.input_image)
        self._validate_file_size(request.input_image, 25)  # 25MB limit

        session = await self._get_session()

        # Convert image to base64
        image_b64 = await self._image_to_base64(request.input_image)

        payload = {
            "model": model,
            "prompt": request.prompt,
            "image": image_b64,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "strength": request.strength,
            "response_format": "b64_json",
        }

        # Add optional parameters
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt

        if request.seed is not None:
            payload["seed"] = request.seed

        if request.guidance_scale != 7.0:
            payload["guidance_scale"] = request.guidance_scale

        # Add mask if provided (for inpainting)
        if request.mask_image and request.mask_image.exists():
            mask_b64 = await self._image_to_base64(request.mask_image)
            payload["mask"] = mask_b64

        try:
            async with session.post(
                f"{self.base_url}/images/edits", json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return await self._process_image_response(data, request)
                else:
                    error_data = (
                        await resp.json()
                        if resp.content_type == "application/json"
                        else {}
                    )
                    raise await self._map_api_error(resp.status, error_data)

        except aiohttp.ClientError as e:
            raise VisionError(
                error_type=VisionErrorType.NETWORK_ERROR,
                message=f"Network error: {str(e)}",
                user_message="Network connection failed. Please try again.",
                provider=VisionProvider.TOGETHER,
            )

    async def _process_image_response(
        self, data: Dict[str, Any], request: VisionRequest
    ) -> VisionResponse:
        """Process API response and save generated images"""
        artifacts_dir = Path(self.config["VISION_ARTIFACTS_DIR"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        job_id = request.idempotency_key
        artifacts = []

        # Extract images from response
        images_data = data.get("data", [])
        if not images_data:
            raise VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message="No images returned from Together.ai",
                user_message="No images were generated. Please try again.",
                provider=VisionProvider.TOGETHER,
            )

        # Save each generated image
        for i, image_data in enumerate(images_data):
            b64_image = image_data.get("b64_json")
            if not b64_image:
                continue

            # Decode base64 image
            image_bytes = base64.b64decode(b64_image)

            # Save to file
            filename = f"{job_id}_{i + 1}.png"
            file_path = artifacts_dir / filename

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            artifacts.append(file_path)

            self.logger.debug(f"Saved generated image: {file_path}")

        # Calculate actual cost (simplified estimation)
        model_config = self._get_model_config(
            "text_to_image", data.get("model", "unknown")
        )
        base_cost = (
            model_config.get("estimated_cost_per_image", 0.04) if model_config else 0.04
        )
        actual_cost = base_cost * len(artifacts)

        # Extract dimensions from first image
        dimensions = None
        if artifacts:
            dimensions = self._extract_dimensions_from_path(artifacts[0])

        return VisionResponse(
            success=True,
            job_id=job_id,
            provider=VisionProvider.TOGETHER,
            model_used=data.get("model", "unknown"),
            artifacts=artifacts,
            actual_cost=actual_cost,
            dimensions=dimensions,
            file_size_bytes=self._calculate_file_size(artifacts),
        )

    async def _image_to_base64(self, image_path: Path) -> str:
        """Convert image file to base64 string [RM]"""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            raise VisionError(
                error_type=VisionErrorType.VALIDATION_ERROR,
                message=f"Failed to read image file: {str(e)}",
                user_message="Could not read the input image. Please try uploading again.",
            )

    async def _map_api_error(
        self, status_code: int, error_data: Dict[str, Any]
    ) -> VisionError:
        """Map Together.ai API errors to VisionError [REH]"""
        error_message = error_data.get("error", {}).get("message", "Unknown API error")
        error_data.get("error", {}).get("type", "")

        # Map specific error types
        if status_code == 400:
            if (
                "content policy" in error_message.lower()
                or "safety" in error_message.lower()
            ):
                return VisionError(
                    error_type=VisionErrorType.CONTENT_FILTERED,
                    message=f"Content filtered: {error_message}",
                    user_message="Your prompt was blocked by content safety filters. Please try a different prompt.",
                    provider=VisionProvider.TOGETHER,
                )
            else:
                return VisionError(
                    error_type=VisionErrorType.VALIDATION_ERROR,
                    message=f"Invalid request: {error_message}",
                    user_message="There was a problem with your request. Please check your parameters.",
                    provider=VisionProvider.TOGETHER,
                )

        elif status_code == 401:
            return VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message="Authentication failed with Together.ai",
                user_message="Authentication error. Please contact an administrator.",
                provider=VisionProvider.TOGETHER,
            )

        elif status_code == 429:
            return VisionError(
                error_type=VisionErrorType.QUOTA_EXCEEDED,
                message="Rate limit or quota exceeded",
                user_message="Rate limit exceeded. Please wait a moment before trying again.",
                provider=VisionProvider.TOGETHER,
                retry_after_seconds=60,
            )

        elif status_code >= 500:
            return VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message=f"Together.ai server error: {error_message}",
                user_message="The service is temporarily unavailable. Please try again later.",
                provider=VisionProvider.TOGETHER,
                retry_after_seconds=30,
            )

        else:
            return VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message=f"Together.ai API error {status_code}: {error_message}",
                user_message="An error occurred with the generation service. Please try again.",
                provider=VisionProvider.TOGETHER,
            )

    async def get_job_status(self, provider_job_id: str) -> Dict[str, Any]:
        """Together.ai uses synchronous API, so jobs complete immediately"""
        return {"status": "completed", "progress": 100, "completed": True}

    async def cancel_job(self, provider_job_id: str) -> bool:
        """Together.ai uses synchronous API, so no cancellation needed"""
        return True  # Always "successful" since jobs complete immediately

    async def close(self) -> None:
        """Clean up resources [RM]"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("Closed Together.ai HTTP session")
