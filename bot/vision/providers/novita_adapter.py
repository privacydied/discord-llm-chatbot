"""
Novita.ai Vision Provider Adapter

Implements Novita.ai API integration for comprehensive vision generation.
Supports text-to-image, text-to-video, and image-to-video generation.
"""

import asyncio
import aiohttp
import time
from pathlib import Path
from typing import Dict, Any, Optional

from bot.util.logging import get_logger
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


class NovitaAdapter(BaseVisionProvider):
    """
    Novita.ai API adapter for comprehensive vision generation

    Supports:
    - Text-to-image with DreamShaper and other models
    - Text-to-video with Kling V1.6, Vidu Q1, and others
    - Image-to-video with various animation models
    - Async job polling for long-running video tasks [PA]
    """

    def __init__(self, config: Dict[str, Any], policy: Dict[str, Any]):
        super().__init__(config, policy)

        self.api_key = config["VISION_API_KEY"]
        self.base_url = "https://api.novita.ai/v3"
        self.session: Optional[aiohttp.ClientSession] = None

        # Novita-specific settings
        self.polling_interval = 5  # seconds
        self.max_poll_attempts = 120  # 10 minutes max

    def _get_model_config(self) -> VisionProvider:
        return VisionProvider.NOVITA

    def get_provider_name(self) -> VisionProvider:
        return VisionProvider.NOVITA

    def _normalize_request_for_novita(
        self, request: VisionRequest, task_type: str
    ) -> Dict[str, Any]:
        """Normalize and validate request parameters for Novita.ai to prevent 400 errors [IV][CMV]"""
        # Start with clean payload - remove None/empty values
        payload = {}

        # Normalize numeric parameters with safe defaults
        if hasattr(request, "steps") and request.steps is not None:
            payload["steps"] = min(max(request.steps, 1), 100)  # Clamp to Novita range

        if hasattr(request, "guidance_scale") and request.guidance_scale is not None:
            payload["guidance_scale"] = min(
                max(request.guidance_scale, 1.0), 20.0
            )  # Clamp to Novita range

        # Normalize seed if provided
        if hasattr(request, "seed") and request.seed is not None:
            try:
                payload["seed"] = int(request.seed)
            except (ValueError, TypeError):
                pass  # Skip invalid seed

        # Handle negative_prompt as list (stringify for Novita)
        if hasattr(request, "negative_prompt") and request.negative_prompt:
            if isinstance(request.negative_prompt, list):
                payload["negative_prompt"] = ", ".join(request.negative_prompt)
            else:
                payload["negative_prompt"] = str(request.negative_prompt)

        # Task-specific normalization
        if task_type == "text_to_image":
            # Normalize dimensions to multiple of 8, within Novita range
            width = min(max(request.width or 1024, 256), 2048)
            height = min(max(request.height or 1024, 256), 2048)
            width = ((width + 7) // 8) * 8  # Round to nearest 8
            height = ((height + 7) // 8) * 8
            payload["size"] = f"{width}*{height}"

            if hasattr(request, "prompt") and request.prompt:
                payload["prompt"] = request.prompt

        elif task_type == "text_to_video":
            # Video-specific parameters
            if hasattr(request, "duration_seconds") and request.duration_seconds:
                payload["duration"] = (
                    f"{min(max(request.duration_seconds, 1), 10)}s"  # Novita range 1-10s
                )

            if hasattr(request, "fps") and request.fps:
                payload["fps"] = min(max(request.fps, 8), 30)  # Novita FPS range

            if hasattr(request, "prompt") and request.prompt:
                payload["prompt"] = request.prompt

        elif task_type == "image_to_video":
            # Image-to-video parameters
            if hasattr(request, "duration_seconds") and request.duration_seconds:
                payload["duration"] = f"{min(max(request.duration_seconds, 1), 10)}s"

            if hasattr(request, "fps") and request.fps:
                payload["fps"] = min(max(request.fps, 8), 30)

            if hasattr(request, "prompt") and request.prompt:
                payload["prompt"] = request.prompt

        # Remove any remaining None values (defensive programming)
        payload = {k: v for k, v in payload.items() if v is not None}

        # Debug log normalized payload (no secrets, just keys and ranges)
        debug_payload = {
            k: "..." if k in ["prompt", "negative_prompt"] else v
            for k, v in payload.items()
        }
        self.logger.debug(f"Normalized Novita request for {task_type}: {debug_payload}")

        return payload

    def _validate_config(self) -> None:
        """Validate Novita.ai specific configuration [IV]"""
        if not self.config.get("VISION_API_KEY"):
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message="VISION_API_KEY not configured for Novita.ai",
                user_message="Vision generation is not properly configured.",
            )

    def _initialize(self) -> None:
        """Initialize HTTP session and validate API access"""
        self.logger.info("Initializing Novita.ai adapter")
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with proper headers [RM]"""
        if self.session is None or self.session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Discord-LLM-Chatbot/1.0",
            }

            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for video
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10),
            )

        return self.session

    async def generate(self, request: VisionRequest, model: str) -> VisionResponse:
        """Generate vision content using Novita.ai API"""
        start_time = time.time()

        self._log_request_start(request, model)

        try:
            if request.task == VisionTask.TEXT_TO_IMAGE:
                response = await self._text_to_image(request, model)
            elif request.task == VisionTask.TEXT_TO_VIDEO:
                response = await self._text_to_video(request, model)
            elif request.task == VisionTask.IMAGE_TO_VIDEO:
                response = await self._image_to_video(request, model)
            else:
                raise VisionError(
                    error_type=VisionErrorType.VALIDATION_ERROR,
                    message=f"Task {request.task.value} not supported by Novita.ai",
                    user_message=f"Sorry, {request.task.value} is not yet available with Novita.ai.",
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
                message=f"Unexpected Novita.ai error: {str(e)}",
                user_message="An unexpected error occurred. Please try again.",
                provider=VisionProvider.NOVITA,
            )
            self._log_request_error(request, error, processing_time)
            return self._create_error_response(
                request.idempotency_key, error, processing_time
            )

    async def _text_to_image(
        self, request: VisionRequest, model: str
    ) -> VisionResponse:
        """Generate image from text using Novita.ai Qwen Image API"""
        session = await self._get_session()

        # Use normalized request to prevent 400 errors
        payload = self._normalize_request_for_novita(request, "text_to_image")

        try:
            async with session.post(
                f"{self.base_url}/async/qwen-image-txt2img", json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    task_id = data.get("task_id")

                    if not task_id:
                        raise VisionError(
                            error_type=VisionErrorType.PROVIDER_ERROR,
                            message="No task_id returned from Novita.ai",
                            user_message="Failed to start image generation.",
                            provider=VisionProvider.NOVITA,
                        )

                    # Poll for completion
                    result = await self._poll_for_completion(task_id)
                    return await self._process_image_result(result, request)

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
                provider=VisionProvider.NOVITA,
            )

    async def _text_to_video(
        self, request: VisionRequest, model: str
    ) -> VisionResponse:
        """Generate video from text using Novita.ai"""
        session = await self._get_session()

        # Map model names to Novita.ai format
        model_mapping = {
            "kling-v1.6": "kling-v1-6-txt2video",
            "vidu-q1": "vidu-q1-txt2video",
        }

        novita_model = model_mapping.get(model, model)

        # Use normalized request for video parameters
        payload = self._normalize_request_for_novita(request, "text_to_video")
        payload["model_name"] = novita_model

        # Add resolution based on normalized dimensions
        width = min(max(request.width or 1024, 256), 2048)
        height = min(max(request.height or 1024, 256), 2048)
        if width == 1280 and height == 720:
            payload["resolution"] = "720p"
        elif width == 1920 and height == 1080:
            payload["resolution"] = "1080p"
        else:
            payload["resolution"] = "720p"  # Default fallback

        # Add seed if not already set by normalizer
        if (
            "seed" not in payload
            and hasattr(request, "seed")
            and request.seed is not None
        ):
            try:
                payload["seed"] = int(request.seed)
            except (ValueError, TypeError):
                payload["seed"] = -1  # Novita default

        try:
            async with session.post(
                f"{self.base_url}/async/txt2video", json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    task_id = data.get("task_id")

                    if not task_id:
                        raise VisionError(
                            error_type=VisionErrorType.PROVIDER_ERROR,
                            message="No task_id returned from Novita.ai",
                            user_message="Failed to start video generation.",
                            provider=VisionProvider.NOVITA,
                        )

                    # Poll for completion (video takes much longer)
                    result = await self._poll_for_completion(
                        task_id, max_attempts=self.max_poll_attempts
                    )
                    return await self._process_video_result(result, request)

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
                provider=VisionProvider.NOVITA,
            )

    async def _image_to_video(
        self, request: VisionRequest, model: str
    ) -> VisionResponse:
        """Generate video from image using Novita.ai"""
        if not request.input_image or not request.input_image.exists():
            raise VisionError(
                error_type=VisionErrorType.VALIDATION_ERROR,
                message="Input image required for image-to-video generation",
                user_message="Please provide an input image for video generation.",
            )

        # Validate image
        self._validate_image_format(request.input_image)
        self._validate_file_size(request.input_image, 25)

        session = await self._get_session()

        # Upload image and get URL
        image_url = await self._upload_image(request.input_image)

        # Map model names
        model_mapping = {"kling-v1.6-img2vid": "kling-v1-6-img2video"}

        novita_model = model_mapping.get(model, model)

        # Use normalized request for image-to-video parameters
        payload = self._normalize_request_for_novita(request, "image_to_video")
        payload["model_name"] = novita_model
        payload["image_url"] = image_url

        # Add seed if not already set by normalizer
        if (
            "seed" not in payload
            and hasattr(request, "seed")
            and request.seed is not None
        ):
            try:
                payload["seed"] = int(request.seed)
            except (ValueError, TypeError):
                payload["seed"] = -1  # Novita default

        # Handle start/end mode if end image provided
        if (
            request.mode == "start_end"
            and request.end_image
            and request.end_image.exists()
        ):
            end_image_url = await self._upload_image(request.end_image)
            payload["end_image_url"] = end_image_url

        try:
            async with session.post(
                f"{self.base_url}/async/img2video", json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    task_id = data.get("task_id")

                    if not task_id:
                        raise VisionError(
                            error_type=VisionErrorType.PROVIDER_ERROR,
                            message="No task_id returned from Novita.ai",
                            user_message="Failed to start video generation.",
                            provider=VisionProvider.NOVITA,
                        )

                    result = await self._poll_for_completion(
                        task_id, max_attempts=self.max_poll_attempts
                    )
                    return await self._process_video_result(result, request)

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
                provider=VisionProvider.NOVITA,
            )

    async def _upload_image(self, image_path: Path) -> str:
        """Upload image to Novita.ai and return URL [RM]"""
        session = await self._get_session()

        with open(image_path, "rb") as f:
            image_data = f.read()

        # Create form data for file upload
        data = aiohttp.FormData()
        data.add_field("file", image_data, filename=image_path.name)

        try:
            async with session.post(f"{self.base_url}/asset/file", data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    image_url = result.get("download_url")

                    if not image_url:
                        raise VisionError(
                            error_type=VisionErrorType.PROVIDER_ERROR,
                            message="Failed to get image URL from upload",
                            user_message="Failed to upload image. Please try again.",
                        )

                    return image_url
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
                message=f"Image upload failed: {str(e)}",
                user_message="Failed to upload image. Please try again.",
            )

    async def _poll_for_completion(
        self, task_id: str, max_attempts: int = 60
    ) -> Dict[str, Any]:
        """Poll Novita.ai task until completion [PA]"""
        session = await self._get_session()

        for attempt in range(max_attempts):
            try:
                async with session.get(
                    f"{self.base_url}/async/task-result?task_id={task_id}"
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()

                        status = result.get("task", {}).get("status")

                        if status == "TASK_STATUS_SUCCEED":
                            self.logger.debug(f"Task {task_id} completed successfully")
                            return result
                        elif status in ["TASK_STATUS_FAILED", "TASK_STATUS_CANCELED"]:
                            error_msg = result.get("task", {}).get(
                                "reason", "Task failed"
                            )
                            raise VisionError(
                                error_type=VisionErrorType.PROVIDER_ERROR,
                                message=f"Novita.ai task failed: {error_msg}",
                                user_message="Generation failed. Please try again with different parameters.",
                                provider=VisionProvider.NOVITA,
                            )
                        elif status in [
                            "TASK_STATUS_PROCESSING",
                            "TASK_STATUS_QUEUED",
                            "TASK_STATUS_RUNNING",
                            "TASK_STATUS_PENDING",
                        ]:
                            # Still processing, wait and retry
                            progress = result.get("task", {}).get("progress", 0)
                            self.logger.debug(
                                f"Task {task_id} status: {status}, progress: {progress}%"
                            )
                            await asyncio.sleep(self.polling_interval)
                            continue
                        else:
                            # Unknown status - log for debugging and treat as processing
                            self.logger.warning(
                                f"Unknown Novita status '{status}' for task {task_id}, treating as processing"
                            )
                            await asyncio.sleep(self.polling_interval)
                            continue

                    elif resp.status == 404:
                        raise VisionError(
                            error_type=VisionErrorType.PROVIDER_ERROR,
                            message=f"Task {task_id} not found",
                            user_message="Generation task not found. Please try again.",
                            provider=VisionProvider.NOVITA,
                        )
                    else:
                        error_data = (
                            await resp.json()
                            if resp.content_type == "application/json"
                            else {}
                        )
                        raise await self._map_api_error(resp.status, error_data)

            except aiohttp.ClientError as e:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self.polling_interval)
                    continue
                else:
                    raise VisionError(
                        error_type=VisionErrorType.NETWORK_ERROR,
                        message=f"Polling failed: {str(e)}",
                        user_message="Network error while checking generation status.",
                        provider=VisionProvider.NOVITA,
                    )

        # Max attempts reached
        raise VisionError(
            error_type=VisionErrorType.TIMEOUT_ERROR,
            message=f"Task {task_id} timed out after {max_attempts * self.polling_interval} seconds",
            user_message="Generation is taking too long. Please try with simpler parameters.",
            provider=VisionProvider.NOVITA,
        )

    async def _process_image_result(
        self, result: Dict[str, Any], request: VisionRequest
    ) -> VisionResponse:
        """Process completed image generation result"""
        artifacts_dir = Path(self.config["VISION_ARTIFACTS_DIR"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        job_id = request.idempotency_key
        artifacts = []

        # Extract images from result
        images = result.get("images", [])
        if not images:
            raise VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message="No images in Novita.ai result",
                user_message="No images were generated. Please try again.",
                provider=VisionProvider.NOVITA,
            )

        # Download and save each image
        session = await self._get_session()

        for i, image_info in enumerate(images):
            image_url = image_info.get("image_url")
            if not image_url:
                continue

            # Download image
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()

                    # Save to file
                    filename = f"{job_id}_{i + 1}.png"
                    file_path = artifacts_dir / filename

                    with open(file_path, "wb") as f:
                        f.write(image_data)

                    artifacts.append(file_path)

                    self.logger.debug(f"Downloaded image: {file_path}")

        if not artifacts:
            raise VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message="Failed to download generated images",
                user_message="Failed to download generated images. Please try again.",
                provider=VisionProvider.NOVITA,
            )

        # Calculate cost (simplified)
        model_config = self._get_model_config(
            "text_to_image", result.get("model_name", "")
        )
        base_cost = (
            model_config.get("estimated_cost_per_image", 0.03) if model_config else 0.03
        )
        actual_cost = base_cost * len(artifacts)

        # Get dimensions
        dimensions = (
            self._extract_dimensions_from_path(artifacts[0]) if artifacts else None
        )

        return VisionResponse(
            success=True,
            job_id=job_id,
            provider=VisionProvider.NOVITA,
            model_used=result.get("model_name", "unknown"),
            artifacts=artifacts,
            actual_cost=actual_cost,
            dimensions=dimensions,
            file_size_bytes=self._calculate_file_size(artifacts),
            provider_job_id=result.get("task", {}).get("task_id"),
        )

    async def _process_video_result(
        self, result: Dict[str, Any], request: VisionRequest
    ) -> VisionResponse:
        """Process completed video generation result"""
        artifacts_dir = Path(self.config["VISION_ARTIFACTS_DIR"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        job_id = request.idempotency_key

        # Extract video URL from result
        videos = result.get("videos", [])
        if not videos:
            raise VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message="No videos in Novita.ai result",
                user_message="No video was generated. Please try again.",
                provider=VisionProvider.NOVITA,
            )

        video_url = videos[0].get("video_url")
        if not video_url:
            raise VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message="No video URL in Novita.ai result",
                user_message="Video generation failed. Please try again.",
                provider=VisionProvider.NOVITA,
            )

        # Download video
        session = await self._get_session()

        async with session.get(video_url) as resp:
            if resp.status == 200:
                video_data = await resp.read()

                # Save video file
                filename = f"{job_id}.mp4"
                file_path = artifacts_dir / filename

                with open(file_path, "wb") as f:
                    f.write(video_data)

                self.logger.debug(f"Downloaded video: {file_path}")
            else:
                raise VisionError(
                    error_type=VisionErrorType.PROVIDER_ERROR,
                    message=f"Failed to download video: HTTP {resp.status}",
                    user_message="Failed to download generated video. Please try again.",
                    provider=VisionProvider.NOVITA,
                )

        # Calculate cost based on duration
        task_type = (
            "text_to_video"
            if request.task == VisionTask.TEXT_TO_VIDEO
            else "image_to_video"
        )
        model_config = self._get_model_config(task_type, result.get("model_name", ""))
        base_cost_per_second = (
            model_config.get("estimated_cost_per_second", 0.50)
            if model_config
            else 0.50
        )
        actual_cost = base_cost_per_second * request.duration_seconds

        return VisionResponse(
            success=True,
            job_id=job_id,
            provider=VisionProvider.NOVITA,
            model_used=result.get("model_name", "unknown"),
            artifacts=[file_path],
            actual_cost=actual_cost,
            duration_seconds=float(request.duration_seconds),
            file_size_bytes=file_path.stat().st_size,
            provider_job_id=result.get("task", {}).get("task_id"),
        )

    async def _map_api_error(
        self, status_code: int, error_data: Dict[str, Any]
    ) -> VisionError:
        """Map Novita.ai API errors to VisionError [REH]"""
        error_message = error_data.get(
            "msg", error_data.get("message", "Unknown API error")
        )

        if status_code == 400:
            if (
                "inappropriate" in error_message.lower()
                or "violation" in error_message.lower()
            ):
                return VisionError(
                    error_type=VisionErrorType.CONTENT_FILTERED,
                    message=f"Content filtered: {error_message}",
                    user_message="Your prompt was blocked by content safety filters. Please try a different prompt.",
                    provider=VisionProvider.NOVITA,
                )
            else:
                return VisionError(
                    error_type=VisionErrorType.VALIDATION_ERROR,
                    message=f"Invalid request: {error_message}",
                    user_message="There was a problem with your request. Please check your parameters.",
                    provider=VisionProvider.NOVITA,
                )

        elif status_code == 401:
            return VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message="Authentication failed with Novita.ai",
                user_message="Authentication error. Please contact an administrator.",
                provider=VisionProvider.NOVITA,
            )

        elif status_code == 429:
            return VisionError(
                error_type=VisionErrorType.QUOTA_EXCEEDED,
                message="Rate limit exceeded",
                user_message="Rate limit exceeded. Please wait a moment before trying again.",
                provider=VisionProvider.NOVITA,
                retry_after_seconds=60,
            )

        elif status_code >= 500:
            return VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message=f"Novita.ai server error: {error_message}",
                user_message="The service is temporarily unavailable. Please try again later.",
                provider=VisionProvider.NOVITA,
                retry_after_seconds=30,
            )

        else:
            return VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message=f"Novita.ai API error {status_code}: {error_message}",
                user_message="An error occurred with the generation service. Please try again.",
                provider=VisionProvider.NOVITA,
            )

    async def get_job_status(self, provider_job_id: str) -> Dict[str, Any]:
        """Get current status of Novita.ai job"""
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.base_url}/async/task-result?task_id={provider_job_id}"
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    task = result.get("task", {})

                    status = task.get("status", "UNKNOWN")
                    progress = task.get("progress", 0)

                    return {
                        "status": status,
                        "progress": progress,
                        "completed": status == "TASK_STATUS_SUCCEED",
                        "failed": status
                        in ["TASK_STATUS_FAILED", "TASK_STATUS_CANCELED"],
                        "result": result if status == "TASK_STATUS_SUCCEED" else None,
                    }
                else:
                    return {
                        "status": "ERROR",
                        "progress": 0,
                        "completed": False,
                        "failed": True,
                        "error": f"HTTP {resp.status}",
                    }

        except Exception as e:
            return {
                "status": "ERROR",
                "progress": 0,
                "completed": False,
                "failed": True,
                "error": str(e),
            }

    async def poll(self, provider_job_id: str):
        """Poll job status and return UnifiedJobStatus (required by unified adapter)"""
        from ..unified_adapter import UnifiedJobStatus, UnifiedStatus

        try:
            status_data = await self.get_job_status(provider_job_id)
            status = status_data.get("status", "UNKNOWN")
            progress = status_data.get("progress", 0)

            # Map Novita status to unified status
            if status == "TASK_STATUS_SUCCEED":
                unified_status = UnifiedStatus.DONE
            elif status in ["TASK_STATUS_FAILED", "TASK_STATUS_CANCELED"]:
                unified_status = UnifiedStatus.FAILED
            elif status in [
                "TASK_STATUS_PROCESSING",
                "TASK_STATUS_QUEUED",
                "TASK_STATUS_RUNNING",
                "TASK_STATUS_PENDING",
            ]:
                unified_status = UnifiedStatus.RUNNING
            else:
                unified_status = UnifiedStatus.RUNNING  # Default for unknown states

            return UnifiedJobStatus(
                status=unified_status,
                progress_percentage=progress,
                phase=f"Novita: {status.replace('TASK_STATUS_', '').lower()}",
            )

        except Exception as e:
            return UnifiedJobStatus(
                status=UnifiedStatus.FAILED,
                progress_percentage=0,
                phase=f"Polling error: {e}",
            )

    async def cancel_job(self, provider_job_id: str) -> bool:
        """Cancel Novita.ai job if possible"""
        # Novita.ai doesn't currently support job cancellation
        # Return False to indicate cancellation not supported
        return False

    async def close(self) -> None:
        """Clean up resources [RM]"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("Closed Novita.ai HTTP session")
