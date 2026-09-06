"""Vision Gateway - Provider-agnostic facade for image/video generation.

Unified gateway using pluggable provider system with automatic failover,
retry logic, and cost estimation following REH and CA principles.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import aiofiles
import aiohttp

from bot.config import load_config
from bot.exceptions import APIError
from bot.retry_utils import API_RETRY_CONFIG, with_retry
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
from .unified_adapter import UnifiedStatus, UnifiedVisionAdapter

logger = get_logger(__name__)

# Download tuning constants [CMV]
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_SIZE = 8192  # bytes per streamed chunk
MIME_SNIFF_BYTES = 32  # header bytes read for image type detection
_URL_LOG_MAX_CHARS = 100  # truncate non-data URLs before logging

# MIME <-> extension mapping [CMV]
_MIME_TO_EXTENSION = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
# Ordered (content-type token, suffix) pairs for extension inference [CMV]
_CONTENT_TYPE_SUFFIXES = (
    ("png", ".png"),
    ("jpeg", ".jpg"),
    ("jpg", ".jpg"),
    ("webp", ".webp"),
    ("gif", ".gif"),
)


def _detect_image_type_from_bytes(data: bytes) -> str:
    """Detect image MIME type from byte signature [CA]."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:12]:
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return "image/png"  # default fallback


def _get_extension_from_mime(mime_type: str) -> str:
    """Map MIME type to file extension [CMV]."""
    return _MIME_TO_EXTENSION.get(mime_type, ".png")


def _scrub_url(url: str) -> str:
    """Redact base64 data URLs and truncate others before logging [REH]."""
    if isinstance(url, str) and url.startswith("data:"):
        return f"<data-url:{url.split(',')[0]}>"
    return url[:_URL_LOG_MAX_CHARS]


async def _decode_data_image_url(url: str) -> tuple[bytes | None, str | None]:
    """Decode ``data:image/*;base64,...`` URLs off the event loop [PA].

    Returns (bytes, mime_type) or (None, None) if not a decodable data URL.
    """
    try:
        if not isinstance(url, str) or not url.startswith("data:"):
            return None, None
        if ";base64," not in url:
            return None, None
        header, b64 = url.split(",", 1)
        mime = header.split(";", 1)[0].replace("data:", "").strip()
        if not mime.startswith("image/"):
            return None, None
        # Multi-MB base64 payloads decode on a worker thread to avoid blocking [PA]
        data = await asyncio.to_thread(base64.b64decode, b64)
        return data, mime
    except (ValueError, AttributeError, TypeError) as exc:
        logger.debug(f"Data URL decode failed: {exc}")
        return None, None


def _resolve_download_paths(resp: aiohttp.ClientResponse, tmp_path: Path, final_path: Path) -> tuple[Path, Path]:
    """Infer a file suffix from the response content-type when missing [CMV]."""
    content_type = resp.headers.get("content-type", "").lower()
    if not content_type or final_path.suffix:
        return tmp_path, final_path
    for token, suffix in _CONTENT_TYPE_SUFFIXES:
        if token in content_type:
            return tmp_path.with_suffix(suffix), final_path.with_suffix(suffix)
    return tmp_path, final_path


@with_retry(API_RETRY_CONFIG)
async def _download_asset(session: aiohttp.ClientSession, url: str, tmp_path: Path, final_path: Path) -> Path:
    """Stream a remote asset to disk with retries; writes offloaded via aiofiles [REH][PA]."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS)) as resp:
            resp.raise_for_status()
            tmp_path, final_path = _resolve_download_paths(resp, tmp_path, final_path)
            async with aiofiles.open(tmp_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                    await f.write(chunk)
        os.replace(tmp_path, final_path)
        return final_path
    except Exception as e:
        # Normalize to APIError to trigger retries consistently
        raise APIError(str(e)) from e


class VisionGateway:
    """Unified gateway for vision generation tasks using pluggable provider system.

    Handles:
    - Automatic provider selection and fallback [REH]
    - Request normalization and validation [IV]
    - Cost estimation and budget enforcement [CMV]
    - Progress tracking and status mapping [PA]
    - Result standardization and error mapping [CA]
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.logger = get_logger("vision.gateway")

        # Initialize unified adapter
        self.adapter = UnifiedVisionAdapter(self.config)
        self.active_jobs: dict[str, dict[str, Any]] = {}
        self._active_jobs_lock = asyncio.Lock()

        # Shared aiohttp session for asset downloads, lazily created per loop [PA][RM]
        self._download_session: aiohttp.ClientSession | None = None
        self._download_session_loop: asyncio.AbstractEventLoop | None = None
        self._download_session_lock = asyncio.Lock()

        # Initialize pricing table for cost calculations [CA]
        self.pricing_table = get_pricing_table()

        self.logger.info("VisionGateway initialized with unified adapter")

    def update_config(self, config: dict[str, Any]) -> None:
        """Hot-reload gateway and adapter config snapshot."""
        self.config = config
        if hasattr(self.adapter, "update_config"):
            self.adapter.update_config(config)
        else:
            self.adapter.config = config

    async def startup(self) -> None:
        """Initialize gateway and adapter connections [REH]."""
        try:
            await self.adapter.startup()
            self.logger.info("VisionGateway startup complete")
        except Exception as e:
            self.logger.exception(f"Failed to start VisionGateway: {e}")
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"Gateway startup failed: {e}",
                user_message="Vision system could not be initialized. Please try again later.",
            ) from e

    async def shutdown(self) -> None:
        """Cleanup gateway resources [RM]."""
        try:
            await self._close_download_session()
            await self.adapter.shutdown()
            self.logger.info("VisionGateway shutdown complete")
        except Exception as e:
            self.logger.exception(f"Error during VisionGateway shutdown: {e}")

    async def _get_download_session(self) -> aiohttp.ClientSession:
        """Return a shared download session, recreating it if closed or loop changed [PA][RM]."""
        loop = asyncio.get_event_loop()
        async with self._download_session_lock:
            session = self._download_session
            if session is None or session.closed or self._download_session_loop is not loop:
                if session is not None and not session.closed:
                    await session.close()
                timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS)
                self._download_session = aiohttp.ClientSession(timeout=timeout)
                self._download_session_loop = loop
            return self._download_session

    async def _close_download_session(self) -> None:
        """Close the shared download session if open [RM]."""
        async with self._download_session_lock:
            if self._download_session is not None and not self._download_session.closed:
                await self._download_session.close()
            self._download_session = None
            self._download_session_loop = None

    async def submit_job(self, request: VisionRequest) -> str:
        """Submit vision generation job through unified adapter [CA].

        Args:
            request: Vision generation request

        Returns:
            Job ID for tracking progress

        Raises:
            VisionError: On submission failure

        """
        # Initialize job_id early for robust logging and error paths
        # Use the request's idempotency key as a provisional identifier until the provider returns a job id.
        job_id = getattr(request, "idempotency_key", None) or "pending"

        try:
            self.logger.info(f"Submitting {request.task.value} job for user {request.user_id}")

            # Submit through unified adapter
            response = await self.adapter.submit(request)

            # Extract provider job details from VisionResponse
            job_id = response.job_id  # replace provisional id with provider-qualified id
            provider_name = response.provider.value

            # Track job metadata with thread-safe access
            async with self._active_jobs_lock:
                self.active_jobs[job_id] = {
                    "request": request,
                    "provider": provider_name,
                    "start_time": asyncio.get_event_loop().time(),
                    "last_poll": 0,
                }

            self.logger.info(f"Job {job_id} submitted to provider {provider_name}")
            return job_id

        except Exception as e:
            # Clean up failed job with thread-safe access
            async with self._active_jobs_lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]

            # Preserve the adapter's diagnosis. Re-wrapping every failure as a
            # generic PROVIDER_ERROR with "please try again" told users to retry a
            # job that could never succeed (e.g. the provider account is out of
            # credit) and erased the error_type downstream handlers key off. [REH]
            if isinstance(e, VisionError):
                self.logger.error(
                    "vision.gateway.failed job=%s error_type=%s provider=%s message=%s",
                    job_id,
                    getattr(e.error_type, "value", e.error_type),
                    getattr(e.provider, "value", e.provider),
                    e.message,
                )
                raise
            self.logger.error(f"Vision gateway failed for job {job_id}: {e}", exc_info=True)
            raise VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message=f"Vision processing failed: {e!s}",
                user_message="I encountered an error while processing your request. Please try again.",
            ) from e

    def _calculate_actual_cost(self, job_meta: dict[str, Any], result) -> Money:
        """Calculate actual cost using pricing table instead of trusting provider values [CA][REH]."""
        try:
            request = job_meta.get("request")
            if not request:
                return Money("0.006")  # Safe fallback

            # Use pricing table to calculate actual cost (same as estimate)
            provider = VisionProvider(result.provider_used.lower()) if result.provider_used else VisionProvider.NOVITA

            return self.pricing_table.estimate_cost(
                provider=provider,
                task=getattr(request, "task", "text_to_image"),
                width=getattr(request, "width", 1024),
                height=getattr(request, "height", 1024),
                num_images=getattr(request, "batch_size", 1) or 1,
                duration_seconds=getattr(request, "duration_seconds", 4.0) or 4.0,
                model=getattr(request, "preferred_model", None) or getattr(request, "model", None),
            )
        except (AttributeError, TypeError, ValueError) as e:
            self.logger.warning(f"Actual cost calculation failed, using fallback: {e}")
            return Money("0.006")

    async def generate(self, request: VisionRequest) -> VisionResponse:
        """Direct generation method - submit job and wait for completion [CA].

        Args:
            request: Vision generation request

        Returns:
            VisionResponse with generated content

        Raises:
            VisionError: On generation failure

        """
        # Exception-safe scoping - initialize at top level
        job_id = None
        reservation = None

        try:
            # Submit job
            job_id = await self.submit_job(request)

            # Poll until completion
            max_wait_seconds = 300  # 5 minutes timeout
            poll_interval = 2.0  # Start with 2 second intervals
            elapsed = 0

            while elapsed < max_wait_seconds:
                status = await self.get_job_status(job_id)
                if not status:
                    raise VisionError(
                        error_type=VisionErrorType.SYSTEM_ERROR,
                        message="Lost track of job status",
                        user_message="Generation tracking failed. Please try again.",
                    )

                if status.get("is_terminal", False):
                    if status.get("state") == "completed":
                        # Get final result
                        result = await self.get_job_result(job_id)
                        if result:
                            return result
                        raise VisionError(
                            error_type=VisionErrorType.PROVIDER_ERROR,
                            message="Job completed but no result available",
                            user_message="Generation completed but result could not be retrieved.",
                        )
                    # Job failed
                    error_msg = status.get("progress_message", "Generation failed")
                    raise VisionError(
                        error_type=VisionErrorType.PROVIDER_ERROR,
                        message=f"Generation failed: {error_msg}",
                        user_message="Image generation failed. Please try again.",
                    )

                # Wait before next poll (exponential backoff)
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                poll_interval = min(poll_interval * 1.2, 10.0)  # Cap at 10 seconds

            # Timeout
            if job_id is not None:
                await self.cancel_job(job_id)
            raise VisionError(
                error_type=VisionErrorType.TIMEOUT_ERROR,
                message=f"Generation timed out after {max_wait_seconds} seconds",
                user_message="Image generation is taking too long. Please try again.",
            )

        except VisionError:
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error in generate(): {e}")
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"Unexpected error: {e}",
                user_message="An unexpected error occurred during generation.",
            ) from e
        finally:
            # Clean up in finally block - safe to reference job_id here
            if reservation is not None:
                # Release reservation if it existed (future budget integration)
                pass

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get current job status through unified adapter [PA].

        Args:
            job_id: Job identifier

        Returns:
            Status dictionary with progress, phase, costs, etc.

        """
        if job_id not in self.active_jobs:
            return None

        try:
            # Poll through unified adapter
            status = await self.adapter.poll(job_id)

            # Update last poll time with thread-safe access
            async with self._active_jobs_lock:
                self.active_jobs[job_id]["last_poll"] = asyncio.get_event_loop().time()

            # Convert unified status to gateway format
            return {
                "job_id": job_id,
                "state": self._map_status_to_state(status.status),
                "progress_percentage": status.progress_percentage,
                "progress_message": status.phase,
                "estimated_cost": status.estimated_cost,
                "actual_cost": status.actual_cost,
                "warnings": status.warnings,
                "is_terminal": status.status
                in [
                    UnifiedStatus.COMPLETED,
                    UnifiedStatus.FAILED,
                    UnifiedStatus.CANCELLED,
                ],
            }

        except Exception as e:
            self.logger.exception(f"Failed to get status for job {job_id}: {e}")
            return {
                "job_id": job_id,
                "state": "failed",
                "progress_percentage": 0,
                "progress_message": f"Status check failed: {e}",
                "is_terminal": True,
            }

    async def get_job_result(self, job_id: str) -> VisionResponse | None:
        """Get final job result through unified adapter [CA].

        Args:
            job_id: Job identifier

        Returns:
            VisionResponse with generated content or None if not ready

        """
        if job_id not in self.active_jobs:
            return None

        try:
            status = await self.adapter.poll(job_id)
            if status.status != UnifiedStatus.COMPLETED:
                return None

            result = await self.adapter.fetch_result(job_id)
            job_meta = self.active_jobs[job_id]
            assets_urls: list[str] = result.assets or []
            artifacts_dir = self._make_artifacts_dir(job_id)

            saved_artifacts, total_size, warnings = await self._download_assets(assets_urls, artifacts_dir, job_id)
            response = self._build_response(job_id, result, job_meta, saved_artifacts, total_size, warnings)

            async with self._active_jobs_lock:
                del self.active_jobs[job_id]

            self.logger.info(f"Job {job_id} completed successfully; assets_saved={len(saved_artifacts)}/{len(assets_urls)} dir={artifacts_dir}")
            return response

        except Exception as e:
            self.logger.exception(f"Failed to get result for job {job_id}: {e}")
            # Clean up failed job with thread-safe access
            async with self._active_jobs_lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
            return None

    def _make_artifacts_dir(self, job_id: str) -> Path:
        """Create a unique per-job artifacts directory to prevent collisions [RM]."""
        artifacts_dir = Path(self.config.get("VISION_ARTIFACTS_DIR", "vision_artifacts")) / f"{job_id}_{int(time.time())}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created artifacts directory: {artifacts_dir} for job {job_id[:8]}")
        return artifacts_dir

    async def _download_assets(self, assets_urls: list[str], artifacts_dir: Path, job_id: str) -> tuple[list[Path], int, list[str]]:
        """Persist all result assets locally → (paths, total_bytes, warnings) [REH][RM]."""
        saved_artifacts: list[Path] = []
        warnings: list[str] = []
        total_size = 0
        session = await self._get_download_session()

        for idx, url in enumerate(assets_urls):
            try:
                saved = await self._process_asset(session, url, artifacts_dir, job_id, idx)
                if saved is not None:
                    saved_artifacts.append(saved)
                    total_size += saved.stat().st_size
                else:
                    self.logger.warning(f"Failed to download artifact {idx} for job {job_id}: {_scrub_url(url)}")
            except Exception as e:  # noqa: BLE001 - per-asset fan-out; logged, warnings collected, loop continues
                warnings.append(f"Asset download failed: {e}")
                self.logger.warning(f"Asset download failed (job_id={job_id}, url={_scrub_url(url)}): {e}")

        return saved_artifacts, total_size, warnings

    async def _process_asset(self, session: aiohttp.ClientSession, url: str, artifacts_dir: Path, job_id: str, idx: int) -> Path | None:
        """Persist a single asset (inline data URL or remote download) [CA]."""
        # OpenRouter may return data URLs for images; decode locally without HTTP
        saved = await self._save_data_url_asset(url, artifacts_dir, job_id, idx)
        if saved is not None:
            return saved
        return await self._download_asset_with_mime(session, url, artifacts_dir, job_id, idx)

    async def _save_data_url_asset(self, url: str, artifacts_dir: Path, job_id: str, idx: int) -> Path | None:
        """Decode and persist a base64 data-URL image, offloading blocking work [PA]."""
        data_bytes, data_mime = await _decode_data_image_url(url)
        if data_bytes is None or data_mime is None:
            return None
        final_path = artifacts_dir / f"generated_{job_id}_{idx}{_get_extension_from_mime(data_mime)}"
        async with aiofiles.open(final_path, "wb") as f:
            await f.write(data_bytes)
        self.logger.info(f"Artifact saved from data URL for job {job_id}: {final_path} ({data_mime})")
        return final_path

    async def _download_asset_with_mime(self, session: aiohttp.ClientSession, url: str, artifacts_dir: Path, job_id: str, idx: int) -> Path | None:
        """Download a remote asset then rename it with a sniffed extension [REH]."""
        parsed = urlparse(url)
        base_name = unquote(os.path.basename(parsed.path)) or f"generated_{job_id}_{idx}"
        if "." in base_name:
            base_name = base_name.rsplit(".", 1)[0]  # strip existing ext for clean detection

        tmp_path = artifacts_dir / f".{base_name}.part"
        final_path = artifacts_dir / f"{base_name}.tmp"  # temp name for detection
        saved = await _download_asset(session, url, tmp_path, final_path)
        if not (saved and saved.exists()):
            return None
        return await self._apply_sniffed_extension(saved, artifacts_dir, base_name, job_id)

    async def _apply_sniffed_extension(self, saved: Path, artifacts_dir: Path, base_name: str, job_id: str) -> Path:
        """Sniff MIME from header bytes (read off-loop) and rename accordingly [PA]."""
        async with aiofiles.open(saved, "rb") as f:
            header_bytes = await f.read(MIME_SNIFF_BYTES)

        detected_mime = _detect_image_type_from_bytes(header_bytes)
        proper_final_path = artifacts_dir / f"{base_name}{_get_extension_from_mime(detected_mime)}"
        if saved != proper_final_path:
            os.replace(saved, proper_final_path)
            saved = proper_final_path

        self.logger.info(f"Artifact saved with MIME detection for job {job_id}: {saved} ({detected_mime})")
        return saved

    def _build_response(self, job_id: str, result, job_meta: dict[str, Any], saved_artifacts: list[Path], total_size: int, warnings: list[str]) -> VisionResponse:
        """Assemble the VisionResponse from downloaded local assets [CA]."""
        provider = VisionProvider(result.provider_used.lower()) if result.provider_used else VisionProvider.NOVITA
        return VisionResponse(
            success=True,
            job_id=job_id,
            provider=provider,
            model_used=result.metadata.get("model", "unknown"),
            artifacts=saved_artifacts,
            processing_time_seconds=asyncio.get_event_loop().time() - job_meta["start_time"],
            actual_cost=self._calculate_actual_cost(job_meta, result),
            file_size_bytes=total_size,
            warnings=warnings,
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel running job [REH].

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled successfully

        """
        if job_id not in self.active_jobs:
            return False

        try:
            success = await self.adapter.cancel(job_id)
            if success:
                async with self._active_jobs_lock:
                    if job_id in self.active_jobs:
                        del self.active_jobs[job_id]
                self.logger.info(f"Job {job_id} cancelled")
            return success

        except Exception as e:
            self.logger.exception(f"Failed to cancel job {job_id}: {e}")
            return False

    def _map_status_to_state(self, status: UnifiedStatus) -> str:
        """Map unified status to gateway state format [CMV]."""
        mapping = {
            UnifiedStatus.QUEUED: "queued",
            UnifiedStatus.RUNNING: "processing",
            UnifiedStatus.UPSCALING: "processing",
            UnifiedStatus.SAFETY_REVIEW: "processing",
            UnifiedStatus.UPLOADING: "finalizing",
            UnifiedStatus.COMPLETED: "completed",
            UnifiedStatus.FAILED: "failed",
            UnifiedStatus.CANCELLED: "cancelled",
        }
        return mapping.get(status, "unknown")

    def get_supported_tasks(self) -> list[VisionTask]:
        """Get list of supported tasks from unified adapter."""
        return self.adapter.get_supported_tasks()

    def get_providers_for_task(self, task: VisionTask) -> list[VisionProvider]:
        """Get available providers for specific task from unified adapter."""
        return self.adapter.get_providers_for_task(task)

    def get_models_for_task(self, task: VisionTask, provider: VisionProvider | None = None) -> list[str]:
        """Get available models for task from unified adapter."""
        return self.adapter.get_models_for_task(task, provider)
