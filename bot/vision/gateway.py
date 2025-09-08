"""
Vision Gateway - Provider-agnostic facade for image/video generation

Unified gateway using pluggable provider system with automatic failover,
retry logic, and cost estimation following REH and CA principles.
"""

from __future__ import annotations
import asyncio
import json
import os
from urllib.parse import urlparse, unquote
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp

from bot.util.logging import get_logger
from bot.config import load_config
from bot.retry_utils import with_retry, API_RETRY_CONFIG
from bot.exceptions import APIError
from .types import (
    VisionRequest, VisionResponse, VisionTask, VisionProvider, 
    VisionError, VisionErrorType
)
from .unified_adapter import UnifiedVisionAdapter, UnifiedStatus, UnifiedResult
from .money import Money

logger = get_logger(__name__)


class VisionGateway:
    """
    Unified gateway for vision generation tasks using pluggable provider system
    
    Handles:
    - Automatic provider selection and fallback [REH]
    - Request normalization and validation [IV]
    - Cost estimation and budget enforcement [CMV]
    - Progress tracking and status mapping [PA]
    - Result standardization and error mapping [CA]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.logger = get_logger("vision.gateway")
        
        # Initialize unified adapter
        self.adapter = UnifiedVisionAdapter(self.config)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("VisionGateway initialized with unified adapter")
    
    async def startup(self):
        """Initialize gateway and adapter connections [REH]"""
        try:
            await self.adapter.startup()
            self.logger.info("VisionGateway startup complete")
        except Exception as e:
            self.logger.error(f"Failed to start VisionGateway: {e}")
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"Gateway startup failed: {e}",
                user_message="Vision system could not be initialized. Please try again later."
            )
    
    async def shutdown(self):
        """Cleanup gateway resources [RM]"""
        try:
            await self.adapter.shutdown()
            self.logger.info("VisionGateway shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during VisionGateway shutdown: {e}")
    
    async def submit_job(self, request: VisionRequest) -> str:
        """
        Submit vision generation job through unified adapter [CA]
        
        Args:
            request: Vision generation request
            
        Returns:
            Job ID for tracking progress
            
        Raises:
            VisionError: On submission failure
        """
        # Initialize job_id early for robust logging and error paths
        # Use the request's idempotency key as a provisional identifier until the provider returns a job id.
        job_id = getattr(request, 'idempotency_key', None) or "pending"
        
        try:
            self.logger.info(
                f"Submitting {request.task.value} job for user {request.user_id}"
            )
            
            # Submit through unified adapter
            response = await self.adapter.submit(request)
            
            # Extract provider job details from VisionResponse
            job_id = response.job_id  # replace provisional id with provider-qualified id
            provider_name = response.provider.value
            
            # Track job metadata
            self.active_jobs[job_id] = {
                "request": request,
                "provider": provider_name,
                "start_time": asyncio.get_event_loop().time(),
                "last_poll": 0
            }
            
            self.logger.info(f"Job {job_id} submitted to provider {provider_name}")
            return job_id
            
        except Exception as e:
            # Clean up failed job
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            self.logger.error(f"Vision gateway failed for job {job_id}: {e}", exc_info=True)
            raise VisionError(
                error_type=VisionErrorType.PROVIDER_ERROR,
                message=f"Vision processing failed: {str(e)}",
                user_message="I encountered an error while processing your request. Please try again.",
                provider=VisionProvider.NOVITA
            )

    def _calculate_actual_cost(self, job_meta: Dict[str, Any], result) -> Money:
        """Calculate actual cost using pricing table instead of trusting provider values [CA][REH]"""
        try:
            request = job_meta.get("request")
            if not request:
                return Money("0.006")  # Safe fallback
            
            # Use pricing table to calculate actual cost (same as estimate)
            provider = VisionProvider(result.provider_used.lower()) if result.provider_used else VisionProvider.NOVITA
            
            return self.pricing_table.estimate_cost(
                provider=provider,
                task=getattr(request, 'task', 'text_to_image'),
                width=getattr(request, 'width', 1024),
                height=getattr(request, 'height', 1024),
                num_images=getattr(request, 'batch_size', 1) or 1,
                duration_seconds=getattr(request, 'duration_seconds', 4.0) or 4.0,
                model=getattr(request, 'preferred_model', None) or getattr(request, 'model', None)
            )
        except Exception as e:
            self.logger.warning(f"Actual cost calculation failed, using fallback: {e}")
            return Money("0.006")

    async def generate(self, request: VisionRequest) -> VisionResponse:
        """
        Direct generation method - submit job and wait for completion [CA]
        
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
                        user_message="Generation tracking failed. Please try again."
                    )
                
                if status.get("is_terminal", False):
                    if status.get("state") == "completed":
                        # Get final result
                        result = await self.get_job_result(job_id)
                        if result:
                            return result
                        else:
                            raise VisionError(
                                error_type=VisionErrorType.PROVIDER_ERROR,
                                message="Job completed but no result available",
                                user_message="Generation completed but result could not be retrieved."
                            )
                    else:
                        # Job failed
                        error_msg = status.get("progress_message", "Generation failed")
                        raise VisionError(
                            error_type=VisionErrorType.PROVIDER_ERROR,
                            message=f"Generation failed: {error_msg}",
                            user_message="Image generation failed. Please try again."
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
                user_message="Image generation is taking too long. Please try again."
            )
            
        except VisionError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in generate(): {e}")
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"Unexpected error: {e}",
                user_message="An unexpected error occurred during generation."
            )
        finally:
            # Clean up in finally block - safe to reference job_id here
            if reservation is not None:
                # Release reservation if it existed (future budget integration)
                pass
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current job status through unified adapter [PA]
        
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
            
            # Update last poll time
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
                "is_terminal": status.status in [UnifiedStatus.COMPLETED, UnifiedStatus.FAILED, UnifiedStatus.CANCELLED]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get status for job {job_id}: {e}")
            return {
                "job_id": job_id,
                "state": "failed", 
                "progress_percentage": 0,
                "progress_message": f"Status check failed: {e}",
                "is_terminal": True
            }
    
    async def get_job_result(self, job_id: str) -> Optional[VisionResponse]:
        """
        Get final job result through unified adapter [CA]
        
        Args:
            job_id: Job identifier
            
        Returns:
            VisionResponse with generated content or None if not ready
        """
        if job_id not in self.active_jobs:
            return None
        
        try:
            # Check if job is complete first
            status = await self.adapter.poll(job_id)
            if status.status != UnifiedStatus.COMPLETED:
                return None
            
            # Fetch result through unified adapter
            result = await self.adapter.fetch_result(job_id)
            job_meta = self.active_jobs[job_id]
            
            # Download and save assets locally with retries [REH][RM]
            assets_urls: List[str] = result.assets or []
            saved_artifacts: List[Path] = []
            warnings: List[str] = []
            total_size = 0
            artifacts_dir = Path(self.config.get("VISION_ARTIFACTS_DIR", "vision_artifacts")) / job_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                @with_retry(API_RETRY_CONFIG)
                async def _download(url: str, tmp_path: Path, final_path: Path) -> Path:
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            resp.raise_for_status()
                            
                            # Try to detect proper file extension from content-type
                            content_type = resp.headers.get('content-type', '').lower()
                            if content_type and not final_path.suffix:
                                if 'png' in content_type:
                                    final_path = final_path.with_suffix('.png')
                                elif 'jpeg' in content_type or 'jpg' in content_type:
                                    final_path = final_path.with_suffix('.jpg')
                                elif 'webp' in content_type:
                                    final_path = final_path.with_suffix('.webp')
                                elif 'gif' in content_type:
                                    final_path = final_path.with_suffix('.gif')
                                tmp_path = tmp_path.with_suffix(final_path.suffix)
                            
                            with open(tmp_path, "wb") as f:
                                async for chunk in resp.content.iter_chunked(8192):
                                    f.write(chunk)
                            os.replace(tmp_path, final_path)
                            return final_path
                    except Exception as e:
                        # Normalize to APIError to trigger retries consistently
                        raise APIError(str(e))

                def _detect_image_type_from_bytes(data: bytes) -> str:
                    """Detect image MIME type from byte signature."""
                    if data.startswith(b'\x89PNG\r\n\x1a\n'):
                        return 'image/png'
                    elif data.startswith(b'\xff\xd8\xff'):
                        return 'image/jpeg'
                    elif data.startswith(b'RIFF') and b'WEBP' in data[:12]:
                        return 'image/webp'
                    elif data.startswith((b'GIF87a', b'GIF89a')):
                        return 'image/gif'
                    return 'image/png'  # default fallback

                def _get_extension_from_mime(mime_type: str) -> str:
                    """Map MIME type to file extension."""
                    mime_map = {
                        'image/png': '.png',
                        'image/jpeg': '.jpg',
                        'image/webp': '.webp', 
                        'image/gif': '.gif'
                    }
                    return mime_map.get(mime_type, '.png')

                for idx, url in enumerate(assets_urls):
                    try:
                        parsed = urlparse(url)
                        base_name = unquote(os.path.basename(parsed.path)) or f"generated_{job_id}_{idx}"
                        # Remove any existing extension for clean detection
                        if "." in base_name:
                            base_name = base_name.rsplit(".", 1)[0]
                        
                        tmp_path = artifacts_dir / f".{base_name}.part"
                        final_path = artifacts_dir / f"{base_name}.tmp"  # temp name for detection
                        # Download to temp file first
                        saved = await _download(url, tmp_path, final_path)
                        if saved and saved.exists():
                            # Read first few bytes to detect image type
                            with open(saved, 'rb') as f:
                                header_bytes = f.read(32)
                            
                            # Detect MIME type and get proper extension
                            detected_mime = _detect_image_type_from_bytes(header_bytes)
                            proper_extension = _get_extension_from_mime(detected_mime)
                            
                            # Rename file with proper extension
                            proper_final_path = artifacts_dir / f"{base_name}{proper_extension}"
                            if saved != proper_final_path:
                                os.replace(saved, proper_final_path)
                                saved = proper_final_path
                            
                            file_size = saved.stat().st_size
                            saved_artifacts.append(saved)
                            total_size += file_size
                            self.logger.info(f"Artifact saved with MIME detection for job {job_id}: {saved} ({file_size} bytes, {detected_mime})")
                        else:
                            self.logger.warning(f"Failed to download artifact {idx} for job {job_id}: {url}")
                    except Exception as e:
                        warn_msg = f"Asset download failed: {e}"
                        warnings.append(warn_msg)
                        self.logger.warning(f"{warn_msg} (job_id={job_id}, url={url})")

            # Build VisionResponse with local paths
            response = VisionResponse(
                success=True,
                job_id=job_id,
                provider=VisionProvider(result.provider_used.lower()) if result.provider_used else VisionProvider.NOVITA,
                model_used=result.metadata.get('model', 'unknown'),
                artifacts=saved_artifacts,
                processing_time_seconds=asyncio.get_event_loop().time() - job_meta["start_time"],
                actual_cost=self._calculate_actual_cost(job_meta, result),
                file_size_bytes=total_size,
                warnings=warnings,
            )
            
            # Clean up completed job
            del self.active_jobs[job_id]
            
            self.logger.info(
                f"Job {job_id} completed successfully; assets_saved={len(saved_artifacts)}/" 
                f"{len(assets_urls)} dir={artifacts_dir}"
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to get result for job {job_id}: {e}")
            # Clean up failed job
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel running job [REH]
        
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
                del self.active_jobs[job_id]
                self.logger.info(f"Job {job_id} cancelled")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def _map_status_to_state(self, status: UnifiedStatus) -> str:
        """Map unified status to gateway state format [CMV]"""
        mapping = {
            UnifiedStatus.QUEUED: "queued",
            UnifiedStatus.RUNNING: "processing", 
            UnifiedStatus.UPSCALING: "processing",
            UnifiedStatus.SAFETY_REVIEW: "processing",
            UnifiedStatus.UPLOADING: "finalizing",
            UnifiedStatus.COMPLETED: "completed",
            UnifiedStatus.FAILED: "failed",
            UnifiedStatus.CANCELLED: "cancelled"
        }
        return mapping.get(status, "unknown")
    
    def get_supported_tasks(self) -> List[VisionTask]:
        """Get list of supported tasks from unified adapter"""
        return self.adapter.get_supported_tasks()
    
    def get_providers_for_task(self, task: VisionTask) -> List[VisionProvider]:
        """Get available providers for specific task from unified adapter"""
        return self.adapter.get_providers_for_task(task)
    
    def get_models_for_task(self, task: VisionTask, provider: Optional[VisionProvider] = None) -> List[str]:
        """Get available models for task from unified adapter"""
        return self.adapter.get_models_for_task(task, provider)
