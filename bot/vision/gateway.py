"""
Vision Gateway - Provider-agnostic facade for image/video generation

Unified gateway using pluggable provider system with automatic failover,
retry logic, and cost estimation following REH and CA principles.
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from bot.util.logging import get_logger
from bot.config import load_config
from .types import (
    VisionRequest, VisionResponse, VisionTask, VisionProvider, 
    VisionError, VisionErrorType
)
from .unified_adapter import UnifiedVisionAdapter, UnifiedStatus, UnifiedResult

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
        try:
            self.logger.info(
                f"Submitting {request.task.value} job for user {request.user_id}"
            )
            
            # Submit through unified adapter
            response = await self.adapter.submit(request)
            
            # Extract job details from VisionResponse
            job_id = response.job_id
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
            self.logger.error(f"Job submission failed: {e}")
            if isinstance(e, VisionError):
                raise
            else:
                raise VisionError(
                    error_type=VisionErrorType.PROVIDER_ERROR,
                    message=f"Submission failed: {e}",
                    user_message="Failed to start vision generation. Please try again."
                )
    
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
            await self.cancel_job(job_id)
            raise VisionError(
                error_type=VisionErrorType.TIMEOUT,
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
            
            # Convert unified result to VisionResponse
            response = VisionResponse(
                success=True,
                job_id=job_id,
                provider=VisionProvider(result.provider_used.lower()) if result.provider_used else VisionProvider.NOVITA,
                model_used=result.metadata.get('model', 'unknown'),
                artifacts=[Path(url) for url in result.assets] if result.assets else [],
                processing_time_seconds=asyncio.get_event_loop().time() - job_meta["start_time"],
                actual_cost=result.final_cost
            )
            
            # Clean up completed job
            del self.active_jobs[job_id]
            
            self.logger.info(f"Job {job_id} completed successfully")
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
