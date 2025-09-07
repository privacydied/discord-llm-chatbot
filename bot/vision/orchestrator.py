"""
Vision Job Orchestrator - Async job management with JSON persistence

Handles the complete lifecycle of vision generation jobs:
- Job submission and validation
- Provider selection and execution 
- Progress tracking and status updates
- Error handling and retry logic
- JSON-based persistence (no database required)
- Discord integration and user feedback

Follows Clean Architecture (CA) and Robust Error Handling (REH) principles.
"""

from __future__ import annotations
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil

from bot.util.logging import get_logger
from bot.config import load_config
from .types import (
    VisionRequest, VisionResponse, VisionJob, VisionJobState, 
    VisionError, VisionErrorType, VisionProvider
)
from .gateway import VisionGateway
from .job_store import VisionJobStore
from .safety_filter import VisionSafetyFilter
from .budget_manager_v2 import VisionBudgetManager
from .money import Money
from .pricing_loader import get_pricing_table

logger = get_logger(__name__)


class VisionOrchestrator:
    """
    Async orchestrator for vision generation jobs
    
    Manages the complete job lifecycle from submission to completion:
    - Validates requests against safety and budget policies
    - Queues jobs with concurrency limits
    - Executes generation via Vision Gateway
    - Tracks progress and persists state to JSON
    - Handles Discord notifications and file delivery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.logger = get_logger("vision.orchestrator")
        
        # Lifecycle management [CA][REH]
        self.ready = False
        self._started = False
        self.reason: str = "unstarted"
        
        # Initialize core components
        self.gateway = VisionGateway(self.config)
        self.job_store = VisionJobStore(self.config)
        self.safety_filter = VisionSafetyFilter(self.config)
        self.budget_manager = VisionBudgetManager(self.config)
        
        # Concurrency control
        self.max_concurrent_jobs = self.config["VISION_MAX_CONCURRENT_JOBS"]
        self.max_user_concurrent_jobs = self.config["VISION_MAX_USER_CONCURRENT_JOBS"]
        
        # Active job tracking
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.user_job_counts: Dict[str, int] = {}
        
        # Background task for cleanup and monitoring
        self._cleanup_task: Optional[asyncio.Task] = None
        self._background_tasks_started = False
        
        self.logger.info(f"Vision Orchestrator initialized - max_concurrent: {self.max_concurrent_jobs}, max_per_user: {self.max_user_concurrent_jobs}")
    
    async def start(self) -> None:
        """Start the orchestrator and verify providers are available [CA][REH]"""
        if self._started:
            return
        
        try:
            # Initialize gateway and verify providers
            await self.gateway.startup()
            
            # Check if at least one T2I provider is available via Unified adapter
            available_providers = []
            adapter = getattr(self.gateway, 'adapter', None)
            try:
                from .types import VisionTask  # local import to avoid cycles
            except Exception:
                VisionTask = None
            if adapter and getattr(adapter, 'providers', None):
                for provider_name, plugin in adapter.providers.items():
                    try:
                        caps = plugin.capabilities()
                        modes = caps.get('modes', []) if isinstance(caps, dict) else []
                        if VisionTask and hasattr(VisionTask, 'TEXT_TO_IMAGE') and (VisionTask.TEXT_TO_IMAGE in modes):
                            available_providers.append(provider_name)
                    except Exception:
                        continue
            
            # Determine readiness reason
            if not available_providers:
                # If API key missing, surface clearer reason
                api_key = self.config.get("VISION_API_KEY")
                if not api_key:
                    self.reason = "creds_missing"
                else:
                    self.reason = "no_providers"
                self.logger.error("VisionOrchestrator: no T2I providers configured (VISION_API_KEY missing)" if not api_key else "VisionOrchestrator: no T2I providers configured")
                self.ready = False
            else:
                self.reason = "ok"
                self.logger.info(f"VisionOrchestrator: providers=[{', '.join(available_providers)}] ready")
                self.ready = True
                
            # Start background tasks
            self._start_background_tasks()
            self._started = True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start Vision Orchestrator: {e}", exc_info=True)
            self.ready = False
            self.reason = "init_error"
            raise
    
    async def ensure_started(self) -> None:
        """Lazy start helper - start if not already started [CA]"""
        if not self._started:
            await self.start()
    
    async def submit_job(self, request: VisionRequest) -> VisionJob:
        """
        Submit vision generation job for async processing
        
        Args:
            request: Validated vision generation request
            
        Returns:
            VisionJob with initial state and job ID
            
        Raises:
            VisionError: On validation, safety, or quota failures
        """
        self.logger.info(f"Submitting vision job - task: {request.task.value}, user_id: {request.user_id}, estimated_cost: {request.estimated_cost}")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        try:
            # Safety validation (content filtering)
            safety_result = await self.safety_filter.validate_request(request)
            if not safety_result.approved:
                raise VisionError(
                    error_type=VisionErrorType.CONTENT_FILTERED,
                    message=f"Content safety violation: {safety_result.reason}",
                    user_message=safety_result.user_message
                )
            
            # Budget validation (quota and spend limits)
            budget_result = await self.budget_manager.check_budget(request)
            if not budget_result.approved:
                raise VisionError(
                    error_type=VisionErrorType.QUOTA_EXCEEDED,
                    message=f"Budget exceeded: {budget_result.reason}",
                    user_message=budget_result.user_message
                )
            
            # Check concurrency limits
            await self._check_concurrency_limits(request.user_id)
            
            # Create job record
            job = VisionJob(
                job_id=job_id,
                request=request,
                state=VisionJobState.CREATED,
                discord_interaction_id=getattr(request, 'discord_interaction_id', None)
            )
            
            # Estimate cost and reserve budget (Money-typed) [REH][CMV]
            estimated_cost = await self._estimate_job_cost(request)
            estimated_cost = self._ensure_money(estimated_cost)
            # Persist on request and job for downstream usage
            request.estimated_cost = estimated_cost
            job.request.estimated_cost = estimated_cost
            
            # Reserve budget (will be adjusted when job completes)
            await self.budget_manager.reserve_budget(request.user_id, estimated_cost)
            
            # Persist initial job state
            await self.job_store.save_job(job)
            
            # Queue job for execution
            job.transition_to(VisionJobState.QUEUED, "Job queued for execution")
            await self.job_store.save_job(job)
            
            # Start async execution
            task = asyncio.create_task(self._execute_job(job))
            self.active_jobs[job_id] = task
            
            # Update user concurrency tracking
            self.user_job_counts[request.user_id] = self.user_job_counts.get(request.user_id, 0) + 1
            
            self.logger.info(f"Vision job queued successfully - job_id: {job_id[:8]}, user_id: {request.user_id}, estimated_cost: {estimated_cost}")
            
            return job
            
        except VisionError as e:
            # Log and re-raise vision errors
            self.logger.warning(f"Vision job submission failed - error_type: {e.error_type.value}, message: {e.message}, user_message: {e.user_message}")
            raise e
            
        except Exception as e:
            # Wrap unexpected errors
            self.logger.error(f"Unexpected error submitting vision job: {str(e)}", exc_info=True)
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"Unexpected error: {str(e)}",
                user_message="An internal error occurred. Please try again."
            )
    
    async def get_job_status(self, job_id: str) -> Optional[VisionJob]:
        """Get current status of job"""
        return await self.job_store.load_job(job_id)
    
    async def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel running job if owned by user"""
        job = await self.job_store.load_job(job_id)
        if not job or job.request.user_id != user_id:
            return False
        
        if job.is_terminal_state():
            return True  # Already finished
        
        # Cancel active task
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
        
        # Update job state
        job.transition_to(VisionJobState.CANCELLED, "Cancelled by user")
        await self.job_store.save_job(job)
        
        # Try to cancel with provider
        if job.provider_assigned and job.response and job.response.provider_job_id:
            try:
                provider = self.gateway.providers.get(job.provider_assigned)
                if provider:
                    await provider.cancel_job(job.response.provider_job_id)
            except Exception as e:
                self.logger.debug(f"Provider cancellation failed: {e}")
        
        self.logger.info(f"Job {job_id[:8]} cancelled by user {user_id}")
        return True
    
    async def _check_concurrency_limits(self, user_id: str) -> None:
        """Check if user can start new job within concurrency limits [CMV]"""
        # Check global limit
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            raise VisionError(
                error_type=VisionErrorType.QUOTA_EXCEEDED,
                message="Maximum concurrent jobs reached",
                user_message="The system is currently busy. Please try again in a few minutes."
            )
        
        # Check per-user limit
        user_active_count = self.user_job_counts.get(user_id, 0)
        if user_active_count >= self.max_user_concurrent_jobs:
            raise VisionError(
                error_type=VisionErrorType.QUOTA_EXCEEDED,
                message="User concurrent job limit reached",
                user_message=f"You can only run {self.max_user_concurrent_jobs} job(s) at a time. Please wait for your current job to complete."
            )
    
    async def _estimate_job_cost(self, request: VisionRequest) -> Money:
        """Estimate job cost using PricingTable (Money-typed) with safe fallback [CMV][REH]"""
        try:
            table = get_pricing_table()
            provider = request.preferred_provider or VisionProvider.TOGETHER
            # Width/height/batch/duration already normalized on request
            cost = table.estimate_cost(
                provider=provider,
                task=request.task,
                width=request.width,
                height=request.height,
                num_images=getattr(request, 'batch_size', 1) or 1,
                duration_seconds=getattr(request, 'duration_seconds', 4.0) or 4.0,
                model=request.preferred_model or request.model
            )
            return self._ensure_money(cost)
        except Exception as e:
            # Default tiny estimate when pricing is unknown/unavailable
            self.logger.warning(f"Cost estimation failed (fallback to minimum): {e}")
            return Money("0.006")

    # --- Helpers -----------------------------------------------------------------
    def _ensure_money(self, x: Any) -> Money:
        """Guardrail: ensure Money instance for budgeting [REH]
        - If x is Money, return as-is
        - If x is None, return default minimum estimate ($0.02)
        - Otherwise, wrap via Money(x)
        """
        try:
            if isinstance(x, Money):
                return x
            if x is None:
                return Money("0.006")
            return Money(x)
        except Exception:
            return Money("0.006")
    
    async def _execute_job(self, job: VisionJob) -> None:
        """Execute vision generation job asynchronously [CA][REH]"""
        job_id = job.job_id
        
        try:
            self.logger.debug(f"Starting job execution: {job_id[:8]}")
            
            # Ensure gateway is initialized before first use [RM]
            if not hasattr(self.gateway, '_startup_complete'):
                await self.gateway.startup()
                self.gateway._startup_complete = True
            
            # Transition to running state
            job.transition_to(VisionJobState.RUNNING, "Starting generation")
            job.provider_assigned = job.request.preferred_provider  # May be overridden by gateway
            await self.job_store.save_job(job)
            
            # Execute generation via gateway
            job.update_progress(10, "Contacting provider")
            await self.job_store.save_job(job)
            
            response = await self.gateway.generate(job.request)
            
            if response.success:
                # Success path
                job.response = response
                
                # CRITICAL FIX: Update job ID if provider returned different ID
                if response.job_id and response.job_id != job_id:
                    self.logger.info(f"🔄 Updating job ID mapping - original: {job_id[:8]}, provider: {response.job_id}")
                    # Update job store with provider job ID for Router to find
                    job.provider_job_id = response.job_id
                
                job.transition_to(VisionJobState.COMPLETED, "Generation completed successfully")
                job.update_progress(100, "Complete")
                
                # Update budget with actual cost
                await self.budget_manager.record_actual_cost(
                    job.request.user_id, 
                    self._ensure_money(job.request.estimated_cost),  # Release reserved amount
                    self._ensure_money(response.actual_cost)  # Record actual cost
                )
                
                self.logger.info(f"Job completed successfully - job_id: {job_id[:8]}, provider: {response.provider.value}, actual_cost: {response.actual_cost}, processing_time: {response.processing_time_seconds}s")
            else:
                # Provider returned failure
                job.error = response.error
                job.response = response
                job.transition_to(VisionJobState.FAILED, f"Generation failed: {response.error.message if response.error else 'Unknown error'}")
                
                # Release reserved budget on failure
                await self.budget_manager.release_reservation(job.request.user_id, self._ensure_money(job.request.estimated_cost))
                
                self.logger.error(f"Job failed at provider - job_id: {job_id[:8]}, error: {response.error.message if response.error else 'Unknown error'}")
        
        except asyncio.CancelledError:
            # Job was cancelled
            job.transition_to(VisionJobState.CANCELLED, "Job cancelled")
            await self.budget_manager.release_reservation(
                job.request.user_id,
                self._ensure_money(job.request.estimated_cost)
            )
            self.logger.info(f"Job cancelled: {job_id[:8]}")
            
        except VisionError as e:
            # Handle vision-specific errors
            job.error = e
            job.transition_to(VisionJobState.FAILED, f"Error: {e.message}")
            
            await self.budget_manager.release_reservation(
                job.request.user_id,
                self._ensure_money(job.request.estimated_cost)
            )
            
            self.logger.error(f"Job failed with VisionError - job_id: {job_id[:8]}, error_type: {e.error_type.value}, message: {e.message}")
            
        except Exception as e:
            # Handle unexpected errors
            error = VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"Unexpected error: {str(e)}",
                user_message="An internal error occurred during generation."
            )
            job.error = error
            job.transition_to(VisionJobState.FAILED, f"Unexpected error: {str(e)}")
            
            await self.budget_manager.release_reservation(
                job.request.user_id,
                self._ensure_money(job.request.estimated_cost)
            )
            
            self.logger.error(f"Job failed with unexpected error - job_id: {job_id[:8]}, error: {str(e)}", exc_info=True)
        
        finally:
            # Always persist final state and cleanup
            await self.job_store.save_job(job)
            
            # Remove from active tracking
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # Decrement user job count
            user_id = job.request.user_id
            if user_id in self.user_job_counts:
                self.user_job_counts[user_id] = max(0, self.user_job_counts[user_id] - 1)
                if self.user_job_counts[user_id] == 0:
                    del self.user_job_counts[user_id]
    
    def _start_background_tasks(self) -> None:
        """Start background monitoring and cleanup tasks [PA]"""
        try:
            # Only start if we're in an async context with a running event loop
            loop = asyncio.get_running_loop()
            if not self._background_tasks_started:
                self._cleanup_task = asyncio.create_task(self._background_cleanup())
                self._background_tasks_started = True
        except RuntimeError:
            # No running event loop - tasks will be started lazily when needed
            pass
    
    async def _background_cleanup(self) -> None:
        """Background task for cleanup and monitoring [RM]"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup expired jobs
                await self._cleanup_expired_jobs()
                
                # Cleanup old artifacts 
                await self._cleanup_old_artifacts()
                
                # Log system health (metadata via 'extra' to avoid kwargs to logger) [REH]
                self.logger.debug(
                    "Orchestrator health check",
                    extra={
                        "active_jobs": len(self.active_jobs),
                        "user_counts": len(self.user_job_counts),
                    }
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")
    
    async def _cleanup_expired_jobs(self) -> None:
        """Mark expired jobs as failed and cleanup [RM]"""
        timeout_seconds = self.config["VISION_JOB_TIMEOUT_SECONDS"]
        
        # Load all active jobs
        active_job_ids = list(self.active_jobs.keys())
        
        for job_id in active_job_ids:
            try:
                job = await self.job_store.load_job(job_id)
                if job and not job.is_terminal_state() and job.is_expired(timeout_seconds):
                    # Cancel expired job
                    if job_id in self.active_jobs:
                        self.active_jobs[job_id].cancel()
                    
                    # Mark as expired
                    job.transition_to(VisionJobState.EXPIRED, f"Job expired after {timeout_seconds}s")
                    await self.job_store.save_job(job)
                    
                    # Release budget reservation
                    await self.budget_manager.release_reservation(
                        job.request.user_id,
                        self._ensure_money(job.request.estimated_cost)
                    )
                    
                    self.logger.warning(f"Job expired: {job_id[:8]}")
                    
            except Exception as e:
                self.logger.error(f"Error cleaning up job {job_id[:8]}: {e}")
    
    async def _cleanup_old_artifacts(self) -> None:
        """Remove old artifact files based on TTL [RM]"""
        artifacts_dir = Path(self.config["VISION_ARTIFACTS_DIR"])
        if not artifacts_dir.exists():
            return
        
        ttl_days = self.config["VISION_ARTIFACT_TTL_DAYS"]
        cutoff_time = datetime.now(timezone.utc).timestamp() - (ttl_days * 24 * 3600)
        
        cleaned_count = 0
        cleaned_size = 0
        
        try:
            for file_path in artifacts_dir.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    cleaned_count += 1
                    cleaned_size += file_size
            
            if cleaned_count > 0:
                # Include metrics via 'extra' to avoid unexpected kwargs to logger [REH]
                self.logger.info(
                    "Cleaned up old artifacts",
                    extra={
                        "count": cleaned_count,
                        "size_mb": round(cleaned_size / (1024*1024), 1),
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Artifact cleanup error: {e}")
    
    async def close(self) -> None:
        """Shutdown orchestrator and cleanup resources [RM]"""
        self.logger.info("Shutting down Vision Orchestrator")
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active jobs
        for job_id, task in self.active_jobs.items():
            task.cancel()
            self.logger.debug(f"Cancelled job during shutdown: {job_id[:8]}")
        
        # Wait for tasks to complete with timeout
        if self.active_jobs:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_jobs.values(), return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some jobs did not complete during shutdown")
        
        # Close gateway
        await self.gateway.shutdown()
        
        self.logger.info("Vision Orchestrator shutdown complete")
