"""
Vision Job Watcher - Single-flight watcher registry with proper terminal exit

Implements the fixes from the "stop debug spam" prompt:
- Per-job watcher registry to prevent duplicate watch tasks
- Deterministic terminal exit with no loops after completion
- Finalize-once guard for idempotent cleanup
- Proper poll cadence with jitter and backoff
- State-change logging instead of per-tick spam
"""

import asyncio
import random
import time
from typing import Dict, Optional, Set
from bot.util.logging import get_logger

logger = get_logger(__name__)


class JobWatcherRegistry:
    """Single-flight watcher registry to prevent duplicate polling loops"""
    
    def __init__(self):
        self._watchers: Dict[str, asyncio.Task] = {}
        self._finalized_jobs: Set[str] = set()
        self._last_states: Dict[str, str] = {}
        self._poll_counts: Dict[str, int] = {}
        
    async def watch_job(self, job_id: str, orchestrator, progress_msg=None, original_msg=None, timeout_seconds: int = 600):
        """
        Watch a job with single-flight guarantee
        
        Args:
            job_id: Unique job identifier
            orchestrator: Vision orchestrator instance
            progress_msg: Optional Discord message to update
            original_msg: Optional original Discord message
            timeout_seconds: Maximum time to wait for completion
            
        Returns:
            Job result when terminal state reached
        """
        # Reuse existing watcher if present and not done
        if job_id in self._watchers:
            existing_task = self._watchers[job_id]
            if not existing_task.done():
                logger.debug(f"Reusing existing watcher for job {job_id[:8]}")
                return await existing_task
            else:
                # Clean up completed watcher
                del self._watchers[job_id]
        
        # Create new watcher task
        watcher_task = asyncio.create_task(
            self._watch_job_impl(job_id, orchestrator, progress_msg, original_msg, timeout_seconds)
        )
        self._watchers[job_id] = watcher_task
        
        try:
            return await watcher_task
        finally:
            # Clean up watcher registry
            if job_id in self._watchers:
                del self._watchers[job_id]
    
    async def _watch_job_impl(self, job_id: str, orchestrator, progress_msg, original_msg, timeout_seconds: int):
        """Internal watcher implementation with proper terminal exit"""
        start_time = time.time()
        poll_interval = 0.5  # Start with 500ms
        max_interval = 5.0   # Cap at 5 seconds
        last_state = None
        poll_count = 0
        
        logger.debug(f"Starting job watcher for {job_id[:8]}")
        
        try:
            while True:
                elapsed_time = time.time() - start_time
                poll_count += 1
                self._poll_counts[job_id] = poll_count
                
                # Check timeout
                if elapsed_time > timeout_seconds:
                    logger.warning(f"Job watcher timeout - job_id: {job_id[:8]}, elapsed: {elapsed_time:.1f}s")
                    return None
                
                # Get current job status ONCE per iteration
                try:
                    updated_job = await orchestrator.get_job_status(job_id)
                except Exception as e:
                    logger.error(f"Failed to get job status for {job_id[:8]}: {e}")
                    await asyncio.sleep(poll_interval)
                    continue
                
                if not updated_job:
                    logger.warning(f"Job not found during monitoring - job_id: {job_id[:8]}")
                    return None
                
                current_state = updated_job.state.value
                
                # Log only on state changes or every 10th poll for heartbeat
                if current_state != last_state:
                    logger.info(f"Job state change - job_id: {job_id[:8]}, {last_state or 'init'} -> {current_state}")
                    last_state = current_state
                    self._last_states[job_id] = current_state
                elif poll_count % 10 == 0:
                    logger.debug(f"Job heartbeat - job_id: {job_id[:8]}, state: {current_state}, poll: {poll_count}")
                
                # Check if job reached terminal state
                if updated_job.is_terminal_state():
                    logger.info(f"Job reached terminal state - job_id: {job_id[:8]}, state: {current_state}, polls: {poll_count}")
                    
                    # Finalize once and exit immediately
                    await self._finalize_job_once(job_id, updated_job, progress_msg, original_msg)
                    return updated_job  # CRITICAL: Exit immediately, no more polling
                
                # Update progress if message provided and state changed
                if progress_msg and current_state != self._last_states.get(job_id):
                    try:
                        await self._update_progress_message(progress_msg, updated_job, original_msg, elapsed_time)
                    except Exception as e:
                        logger.debug(f"Could not update progress message: {e}")
                
                # Apply poll cadence with jitter - NO SLEEP after terminal
                jitter = random.uniform(-0.1, 0.1)  # ±100ms jitter
                sleep_time = min(poll_interval + jitter, max_interval)
                await asyncio.sleep(max(0.1, sleep_time))  # Minimum 100ms
                
                # Gradually increase poll interval (linear backoff)
                poll_interval = min(poll_interval + 0.1, max_interval)
                
        except asyncio.CancelledError:
            logger.debug(f"Job watcher cancelled - job_id: {job_id[:8]}")
            raise
        except Exception as e:
            logger.error(f"Job watcher error - job_id: {job_id[:8]}: {e}", exc_info=True)
            return None
    
    async def _finalize_job_once(self, job_id: str, job, progress_msg, original_msg):
        """Finalize job exactly once with idempotent guard"""
        if job_id in self._finalized_jobs:
            logger.debug(f"Job already finalized - job_id: {job_id[:8]}")
            return
        
        # Mark as finalized immediately to prevent re-entry
        self._finalized_jobs.add(job_id)
        
        try:
            # Perform finalization work
            if progress_msg and hasattr(job, 'state'):
                if job.state.value == "completed" and job.response:
                    logger.info(f"Finalizing successful job - job_id: {job_id[:8]}")
                    # Handle successful completion
                    if original_msg and hasattr(original_msg, 'channel'):
                        from bot.router import Router
                        if hasattr(Router, '_handle_vision_success'):
                            # This would need router instance - simplified for now
                            pass
                else:
                    logger.info(f"Finalizing failed job - job_id: {job_id[:8]}, state: {job.state.value}")
                    # Handle failure
                    pass
            
            logger.debug(f"Job finalization complete - job_id: {job_id[:8]}")
            
        except Exception as e:
            logger.error(f"Error during job finalization - job_id: {job_id[:8]}: {e}", exc_info=True)
    
    async def _update_progress_message(self, progress_msg, job, original_msg, elapsed_time):
        """Update Discord progress message"""
        try:
            # Import here to avoid circular imports
            import discord
            
            if hasattr(progress_msg, 'edit'):
                embed = discord.Embed(
                    title="🎨 Vision Generation Working",
                    color=0xffaa00,
                    description=f"Processing... ({elapsed_time:.0f}s elapsed)"
                )
                embed.add_field(name="Job ID", value=f"`{job.job_id[:8]}`", inline=True)
                embed.add_field(name="Status", value=f"🟡 {job.state.value.title()}", inline=True)
                embed.add_field(name="Progress", value=f"{job.progress_percentage or 0}%", inline=True)
                
                await progress_msg.edit(embed=embed)
        except Exception as e:
            logger.debug(f"Progress message update failed: {e}")
    
    def cancel_all_watchers(self):
        """Cancel all active watchers (for shutdown)"""
        logger.info(f"Cancelling {len(self._watchers)} active job watchers")
        
        for job_id, task in self._watchers.items():
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelled watcher for job {job_id[:8]}")
        
        # Gather all tasks for cleanup
        tasks = [task for task in self._watchers.values() if not task.done()]
        if tasks:
            return asyncio.gather(*tasks, return_exceptions=True)
    
    def get_active_watcher_count(self) -> int:
        """Get count of active watchers for monitoring"""
        active_count = sum(1 for task in self._watchers.values() if not task.done())
        return active_count


# Global registry instance
_global_registry = None

def get_watcher_registry() -> JobWatcherRegistry:
    """Get or create global watcher registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = JobWatcherRegistry()
    return _global_registry
