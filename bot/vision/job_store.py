"""Vision Job Store - JSON-based job persistence.

Handles persistence of vision generation jobs using JSON files on disk:
- Job creation, updates, and retrieval
- JSONL append-only progress logs for auditability
- Atomic file operations with backup/recovery
- Job querying and filtering capabilities
- No external database dependencies

Follows Clean Architecture (CA) and Resource Management (RM) principles.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles

from bot.config import load_config
from bot.utils.logging import get_logger

from .money import Money
from .types import VisionError, VisionErrorType, VisionJob, VisionJobState

logger = get_logger(__name__)


class VisionJobStore:
    """JSON-based job persistence with atomic operations and audit logging.

    Features:
    - Atomic file writes with backup/recovery
    - JSONL append-only progress logs
    - Concurrent access protection with file locking
    - Job filtering and querying
    - Automatic cleanup and archival
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.logger = get_logger("vision.job_store")

        # Initialize storage directories
        self.jobs_dir = Path(self.config["VISION_JOBS_DIR"])
        self.ledger_path = Path(self.config["VISION_LEDGER_PATH"])

        # Create directories if needed
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # File locking for concurrent access
        self._locks: dict[str, asyncio.Lock] = {}
        self._ledger_lock = asyncio.Lock()

        self.logger.info(f"Vision Job Store initialized - jobs_dir: {self.jobs_dir}, ledger_path: {self.ledger_path}")

    async def save_job(self, job: VisionJob) -> None:
        """Save job state to JSON file atomically.

        Args:
            job: VisionJob instance to persist

        Raises:
            VisionError: On file I/O errors

        """
        job_id = job.job_id

        try:
            async with self._get_job_lock(job_id):
                # Atomic write with temp file + rename
                job_file = self.jobs_dir / f"{job_id}.json"
                temp_file = self.jobs_dir / f"{job_id}.json.tmp"

                # Serialize job data
                job_data = job.to_dict()
                job_data["last_updated"] = datetime.now(UTC).isoformat()

                # Write to temp file
                async with aiofiles.open(temp_file, "w") as f:
                    await f.write(json.dumps(job_data, indent=2, default=str))
                    # Note: aiofiles doesn't support fsync(), but write is already persistent

                # Atomic rename
                temp_file.rename(job_file)

                # Append progress log entry (JSONL format)
                await self._append_progress_log(job)

                self.logger.debug(f"Job state saved - job_id: {job_id[:8]}, state: {job.state.value}, progress: {getattr(job, 'progress_percentage', 0)}")

        except Exception as e:
            self.logger.exception(f"Failed to save job {job_id[:8]}: {e!s}")
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"Failed to save job: {e!s}",
                user_message="Failed to save job progress. Please try again.",
            ) from e

    async def load_job(self, job_id: str) -> VisionJob | None:
        """Load job from JSON file.

        Args:
            job_id: Job ID to load

        Returns:
            VisionJob instance or None if not found

        Raises:
            VisionError: On file corruption or I/O errors

        """
        try:
            async with self._get_job_lock(job_id):
                job_file = self.jobs_dir / f"{job_id}.json"

                if not job_file.exists():
                    return None

                # Read and deserialize
                async with aiofiles.open(job_file) as f:
                    content = await f.read()
                    job_data = json.loads(content)

                # Reconstruct job from dict
                job = VisionJob.from_dict(job_data)

                # Only log on state changes, not every load
                if not hasattr(self, "_last_loaded_states"):
                    self._last_loaded_states = {}

                current_state = job.state.value
                if job_id not in self._last_loaded_states or self._last_loaded_states[job_id] != current_state:
                    self.logger.debug(f"Job loaded - job_id: {job_id[:8]}, state: {current_state}")
                    self._last_loaded_states[job_id] = current_state

                return job

        except json.JSONDecodeError as e:
            self.logger.exception(f"Job file corrupted - job_id: {job_id[:8]}, error: {e!s}")
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message=f"Job file corrupted: {e!s}",
                user_message="Job data is corrupted. Please contact support.",
            ) from e

        except Exception as e:
            self.logger.exception(f"Failed to load job - job_id: {job_id[:8]}, error: {e!s}")
            return None  # Treat as not found for robustness

    async def list_jobs(
        self,
        user_id: str | None = None,
        state: VisionJobState | None = None,
        limit: int = 100,
    ) -> list[VisionJob]:
        """List jobs with optional filtering.

        Args:
            user_id: Filter by user ID
            state: Filter by job state
            limit: Maximum jobs to return

        Returns:
            List of VisionJob instances

        """
        jobs = []

        try:
            # Iterate through job files
            job_files = list(self.jobs_dir.glob("*.json"))
            job_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # Newest first

            for job_file in job_files[: limit * 2]:  # Load extra in case of filtering
                if len(jobs) >= limit:
                    break

                try:
                    job_id = job_file.stem
                    job = await self.load_job(job_id)

                    if job is None:
                        continue

                    # Apply filters
                    if user_id and job.request.user_id != user_id:
                        continue
                    if state and job.state != state:
                        continue

                    jobs.append(job)

                except Exception as e:
                    self.logger.warning(f"Failed to load job {job_file.stem}: {e}")
                    continue

            self.logger.debug(f"Jobs listed - count: {len(jobs)}, user_filter: {user_id is not None}, state_filter: {state.value if state else None}")

            return jobs

        except Exception as e:
            self.logger.exception(f"Failed to list jobs: {e}")
            return []

    async def delete_job(self, job_id: str) -> bool:
        """Delete job and its progress log.

        Args:
            job_id: Job ID to delete

        Returns:
            True if deleted, False if not found

        """
        try:
            async with self._get_job_lock(job_id):
                job_file = self.jobs_dir / f"{job_id}.json"

                if not job_file.exists():
                    return False

                # Remove job file
                job_file.unlink()

                # Append deletion record to ledger
                await self._append_ledger_entry(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "event": "job_deleted",
                        "job_id": job_id,
                    },
                )

                self.logger.info(f"Job deleted: {job_id[:8]}")
                return True

        except Exception as e:
            self.logger.exception(f"Failed to delete job {job_id[:8]}: {e}")
            return False

    async def get_user_active_count(self, user_id: str) -> int:
        """Get count of active jobs for user [CMV]."""
        try:
            active_states = {
                VisionJobState.CREATED,
                VisionJobState.QUEUED,
                VisionJobState.RUNNING,
            }
            active_jobs = []

            for state in active_states:
                jobs = await self.list_jobs(user_id=user_id, state=state, limit=50)
                active_jobs.extend(jobs)

            return len(active_jobs)

        except Exception as e:
            self.logger.exception(f"Failed to count user active jobs: {e}")
            return 0

    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """Archive/delete jobs older than specified days.

        Args:
            days: Age threshold for cleanup

        Returns:
            Number of jobs cleaned up

        """
        cleaned_count = 0
        cutoff_time = datetime.now(UTC).timestamp() - (days * 24 * 3600)

        try:
            job_files = list(self.jobs_dir.glob("*.json"))

            for job_file in job_files:
                if job_file.stat().st_mtime < cutoff_time:
                    try:
                        # Load job to get details for audit
                        job_id = job_file.stem
                        job = await self.load_job(job_id)

                        # Archive to ledger before deletion
                        if job:
                            await self._append_ledger_entry(
                                {
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "event": "job_archived",
                                    "job_id": job_id,
                                    "final_state": job.state.value,
                                    "user_id": job.request.user_id,
                                    "created_at": job.created_at.isoformat(),
                                },
                            )

                        # Delete job file
                        job_file.unlink()
                        cleaned_count += 1

                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup job {job_file.name}: {e}")
                        continue

            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old jobs")

            return cleaned_count

        except Exception as e:
            self.logger.exception(f"Job cleanup error: {e}")
            return 0

    async def _append_progress_log(self, job: VisionJob) -> None:
        """Append job progress to JSONL audit log [CMV]."""
        try:
            log_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "job_id": job.job_id,
                "state": job.state.value,
                "progress_percentage": job.progress_percentage,
                "progress_message": getattr(job, "progress_message", ""),
                "user_id": job.request.user_id,
                "task": job.request.task.value,
            }

            if job.error:
                log_entry["error_type"] = job.error.error_type.value
                log_entry["error_message"] = job.error.message

            if job.response:
                log_entry["provider"] = job.response.provider.value
                log_entry["success"] = job.response.success
                # Persist Money as JSON-safe string; accept legacy numerics [REH]
                try:
                    ac = job.response.actual_cost
                    if isinstance(ac, Money):
                        log_entry["actual_cost"] = ac.to_json_value()
                    elif ac is not None:
                        log_entry["actual_cost"] = Money(ac).to_json_value()
                except Exception as exc:
                    # Skip actual_cost if it cannot be normalized
                    logger.debug(f"actual_cost normalization failed: {exc}")

            await self._append_ledger_entry(log_entry)

        except Exception as e:
            # Log errors but don't fail the main operation
            self.logger.debug(f"Progress log append failed: {e}")

    async def _append_ledger_entry(self, entry: dict[str, Any]) -> None:
        """Append entry to JSONL ledger file [CMV]."""
        try:
            # Use async lock instead of blocking fcntl.flock to prevent deadlocks
            async with self._ledger_lock, aiofiles.open(self.ledger_path, "a") as f:
                line = json.dumps(entry, ensure_ascii=False) + "\n"
                await f.write(line)
                await f.flush()

                # Force sync to disk for data integrity
                if hasattr(f, "fileno"):
                    try:
                        fd = f.fileno()  # type: ignore[attr-defined]
                        os.fsync(fd)
                    except Exception as exc:
                        logger.debug(f"fsync failed: {exc}")
        except Exception as e:
            self.logger.debug(f"Ledger append failed: {e}")

    @asynccontextmanager
    async def _get_job_lock(self, job_id: str):
        """Async context manager for per-job lock [RM]."""
        lock = self._locks.get(job_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[job_id] = lock
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()

    async def get_job_stats(self) -> dict[str, Any]:
        """Get statistics about jobs in store [CMV]."""
        try:
            job_files = list(self.jobs_dir.glob("*.json"))
            total_jobs = len(job_files)

            # Count by state (sample recent jobs for performance)
            state_counts = {}
            recent_files = sorted(job_files, key=lambda p: p.stat().st_mtime, reverse=True)[:200]

            for job_file in recent_files:
                try:
                    async with aiofiles.open(job_file) as f:
                        job_data = json.loads(await f.read())
                        state = job_data.get("state", "unknown")
                        state_counts[state] = state_counts.get(state, 0) + 1
                except Exception as exc:
                    logger.debug(f"job file read failed: {exc}")
                    continue

            return {
                "total_jobs": total_jobs,
                "state_counts": state_counts,
                "storage_path": str(self.jobs_dir),
                "ledger_size": self.ledger_path.stat().st_size if self.ledger_path.exists() else 0,
            }

        except Exception as e:
            self.logger.exception(f"Failed to get job stats: {e}")
            return {"error": str(e)}
