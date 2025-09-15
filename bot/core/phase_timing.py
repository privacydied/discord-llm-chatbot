"""
Phase Timing Utilities - Track pipeline phases with correlation IDs and metrics.
Implements logging workflow with Rich Pretty Console + JSONL dual sinks.
"""

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, AsyncGenerator, List
from rich.panel import Panel
from rich.tree import Tree

from .phase_constants import PhaseConstants as PC
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PhaseMetrics:
    """Individual phase timing metrics."""

    phase: str
    start_ts: float
    end_ts: Optional[float] = None
    duration_ms: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, success: bool = True, error: Optional[str] = None, **metadata):
        """Mark phase as completed with timing data."""
        self.end_ts = time.time()
        self.duration_ms = int((self.end_ts - self.start_ts) * 1000)
        self.success = success
        self.error = error
        self.metadata.update(metadata)


@dataclass
class PipelineTracker:
    """Tracks complete pipeline execution with correlation ID."""

    corr_id: str
    msg_id: str
    user_id: str
    guild_id: Optional[str]
    is_dm: bool
    start_ts: float = field(default_factory=time.time)
    phases: Dict[str, PhaseMetrics] = field(default_factory=dict)
    total_duration_ms: Optional[int] = None

    def start_phase(self, phase: str, **metadata) -> PhaseMetrics:
        """Start tracking a new phase."""
        if phase in self.phases:
            logger.warning(
                f"⚠️ Phase {phase} already started for corr_id {self.corr_id}"
            )

        phase_metric = PhaseMetrics(
            phase=phase, start_ts=time.time(), metadata=metadata
        )
        self.phases[phase] = phase_metric

        # Log phase start with Rich formatting [CA]
        self._log_phase_event("start", phase, phase_metric, **metadata)
        return phase_metric

    def complete_phase(
        self, phase: str, success: bool = True, error: Optional[str] = None, **metadata
    ):
        """Complete a phase and log results."""
        if phase not in self.phases:
            logger.error(f"❌ Phase {phase} not found for corr_id {self.corr_id}")
            return

        phase_metric = self.phases[phase]
        phase_metric.complete(success=success, error=error, **metadata)

        # Log completion with SLO check [REH]
        self._log_phase_event("complete", phase, phase_metric, **metadata)
        self._check_slo_breach(phase, phase_metric)

    def complete_pipeline(self):
        """Mark entire pipeline as complete and generate summary."""
        self.total_duration_ms = int((time.time() - self.start_ts) * 1000)

        # Generate final pipeline summary [PA]
        self._log_pipeline_summary()

    def _log_phase_event(
        self, event: str, phase: str, phase_metric: PhaseMetrics, **metadata
    ):
        """Log phase event with Rich formatting and JSONL structure."""
        # Determine icon based on event and success [CA]
        if event == "start":
            icon = "🔄"
            level = "INFO"
        elif event == "complete":
            icon = "✔" if phase_metric.success else "✖"
            level = "INFO" if phase_metric.success else "ERROR"
        else:
            icon = "ℹ"
            level = "INFO"

        # Create readable message with duration for completed phases
        if event == "complete" and phase_metric.duration_ms is not None:
            duration_str = f"{phase_metric.duration_ms}ms"
            msg = f"{icon} {phase} completed: {duration_str}"
            if phase_metric.error:
                msg += f" (error: {phase_metric.error})"
        else:
            msg = f"{icon} {phase} {event}"

        # Log with structured data for JSONL [CMV]
        extra_detail = {
            "corr_id": self.corr_id,
            "phase": phase,
            "event": event,
            "duration_ms": phase_metric.duration_ms,
            "success": phase_metric.success,
            "msg_id": self.msg_id,
            "user_id": self.user_id,
            "guild_id": self.guild_id,
            "is_dm": self.is_dm,
            **metadata,
            **phase_metric.metadata,
        }

        # Use proper logging level
        if level == "ERROR":
            logger.error(msg, extra={"detail": extra_detail})
        else:
            logger.info(msg, extra={"detail": extra_detail})

    def _check_slo_breach(self, phase: str, phase_metric: PhaseMetrics):
        """Check if phase breached SLO targets and alert."""
        slo_targets = PC.get_slo_targets()
        target_ms = slo_targets.get(phase)

        if (
            target_ms
            and phase_metric.duration_ms
            and phase_metric.duration_ms > target_ms
        ):
            breach_pct = int((phase_metric.duration_ms / target_ms - 1) * 100)
            logger.warning(
                f"⚠️ SLO BREACH: {phase} took {phase_metric.duration_ms}ms (target: {target_ms}ms, +{breach_pct}%)",
                extra={
                    "detail": {
                        "corr_id": self.corr_id,
                        "phase": phase,
                        "event": "slo_breach",
                        "duration_ms": phase_metric.duration_ms,
                        "target_ms": target_ms,
                        "breach_pct": breach_pct,
                        "msg_id": self.msg_id,
                    }
                },
            )

    def _log_pipeline_summary(self):
        """Generate comprehensive pipeline summary with Rich Tree/Panel."""
        total_target = PC.get_slo_targets()["pipeline_total"]
        breach_status = "BREACH" if self.total_duration_ms > total_target else "OK"

        # Create summary message [CA]
        summary_msg = f"✅ Pipeline completed: {self.total_duration_ms}ms (target: {total_target}ms) - {breach_status}"

        # Detailed breakdown for JSONL
        phase_breakdown = {}
        for phase_name in PC.get_all_phases():
            if phase_name in self.phases:
                metric = self.phases[phase_name]
                phase_breakdown[phase_name] = {
                    "duration_ms": metric.duration_ms,
                    "success": metric.success,
                    "error": metric.error,
                }

        logger.info(
            summary_msg,
            extra={
                "detail": {
                    "corr_id": self.corr_id,
                    "event": "pipeline_complete",
                    "total_duration_ms": self.total_duration_ms,
                    "target_ms": total_target,
                    "breach": breach_status == "BREACH",
                    "phases": phase_breakdown,
                    "msg_id": self.msg_id,
                    "user_id": self.user_id,
                    "guild_id": self.guild_id,
                    "is_dm": self.is_dm,
                }
            },
        )

        # Rich Panel summary for DEBUG builds [PA]
        if logger.level <= 10:  # DEBUG level
            self._create_debug_panel()

    def _create_debug_panel(self):
        """Create Rich Panel with Tree for DEBUG visibility."""
        tree = Tree(f"Pipeline Summary (corr_id: {self.corr_id[:8]})")

        for phase_name in PC.get_all_phases():
            if phase_name in self.phases:
                metric = self.phases[phase_name]
                status_icon = "✅" if metric.success else "❌"
                duration_str = (
                    f"{metric.duration_ms}ms" if metric.duration_ms else "N/A"
                )
                tree.add(f"{status_icon} {phase_name}: {duration_str}")

        panel = Panel(tree, title=f"Pipeline Timing ({self.total_duration_ms}ms total)")

        # Log panel as debug info
        logger.debug("Pipeline Debug Summary", extra={"rich_panel": panel})


class PhaseTimingManager:
    """Global manager for pipeline tracking."""

    def __init__(self):
        self.active_trackers: Dict[str, PipelineTracker] = {}
        self.completed_trackers: List[PipelineTracker] = []
        self.max_history = 100  # Keep last 100 for analysis

    def create_pipeline_tracker(
        self, msg_id: str, user_id: str, guild_id: Optional[str] = None
    ) -> PipelineTracker:
        """Create new pipeline tracker with correlation ID."""
        corr_id = str(uuid.uuid4())[:8]  # Short correlation ID
        is_dm = guild_id is None

        tracker = PipelineTracker(
            corr_id=corr_id,
            msg_id=msg_id,
            user_id=user_id,
            guild_id=guild_id,
            is_dm=is_dm,
        )

        self.active_trackers[corr_id] = tracker
        logger.info(
            f"🔄 === PIPELINE STARTED: {corr_id} MSG {msg_id} ===",
            extra={
                "detail": {
                    "corr_id": corr_id,
                    "msg_id": msg_id,
                    "event": "pipeline_start",
                }
            },
        )

        return tracker

    def complete_tracker(self, tracker: PipelineTracker):
        """Move tracker from active to completed."""
        tracker.complete_pipeline()

        # Move to completed history
        if tracker.corr_id in self.active_trackers:
            del self.active_trackers[tracker.corr_id]

        self.completed_trackers.append(tracker)

        # Trim history if needed
        if len(self.completed_trackers) > self.max_history:
            self.completed_trackers.pop(0)

    @asynccontextmanager
    async def track_phase(
        self, tracker: PipelineTracker, phase: str, **metadata
    ) -> AsyncGenerator[PhaseMetrics, None]:
        """Async context manager for phase timing."""
        phase_metric = tracker.start_phase(phase, **metadata)

        try:
            yield phase_metric
            tracker.complete_phase(phase, success=True, **metadata)
        except Exception as e:
            error_str = str(e)
            tracker.complete_phase(phase, success=False, error=error_str, **metadata)
            raise


# Global instance
_timing_manager = PhaseTimingManager()


def get_timing_manager() -> PhaseTimingManager:
    """Get global timing manager instance."""
    return _timing_manager
