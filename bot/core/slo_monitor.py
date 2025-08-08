"""
SLO Monitoring System - Track p95 performance metrics with alerts and Rich dashboards.
Implements PA (Performance Awareness) and REH (Robust Error Handling) rules.
"""
import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Deque
from enum import Enum
import json

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich.columns import Columns
from rich.align import Align

from .phase_constants import PhaseConstants as PC
from .phase_timing import PipelineTracker, PhaseMetrics
from ..util.logging import get_logger

logger = get_logger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class SLOTarget:
    """SLO target configuration."""
    name: str
    target_ms: int
    measurement_window_minutes: int = 5
    alert_threshold_consecutive: int = PC.ALERT_CONSECUTIVE_WINDOWS
    warning_multiplier: float = 0.8  # Warn at 80% of breach
    critical_multiplier: float = 1.2  # Critical at 120% of breach

@dataclass
class MetricWindow:
    """Time-windowed metric collection."""
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    window_minutes: int = 5
    
    def add_value(self, value: float, timestamp: float = None):
        """Add new metric value with timestamp."""
        if timestamp is None:
            timestamp = time.time()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        # Clean old values outside window
        cutoff_time = timestamp - (self.window_minutes * 60)
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.values.popleft()
            self.timestamps.popleft()
    
    def get_percentile(self, percentile: int) -> Optional[float]:
        """Get percentile value from current window."""
        if not self.values:
            return None
        
        try:
            return statistics.quantiles(list(self.values), n=100)[percentile - 1]
        except (statistics.StatisticsError, IndexError):
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for window."""
        if not self.values:
            return {"count": 0}
        
        values_list = list(self.values)
        return {
            "count": len(values_list),
            "mean": statistics.mean(values_list),
            "median": statistics.median(values_list),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
            "min": min(values_list),
            "max": max(values_list),
            "window_minutes": self.window_minutes
        }

@dataclass
class AlertHistory:
    """Alert history tracking."""
    alert_level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    value: float
    target: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SLOMonitor:
    """Comprehensive SLO monitoring with alerts and dashboards."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        
        # SLO targets from phase constants [CMV]
        slo_targets = PC.get_slo_targets()
        self.targets = {
            PC.PHASE_ROUTER_DISPATCH: SLOTarget("Router Dispatch", slo_targets[PC.PHASE_ROUTER_DISPATCH]),
            PC.PHASE_CONTEXT_GATHER: SLOTarget("Context Gather", slo_targets[PC.PHASE_CONTEXT_GATHER]),
            PC.PHASE_RAG_QUERY: SLOTarget("RAG Query", slo_targets[PC.PHASE_RAG_QUERY]),
            PC.PHASE_PREP_GEN: SLOTarget("Prompt Preparation", slo_targets[PC.PHASE_PREP_GEN]),
            PC.PHASE_LLM_CALL: SLOTarget("LLM API Call", slo_targets[PC.PHASE_LLM_CALL]),
            PC.PHASE_DISCORD_DISPATCH: SLOTarget("Discord Send", slo_targets[PC.PHASE_DISCORD_DISPATCH]),
            "pipeline_total": SLOTarget("Total Pipeline", slo_targets["pipeline_total"])
        }
        
        # Metric windows for each target
        self.metric_windows = {
            name: MetricWindow() for name in self.targets.keys()
        }
        
        # Alert state tracking
        self.consecutive_breaches = {name: 0 for name in self.targets.keys()}
        self.alert_history: List[AlertHistory] = []
        self.alert_callbacks: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [],
            AlertLevel.CRITICAL: []
        }
        
        # Dashboard refresh state
        self.last_dashboard_update = 0
        self.dashboard_refresh_interval = 30  # 30 seconds
        
        # Performance stats
        self.monitoring_stats = {
            "total_measurements": 0,
            "alerts_fired": 0,
            "slo_breaches": 0,
            "dashboard_renders": 0
        }
        
        logger.info("📊 SLOMonitor initialized with Rich dashboard support")
    
    def record_phase_metric(self, phase: str, duration_ms: int, tracker: Optional[PipelineTracker] = None):
        """Record phase performance metric [PA]."""
        if phase not in self.metric_windows:
            logger.debug(f"Unknown phase for SLO monitoring: {phase}")
            return
        
        # Record the measurement
        self.metric_windows[phase].add_value(duration_ms)
        self.monitoring_stats["total_measurements"] += 1
        
        # Check for SLO breach
        self._check_slo_breach(phase, duration_ms, tracker)
        
        # Update dashboard if in DEBUG mode
        if logger.level <= 10:  # DEBUG level
            self._maybe_refresh_dashboard()
    
    def record_pipeline_completion(self, tracker: PipelineTracker):
        """Record complete pipeline metrics from tracker [PA]."""
        # Record individual phase metrics
        for phase_name, phase_metric in tracker.phases.items():
            if phase_metric.duration_ms is not None:
                self.record_phase_metric(phase_name, phase_metric.duration_ms, tracker)
        
        # Record total pipeline time
        if tracker.total_duration_ms:
            self.record_phase_metric("pipeline_total", tracker.total_duration_ms, tracker)
    
    def _check_slo_breach(self, phase: str, duration_ms: int, tracker: Optional[PipelineTracker] = None):
        """Check if measurement breaches SLO target [REH]."""
        target = self.targets[phase]
        is_breach = duration_ms > target.target_ms
        
        if is_breach:
            self.consecutive_breaches[phase] += 1
            self.monitoring_stats["slo_breaches"] += 1
            
            breach_percentage = int((duration_ms / target.target_ms - 1) * 100)
            
            # Determine alert level
            if duration_ms > target.target_ms * target.critical_multiplier:
                alert_level = AlertLevel.CRITICAL
            elif duration_ms > target.target_ms * target.warning_multiplier:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.INFO
            
            # Fire alert if consecutive threshold reached
            consecutive_count = self.consecutive_breaches[phase]
            if consecutive_count >= target.alert_threshold_consecutive:
                self._fire_alert(
                    alert_level=alert_level,
                    metric_name=phase,
                    message=f"SLO breach: {target.name} took {duration_ms}ms (target: {target.target_ms}ms, +{breach_percentage}%)",
                    value=duration_ms,
                    target=target.target_ms,
                    consecutive_count=consecutive_count,
                    tracker=tracker
                )
        else:
            # Reset consecutive breaches on success
            self.consecutive_breaches[phase] = 0
    
    def _fire_alert(
        self, 
        alert_level: AlertLevel,
        metric_name: str,
        message: str,
        value: float,
        target: float,
        consecutive_count: int = 1,
        tracker: Optional[PipelineTracker] = None
    ):
        """Fire SLO alert with callbacks [REH]."""
        alert = AlertHistory(
            alert_level=alert_level,
            message=message,
            timestamp=time.time(),
            metric_name=metric_name,
            value=value,
            target=target,
            metadata={
                "consecutive_count": consecutive_count,
                "tracker_id": tracker.corr_id if tracker else None,
                "msg_id": tracker.msg_id if tracker else None
            }
        )
        
        self.alert_history.append(alert)
        self.monitoring_stats["alerts_fired"] += 1
        
        # Limit alert history size
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)
        
        # Log alert with appropriate level
        alert_icons = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️", 
            AlertLevel.CRITICAL: "🚨"
        }
        
        icon = alert_icons[alert_level]
        log_msg = f"{icon} SLO ALERT ({alert_level.value.upper()}): {message}"
        
        if alert_level == AlertLevel.CRITICAL:
            logger.error(log_msg, extra={"detail": alert.metadata})
        elif alert_level == AlertLevel.WARNING:
            logger.warning(log_msg, extra={"detail": alert.metadata})
        else:
            logger.info(log_msg, extra={"detail": alert.metadata})
        
        # Execute registered callbacks
        for callback in self.alert_callbacks.get(alert_level, []):
            try:
                asyncio.create_task(callback(alert))
            except Exception as e:
                logger.error(f"❌ Alert callback error: {e}")
    
    def register_alert_callback(self, alert_level: AlertLevel, callback: Callable):
        """Register callback for alert level [REH]."""
        self.alert_callbacks[alert_level].append(callback)
        logger.debug(f"Registered {alert_level.value} alert callback")
    
    def _maybe_refresh_dashboard(self):
        """Refresh dashboard if interval elapsed [PA]."""
        current_time = time.time()
        if (current_time - self.last_dashboard_update) >= self.dashboard_refresh_interval:
            self._render_debug_dashboard()
            self.last_dashboard_update = current_time
    
    def _render_debug_dashboard(self):
        """Render Rich dashboard for DEBUG mode [CA]."""
        try:
            dashboard = self._create_slo_dashboard()
            
            # Log dashboard as special debug info
            logger.debug("SLO Performance Dashboard", extra={"rich_panel": dashboard})
            self.monitoring_stats["dashboard_renders"] += 1
            
        except Exception as e:
            logger.error(f"❌ Dashboard render error: {e}")
    
    def _create_slo_dashboard(self) -> Panel:
        """Create comprehensive SLO dashboard with Rich components [CA]."""
        # Main tree structure
        tree = Tree("📊 SLO Performance Dashboard")
        
        # Add SLO targets overview
        targets_node = tree.add("🎯 SLO Targets & Current Status")
        
        for phase, target in self.targets.items():
            window = self.metric_windows[phase]
            stats = window.get_stats()
            
            if stats["count"] == 0:
                status = "❓ No Data"
                detail = "No measurements recorded"
            else:
                p95 = stats.get("p95", 0)
                if p95 is None:
                    status = "❓ Insufficient Data"
                    detail = f"{stats['count']} samples"
                elif p95 <= target.target_ms:
                    status = "✅ Within SLO"
                    breach_pct = int((p95 / target.target_ms) * 100)
                    detail = f"p95: {p95:.0f}ms ({breach_pct}% of target)"
                else:
                    status = "❌ SLO Breach" 
                    breach_pct = int((p95 / target.target_ms - 1) * 100)
                    detail = f"p95: {p95:.0f}ms (+{breach_pct}% over target)"
            
            phase_display = target.name.replace("_", " ").title()
            targets_node.add(f"{status} {phase_display}: {detail}")
        
        # Add recent alerts
        alerts_node = tree.add("🚨 Recent Alerts")
        recent_alerts = self.alert_history[-5:]  # Last 5 alerts
        
        if not recent_alerts:
            alerts_node.add("✅ No Recent Alerts")
        else:
            for alert in reversed(recent_alerts):
                alert_time = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
                alert_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[alert.alert_level.value]
                alerts_node.add(f"{alert_icon} {alert_time}: {alert.message}")
        
        # Add performance statistics
        stats_node = tree.add("📈 Monitoring Statistics")
        stats_node.add(f"Total Measurements: {self.monitoring_stats['total_measurements']:,}")
        stats_node.add(f"SLO Breaches: {self.monitoring_stats['slo_breaches']:,}")
        stats_node.add(f"Alerts Fired: {self.monitoring_stats['alerts_fired']:,}")
        stats_node.add(f"Dashboard Renders: {self.monitoring_stats['dashboard_renders']:,}")
        
        # Create panel with title
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        panel = Panel(
            tree,
            title=f"SLO Dashboard - {current_time}",
            border_style="blue",
            padding=(1, 2)
        )
        
        return panel
    
    def get_current_slo_status(self) -> Dict[str, Any]:
        """Get current SLO status summary [PA]."""
        status = {}
        
        for phase, target in self.targets.items():
            window = self.metric_windows[phase]
            stats = window.get_stats()
            
            p95 = stats.get("p95")
            status[phase] = {
                "target_ms": target.target_ms,
                "current_p95_ms": p95,
                "within_slo": p95 <= target.target_ms if p95 is not None else None,
                "sample_count": stats["count"],
                "consecutive_breaches": self.consecutive_breaches[phase],
                "breach_percentage": int((p95 / target.target_ms - 1) * 100) if p95 and p95 > target.target_ms else 0
            }
        
        return status
    
    def get_performance_report(self) -> str:
        """Generate text performance report [CA]."""
        report = []
        report.append("=== SLO Performance Report ===")
        report.append("")
        
        for phase, target in self.targets.items():
            window = self.metric_windows[phase]
            stats = window.get_stats()
            
            phase_name = target.name
            report.append(f"📊 {phase_name}:")
            report.append(f"   Target: {target.target_ms}ms")
            
            if stats["count"] == 0:
                report.append("   Status: No data available")
            else:
                p95 = stats.get("p95", 0)
                if p95:
                    status_icon = "✅" if p95 <= target.target_ms else "❌"
                    report.append(f"   Status: {status_icon} p95={p95:.0f}ms ({stats['count']} samples)")
                    report.append(f"   Range: {stats['min']:.0f}ms - {stats['max']:.0f}ms")
                    report.append(f"   Mean: {stats['mean']:.0f}ms, Median: {stats['median']:.0f}ms")
                else:
                    report.append("   Status: Insufficient data for p95")
            
            report.append("")
        
        # Add alert summary
        recent_alerts = len([a for a in self.alert_history if time.time() - a.timestamp < 3600])  # Last hour
        report.append(f"🚨 Alerts in last hour: {recent_alerts}")
        report.append(f"📊 Total measurements: {self.monitoring_stats['total_measurements']:,}")
        
        return "\n".join(report)
    
    async def cleanup(self):
        """Clean up monitoring resources [RM]."""
        self.metric_windows.clear()
        self.alert_history.clear()
        self.alert_callbacks.clear()
        logger.debug("🧹 SLOMonitor cleaned up")

# Global SLO monitor instance [PA]
_slo_monitor_instance: Optional[SLOMonitor] = None

def get_slo_monitor() -> SLOMonitor:
    """Get global SLO monitor instance."""
    global _slo_monitor_instance
    
    if _slo_monitor_instance is None:
        _slo_monitor_instance = SLOMonitor()
        logger.info("🚀 Global SLOMonitor created")
    
    return _slo_monitor_instance

async def cleanup_slo_monitor():
    """Clean up global SLO monitor [RM]."""
    global _slo_monitor_instance
    if _slo_monitor_instance:
        await _slo_monitor_instance.cleanup()
        _slo_monitor_instance = None
