"""System resource monitoring with event loop lag and RSS tracking.

Provides comprehensive system resource monitoring including event loop lag detection,
memory usage tracking, and threshold-based warning systems for operational awareness.

[RAT: PA] - Performance Awareness
"""

from __future__ import annotations

import asyncio
import os
import time
import logging
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from bot.util.logging import get_logger
from bot.metrics import metrics, METRIC_PROCESS_RSS_BYTES, METRIC_EVENT_LOOP_LAG_SECONDS


# Resource monitoring thresholds [CMV]
DEFAULT_EVENT_LOOP_LAG_WARNING_MS = 50.0    # Warn if event loop lag > 50ms
DEFAULT_EVENT_LOOP_LAG_CRITICAL_MS = 200.0  # Critical if event loop lag > 200ms
DEFAULT_RSS_WARNING_MB = 1024.0             # Warn if RSS > 1GB
DEFAULT_RSS_CRITICAL_MB = 2048.0            # Critical if RSS > 2GB
DEFAULT_CPU_WARNING_PERCENT = 80.0          # Warn if CPU > 80%
DEFAULT_CPU_CRITICAL_PERCENT = 95.0         # Critical if CPU > 95%

# Monitoring intervals [CMV]
DEFAULT_RESOURCE_CHECK_INTERVAL = 30.0      # Seconds between resource checks
DEFAULT_EVENT_LOOP_CHECK_INTERVAL = 10.0   # Seconds between event loop lag checks
DEFAULT_SAMPLE_COUNT = 5                    # Number of samples for averaging


@dataclass
class ResourceThresholds:
    """Resource monitoring thresholds configuration."""
    event_loop_lag_warning_ms: float = DEFAULT_EVENT_LOOP_LAG_WARNING_MS
    event_loop_lag_critical_ms: float = DEFAULT_EVENT_LOOP_LAG_CRITICAL_MS
    rss_warning_mb: float = DEFAULT_RSS_WARNING_MB
    rss_critical_mb: float = DEFAULT_RSS_CRITICAL_MB
    cpu_warning_percent: float = DEFAULT_CPU_WARNING_PERCENT
    cpu_critical_percent: float = DEFAULT_CPU_CRITICAL_PERCENT


@dataclass
class ResourceSnapshot:
    """Point-in-time system resource snapshot."""
    timestamp: float = field(default_factory=time.time)
    rss_bytes: int = 0
    rss_mb: float = 0.0
    cpu_percent: float = 0.0
    event_loop_lag_ms: float = 0.0
    open_files: int = 0
    thread_count: int = 0


@dataclass
class ResourceStats:
    """Aggregated resource statistics over time."""
    sample_count: int = 0
    avg_rss_mb: float = 0.0
    max_rss_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_event_loop_lag_ms: float = 0.0
    max_event_loop_lag_ms: float = 0.0
    last_warning_time: Dict[str, float] = field(default_factory=dict)


class EventLoopMonitor:
    """Event loop lag monitoring with high-precision measurement.
    
    Measures event loop responsiveness by timing small async operations
    and comparing against expected duration.
    
    [RAT: PA] - Performance Awareness
    """

    def __init__(self, sample_duration_ms: float = 1.0):
        """Initialize event loop monitor.
        
        Args:
            sample_duration_ms: Duration of test async operation in milliseconds
        """
        self.logger = get_logger(__name__)
        self.sample_duration_ms = sample_duration_ms
        self.sample_duration_s = sample_duration_ms / 1000.0
        self._baseline_overhead = None

    async def _calibrate_baseline(self) -> float:
        """Calibrate baseline overhead for accurate lag measurement."""
        if self._baseline_overhead is not None:
            return self._baseline_overhead

        # Perform multiple baseline measurements
        baseline_samples = []
        for _ in range(10):
            start = time.perf_counter()
            await asyncio.sleep(0)  # Minimal async operation
            end = time.perf_counter()
            baseline_samples.append((end - start) * 1000)

        # Use median to filter out outliers
        baseline_samples.sort()
        self._baseline_overhead = baseline_samples[len(baseline_samples) // 2]
        
        self.logger.debug(f"Event loop baseline overhead: {self._baseline_overhead:.3f}ms",
                         extra={"subsys": "resource_monitor"})
        
        return self._baseline_overhead

    async def measure_lag(self) -> float:
        """Measure current event loop lag in milliseconds.
        
        Returns:
            Event loop lag in milliseconds
        """
        try:
            # Ensure baseline is calibrated
            baseline = await self._calibrate_baseline()
            
            # Measure actual sleep duration
            start = time.perf_counter()
            await asyncio.sleep(self.sample_duration_s)
            end = time.perf_counter()
            
            actual_duration_ms = (end - start) * 1000
            expected_duration_ms = self.sample_duration_ms
            
            # Calculate lag (subtract baseline overhead)
            raw_lag = actual_duration_ms - expected_duration_ms
            lag_ms = max(0, raw_lag - baseline)
            
            return lag_ms
            
        except Exception as e:
            self.logger.warning(f"Failed to measure event loop lag: {e}",
                               extra={"subsys": "resource_monitor"})
            return 0.0

    async def measure_lag_averaged(self, sample_count: int = 3) -> float:
        """Measure average event loop lag over multiple samples.
        
        Args:
            sample_count: Number of samples to average
            
        Returns:
            Average event loop lag in milliseconds
        """
        if sample_count <= 0:
            return 0.0
            
        samples = []
        for _ in range(sample_count):
            lag = await self.measure_lag()
            samples.append(lag)
            
            # Small delay between samples to prevent measurement interference
            if len(samples) < sample_count:
                await asyncio.sleep(0.1)
        
        return sum(samples) / len(samples) if samples else 0.0


class ResourceMonitor:
    """Comprehensive system resource monitoring with threshold alerting.
    
    Features:
    - Real-time RSS memory tracking
    - Event loop lag monitoring  
    - CPU usage monitoring
    - File descriptor and thread monitoring
    - Configurable warning/critical thresholds
    - Metrics emission and structured logging
    - Rate-limited alerting to prevent spam
    
    [RAT: PA] - Performance Awareness
    """

    def __init__(self, 
                 thresholds: Optional[ResourceThresholds] = None,
                 check_interval: float = DEFAULT_RESOURCE_CHECK_INTERVAL):
        """Initialize resource monitor.
        
        Args:
            thresholds: Resource warning/critical thresholds
            check_interval: Seconds between monitoring checks
        """
        self.logger = get_logger(__name__)
        self.thresholds = thresholds or ResourceThresholds()
        self.check_interval = check_interval
        self.event_loop_monitor = EventLoopMonitor()
        
        # State tracking
        self.stats = ResourceStats()
        self.enabled = os.getenv("OBS_ENABLE_RESOURCE_METRICS", "true").lower() == "true"
        self.monitor_task: Optional[asyncio.Task] = None
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Process reference for efficient monitoring
        try:
            self.process = psutil.Process()
        except Exception as e:
            self.logger.error(f"Failed to initialize process monitor: {e}",
                             extra={"subsys": "resource_monitor"})
            self.process = None

    async def get_resource_snapshot(self) -> ResourceSnapshot:
        """Get current system resource snapshot.
        
        Returns:
            ResourceSnapshot with current system metrics
        """
        snapshot = ResourceSnapshot()
        
        if not self.process:
            return snapshot
            
        try:
            # Memory information
            memory_info = self.process.memory_info()
            snapshot.rss_bytes = memory_info.rss
            snapshot.rss_mb = memory_info.rss / (1024 * 1024)
            
            # CPU usage (non-blocking)
            snapshot.cpu_percent = self.process.cpu_percent()
            
            # File descriptors and threads
            try:
                snapshot.open_files = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            except (AttributeError, psutil.AccessDenied):
                snapshot.open_files = 0
                
            try:
                snapshot.thread_count = self.process.num_threads()
            except psutil.AccessDenied:
                snapshot.thread_count = 0
            
            # Event loop lag measurement
            snapshot.event_loop_lag_ms = await self.event_loop_monitor.measure_lag_averaged(3)
            
        except Exception as e:
            self.logger.warning(f"Error collecting resource snapshot: {e}",
                               extra={"subsys": "resource_monitor"})
            
        return snapshot

    def _check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """Check resource thresholds and emit warnings if exceeded.
        
        Args:
            snapshot: Resource snapshot to check
        """
        current_time = time.time()
        warning_cooldown = 300.0  # 5 minute cooldown between same warnings
        
        # Event loop lag checks
        if snapshot.event_loop_lag_ms > self.thresholds.event_loop_lag_critical_ms:
            if self._should_emit_warning("event_loop_lag_critical", current_time, warning_cooldown):
                self.logger.critical(
                    f"🚨 CRITICAL: Event loop lag {snapshot.event_loop_lag_ms:.1f}ms "
                    f"(threshold: {self.thresholds.event_loop_lag_critical_ms:.1f}ms). "
                    f"System may be severely overloaded or blocked.",
                    extra={"subsys": "resource_monitor", "event_loop_lag_ms": snapshot.event_loop_lag_ms}
                )
        elif snapshot.event_loop_lag_ms > self.thresholds.event_loop_lag_warning_ms:
            if self._should_emit_warning("event_loop_lag_warning", current_time, warning_cooldown):
                self.logger.warning(
                    f"⚠️  WARNING: Event loop lag {snapshot.event_loop_lag_ms:.1f}ms "
                    f"(threshold: {self.thresholds.event_loop_lag_warning_ms:.1f}ms). "
                    f"Consider reducing async workload.",
                    extra={"subsys": "resource_monitor", "event_loop_lag_ms": snapshot.event_loop_lag_ms}
                )

        # RSS memory checks
        if snapshot.rss_mb > self.thresholds.rss_critical_mb:
            if self._should_emit_warning("rss_critical", current_time, warning_cooldown):
                self.logger.critical(
                    f"🚨 CRITICAL: RSS memory usage {snapshot.rss_mb:.1f}MB "
                    f"(threshold: {self.thresholds.rss_critical_mb:.1f}MB). "
                    f"System may be running out of memory.",
                    extra={"subsys": "resource_monitor", "rss_mb": snapshot.rss_mb}
                )
        elif snapshot.rss_mb > self.thresholds.rss_warning_mb:
            if self._should_emit_warning("rss_warning", current_time, warning_cooldown):
                self.logger.warning(
                    f"⚠️  WARNING: RSS memory usage {snapshot.rss_mb:.1f}MB "
                    f"(threshold: {self.thresholds.rss_warning_mb:.1f}MB). "
                    f"Monitor for memory leaks.",
                    extra={"subsys": "resource_monitor", "rss_mb": snapshot.rss_mb}
                )

        # CPU usage checks  
        if snapshot.cpu_percent > self.thresholds.cpu_critical_percent:
            if self._should_emit_warning("cpu_critical", current_time, warning_cooldown):
                self.logger.critical(
                    f"🚨 CRITICAL: CPU usage {snapshot.cpu_percent:.1f}% "
                    f"(threshold: {self.thresholds.cpu_critical_percent:.1f}%). "
                    f"System may be severely overloaded.",
                    extra={"subsys": "resource_monitor", "cpu_percent": snapshot.cpu_percent}
                )
        elif snapshot.cpu_percent > self.thresholds.cpu_warning_percent:
            if self._should_emit_warning("cpu_warning", current_time, warning_cooldown):
                self.logger.warning(
                    f"⚠️  WARNING: CPU usage {snapshot.cpu_percent:.1f}% "
                    f"(threshold: {self.thresholds.cpu_warning_percent:.1f}%). "
                    f"Consider reducing computational workload.",
                    extra={"subsys": "resource_monitor", "cpu_percent": snapshot.cpu_percent}
                )

    def _should_emit_warning(self, warning_type: str, current_time: float, cooldown: float) -> bool:
        """Check if warning should be emitted based on cooldown period."""
        last_warning = self.stats.last_warning_time.get(warning_type, 0)
        if current_time - last_warning >= cooldown:
            self.stats.last_warning_time[warning_type] = current_time
            return True
        return False

    def _emit_metrics(self, snapshot: ResourceSnapshot) -> None:
        """Emit resource metrics for monitoring systems.
        
        Args:
            snapshot: Resource snapshot to emit metrics for
        """
        if not self.enabled:
            return
            
        try:
            # Core metrics
            metrics.gauge(METRIC_PROCESS_RSS_BYTES, snapshot.rss_bytes)
            metrics.gauge(METRIC_EVENT_LOOP_LAG_SECONDS, snapshot.event_loop_lag_ms / 1000.0)
            
            # Additional metrics (using observe for histograms)
            metrics.observe("bot_process_cpu_percent", snapshot.cpu_percent)
            metrics.observe("bot_process_open_files", snapshot.open_files)
            metrics.observe("bot_process_thread_count", snapshot.thread_count)
            
        except Exception as e:
            self.logger.debug(f"Failed to emit resource metrics: {e}",
                             extra={"subsys": "resource_monitor"})

    def _update_stats(self, snapshot: ResourceSnapshot) -> None:
        """Update aggregated statistics with new snapshot.
        
        Args:
            snapshot: New resource snapshot
        """
        self.stats.sample_count += 1
        
        # Update running averages (simple moving average)
        if self.stats.sample_count == 1:
            # First sample - initialize
            self.stats.avg_rss_mb = snapshot.rss_mb
            self.stats.avg_cpu_percent = snapshot.cpu_percent
            self.stats.avg_event_loop_lag_ms = snapshot.event_loop_lag_ms
        else:
            # Update running averages
            alpha = 1.0 / min(self.stats.sample_count, 20)  # Limit window to 20 samples
            self.stats.avg_rss_mb = (1 - alpha) * self.stats.avg_rss_mb + alpha * snapshot.rss_mb
            self.stats.avg_cpu_percent = (1 - alpha) * self.stats.avg_cpu_percent + alpha * snapshot.cpu_percent
            self.stats.avg_event_loop_lag_ms = (1 - alpha) * self.stats.avg_event_loop_lag_ms + alpha * snapshot.event_loop_lag_ms
        
        # Update maximums
        self.stats.max_rss_mb = max(self.stats.max_rss_mb, snapshot.rss_mb)
        self.stats.max_cpu_percent = max(self.stats.max_cpu_percent, snapshot.cpu_percent)
        self.stats.max_event_loop_lag_ms = max(self.stats.max_event_loop_lag_ms, snapshot.event_loop_lag_ms)

    async def start_monitoring(self) -> None:
        """Start continuous resource monitoring."""
        if not self.enabled:
            self.logger.info("📊 Resource monitoring disabled by config",
                           extra={"subsys": "resource_monitor"})
            return
            
        if self.monitor_task and not self.monitor_task.done():
            self.logger.warning("Resource monitoring already running",
                               extra={"subsys": "resource_monitor"})
            return
            
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(), name="resource_monitor")
        
        self.logger.info(
            f"📊 Started resource monitoring (interval: {self.check_interval}s, "
            f"event_loop_lag_warning: {self.thresholds.event_loop_lag_warning_ms}ms, "
            f"rss_warning: {self.thresholds.rss_warning_mb}MB)",
            extra={"subsys": "resource_monitor"}
        )

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running and not self._shutdown_event.is_set():
            try:
                # Get current resource snapshot
                snapshot = await self.get_resource_snapshot()
                
                # Update statistics
                self._update_stats(snapshot)
                
                # Check thresholds and emit warnings
                self._check_thresholds(snapshot)
                
                # Emit metrics
                self._emit_metrics(snapshot)
                
                # Log periodic status (every 10 checks)
                if self.stats.sample_count % 10 == 0:
                    self.logger.info(
                        f"📊 Resource status: RSS {snapshot.rss_mb:.1f}MB, "
                        f"CPU {snapshot.cpu_percent:.1f}%, "
                        f"Event loop lag {snapshot.event_loop_lag_ms:.1f}ms",
                        extra={
                            "subsys": "resource_monitor",
                            "rss_mb": snapshot.rss_mb,
                            "cpu_percent": snapshot.cpu_percent,
                            "event_loop_lag_ms": snapshot.event_loop_lag_ms
                        }
                    )
                
                # Wait for next check or shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.check_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue monitoring
                    
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}", exc_info=True,
                                 extra={"subsys": "resource_monitor"})
                await asyncio.sleep(self.check_interval)

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring gracefully."""
        self.running = False
        self._shutdown_event.set()
        
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await asyncio.wait_for(self.monitor_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
                
        self.logger.info("📊 Resource monitoring stopped",
                        extra={"subsys": "resource_monitor"})

    def get_resource_stats(self) -> Dict[str, any]:
        """Get current resource statistics summary.
        
        Returns:
            Dictionary with resource statistics
        """
        return {
            "enabled": self.enabled,
            "sample_count": self.stats.sample_count,
            "avg_rss_mb": round(self.stats.avg_rss_mb, 2),
            "max_rss_mb": round(self.stats.max_rss_mb, 2),
            "avg_cpu_percent": round(self.stats.avg_cpu_percent, 2),
            "max_cpu_percent": round(self.stats.max_cpu_percent, 2),
            "avg_event_loop_lag_ms": round(self.stats.avg_event_loop_lag_ms, 2),
            "max_event_loop_lag_ms": round(self.stats.max_event_loop_lag_ms, 2),
            "thresholds": {
                "event_loop_lag_warning_ms": self.thresholds.event_loop_lag_warning_ms,
                "event_loop_lag_critical_ms": self.thresholds.event_loop_lag_critical_ms,
                "rss_warning_mb": self.thresholds.rss_warning_mb,
                "rss_critical_mb": self.thresholds.rss_critical_mb,
                "cpu_warning_percent": self.thresholds.cpu_warning_percent,
                "cpu_critical_percent": self.thresholds.cpu_critical_percent,
            }
        }


# Global resource monitor instance
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance."""
    global _resource_monitor
    if _resource_monitor is None:
        # Load custom thresholds from environment if set
        thresholds = ResourceThresholds(
            event_loop_lag_warning_ms=float(os.getenv("RESOURCE_EVENT_LOOP_LAG_WARNING_MS", DEFAULT_EVENT_LOOP_LAG_WARNING_MS)),
            event_loop_lag_critical_ms=float(os.getenv("RESOURCE_EVENT_LOOP_LAG_CRITICAL_MS", DEFAULT_EVENT_LOOP_LAG_CRITICAL_MS)),
            rss_warning_mb=float(os.getenv("RESOURCE_RSS_WARNING_MB", DEFAULT_RSS_WARNING_MB)),
            rss_critical_mb=float(os.getenv("RESOURCE_RSS_CRITICAL_MB", DEFAULT_RSS_CRITICAL_MB)),
            cpu_warning_percent=float(os.getenv("RESOURCE_CPU_WARNING_PERCENT", DEFAULT_CPU_WARNING_PERCENT)),
            cpu_critical_percent=float(os.getenv("RESOURCE_CPU_CRITICAL_PERCENT", DEFAULT_CPU_CRITICAL_PERCENT))
        )
        
        check_interval = float(os.getenv("RESOURCE_CHECK_INTERVAL", DEFAULT_RESOURCE_CHECK_INTERVAL))
        _resource_monitor = ResourceMonitor(thresholds, check_interval)
        
    return _resource_monitor
