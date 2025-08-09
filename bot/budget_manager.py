"""
Budget and deadline management system for router optimization. [PA][REH]

This module provides adaptive timeout management with soft/hard deadlines:
- Soft budgets trigger route switches (try alternative methods)
- Hard deadlines trigger cancellation with partial results
- Adaptive budgets based on p95 performance tracking
- Per-family timeout configurations (Tweet, Video, OCR, etc.)
- Hierarchical cancellation when deadlines exceeded

Key features:
- Tracks p95 latencies per operation type
- Adaptive soft budgets = max(baseline, 1.2×p95) within clamps
- Route switching on soft budget exceed (graceful fallback)
- Hard cancellation with partial results on deadline exceed
- Comprehensive metrics for budget tuning
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable, TypeVar
import statistics
from contextlib import asynccontextmanager

from .util.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class BudgetFamily(Enum):
    """Budget families with different timeout characteristics. [CMV]"""
    TWEET_SYNDICATION = "tweet_syndication"       # 3.5s soft, 5s hard
    TWEET_WEB_TIER_A = "tweet_web_tier_a"         # 2.5s soft, 4s hard  
    TWEET_WEB_TIER_B = "tweet_web_tier_b"         # 8s soft, 12s hard
    TWEET_WEB_TIER_C = "tweet_web_tier_c"         # 8s soft, 12s hard
    X_API_CALL = "x_api_call"                     # 6s soft, 10s hard
    VIDEO_PROBE = "video_probe"                   # 0.7s soft, 1.5s hard
    VIDEO_DOWNLOAD = "video_download"             # 2.5s soft, 5s hard
    STT_PROCESSING = "stt_processing"             # 300s soft, 400s hard
    WEB_TIER_A = "web_tier_a"                     # 2s soft, 4s hard
    WEB_TIER_B = "web_tier_b"                     # 8s soft, 12s hard  
    WEB_TIER_C = "web_tier_c"                     # 8s soft, 12s hard
    OCR_BATCH = "ocr_batch"                       # 20s soft, 30s hard
    OCR_GLOBAL = "ocr_global"                     # 240s soft, 300s hard
    VISION_INFERENCE = "vision_inference"         # 15s soft, 25s hard
    TEXT_INFERENCE = "text_inference"             # 10s soft, 20s hard

@dataclass
class BudgetConfig:
    """Configuration for a specific budget family. [CMV]"""
    soft_budget_ms: float      # Route switch threshold
    hard_deadline_ms: float    # Cancellation threshold
    baseline_ms: float         # Minimum soft budget (never go below this)
    max_clamp_ms: float        # Maximum soft budget (never go above this)
    adaptive_enabled: bool = True
    p95_window_size: int = 100  # Number of samples for p95 calculation

@dataclass
class BudgetExecution:
    """Tracks execution of a budget-controlled operation. [CA]"""
    family: BudgetFamily
    operation_id: str
    started_at: float
    soft_budget_ms: float
    hard_deadline_ms: float
    route_switched: bool = False
    cancelled: bool = False
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        end_time = self.completed_at or time.time()
        return (end_time - self.started_at) * 1000
    
    @property
    def is_soft_exceeded(self) -> bool:
        """Check if soft budget has been exceeded."""
        return self.elapsed_ms > self.soft_budget_ms
    
    @property
    def is_hard_exceeded(self) -> bool:
        """Check if hard deadline has been exceeded."""
        return self.elapsed_ms > self.hard_deadline_ms

@dataclass
class BudgetMetrics:
    """Metrics for budget tracking and optimization. [PA]"""
    total_operations: int = 0
    completed_operations: int = 0
    route_switches: int = 0
    hard_cancellations: int = 0
    soft_budget_violations: int = 0
    hard_deadline_violations: int = 0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    recent_latencies: List[float] = field(default_factory=list)
    
    def add_latency_sample(self, latency_ms: float, max_samples: int = 100) -> None:
        """Add latency sample and update percentiles."""
        self.recent_latencies.append(latency_ms)
        
        # Keep only recent samples
        if len(self.recent_latencies) > max_samples:
            self.recent_latencies = self.recent_latencies[-max_samples:]
        
        # Update percentiles
        if self.recent_latencies:
            sorted_latencies = sorted(self.recent_latencies)
            n = len(sorted_latencies)
            
            self.p50_latency_ms = sorted_latencies[int(n * 0.5)]
            self.p95_latency_ms = sorted_latencies[int(n * 0.95)]
            self.p99_latency_ms = sorted_latencies[int(n * 0.99)]
            self.avg_latency_ms = sum(sorted_latencies) / n
    
    def update_success_rate(self) -> None:
        """Update success rate calculation."""
        if self.total_operations > 0:
            self.success_rate = self.completed_operations / self.total_operations

class SoftBudgetExceeded(Exception):
    """Exception raised when soft budget is exceeded (route switch). [REH]"""
    def __init__(self, family: BudgetFamily, elapsed_ms: float, budget_ms: float):
        self.family = family
        self.elapsed_ms = elapsed_ms
        self.budget_ms = budget_ms
        super().__init__(f"{family.value} soft budget exceeded: {elapsed_ms:.1f}ms > {budget_ms:.1f}ms")

class HardDeadlineExceeded(Exception):
    """Exception raised when hard deadline is exceeded (cancellation). [REH]"""
    def __init__(self, family: BudgetFamily, elapsed_ms: float, deadline_ms: float):
        self.family = family
        self.elapsed_ms = elapsed_ms
        self.deadline_ms = deadline_ms
        super().__init__(f"{family.value} hard deadline exceeded: {elapsed_ms:.1f}ms > {deadline_ms:.1f}ms")

class BudgetManager:
    """Manages budgets and deadlines with adaptive optimization. [PA][REH]"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize budget manager with configuration."""
        self.config = config or {}
        
        # Default budget configurations
        self.budget_configs = {
            BudgetFamily.TWEET_SYNDICATION: BudgetConfig(
                soft_budget_ms=float(self.config.get('TWEET_SYNDICATION_TOTAL_DEADLINE_MS', 3500)),
                hard_deadline_ms=float(self.config.get('TWEET_SYNDICATION_TOTAL_DEADLINE_MS', 3500)) * 1.5,
                baseline_ms=2000.0,
                max_clamp_ms=8000.0
            ),
            BudgetFamily.TWEET_WEB_TIER_A: BudgetConfig(
                soft_budget_ms=float(self.config.get('TWEET_WEB_TIER_A_MS', 2500)),
                hard_deadline_ms=float(self.config.get('TWEET_WEB_TIER_A_MS', 2500)) * 1.5,
                baseline_ms=1500.0,
                max_clamp_ms=5000.0
            ),
            BudgetFamily.TWEET_WEB_TIER_B: BudgetConfig(
                soft_budget_ms=float(self.config.get('TWEET_WEB_TIER_B_MS', 8000)),
                hard_deadline_ms=float(self.config.get('TWEET_WEB_TIER_B_MS', 8000)) * 1.5,
                baseline_ms=5000.0,
                max_clamp_ms=15000.0
            ),
            BudgetFamily.TWEET_WEB_TIER_C: BudgetConfig(
                soft_budget_ms=float(self.config.get('TWEET_WEB_TIER_C_MS', 8000)),
                hard_deadline_ms=float(self.config.get('TWEET_WEB_TIER_C_MS', 8000)) * 1.5,
                baseline_ms=5000.0,
                max_clamp_ms=15000.0
            ),
            BudgetFamily.X_API_CALL: BudgetConfig(
                soft_budget_ms=float(self.config.get('X_API_TOTAL_DEADLINE_MS', 6000)),
                hard_deadline_ms=float(self.config.get('X_API_TOTAL_DEADLINE_MS', 6000)) * 1.5,
                baseline_ms=3000.0,
                max_clamp_ms=12000.0
            ),
            BudgetFamily.VIDEO_PROBE: BudgetConfig(
                soft_budget_ms=700.0,
                hard_deadline_ms=1500.0,
                baseline_ms=500.0,
                max_clamp_ms=2000.0
            ),
            BudgetFamily.STT_PROCESSING: BudgetConfig(
                soft_budget_ms=float(self.config.get('STT_TOTAL_DEADLINE_MS', 300000)),
                hard_deadline_ms=float(self.config.get('STT_TOTAL_DEADLINE_MS', 300000)) * 1.2,
                baseline_ms=60000.0,
                max_clamp_ms=600000.0
            ),
            BudgetFamily.WEB_TIER_A: BudgetConfig(
                soft_budget_ms=float(self.config.get('WEB_TIER_A_MS', 2000)),
                hard_deadline_ms=float(self.config.get('WEB_TIER_A_MS', 2000)) * 1.5,
                baseline_ms=1000.0,
                max_clamp_ms=4000.0
            ),
            BudgetFamily.OCR_BATCH: BudgetConfig(
                soft_budget_ms=float(self.config.get('OCR_BATCH_DEADLINE_MS', 20000)),
                hard_deadline_ms=float(self.config.get('OCR_BATCH_DEADLINE_MS', 20000)) * 1.2,
                baseline_ms=10000.0,
                max_clamp_ms=40000.0
            ),
            BudgetFamily.OCR_GLOBAL: BudgetConfig(
                soft_budget_ms=float(self.config.get('OCR_GLOBAL_DEADLINE_MS', 240000)),
                hard_deadline_ms=float(self.config.get('OCR_GLOBAL_DEADLINE_MS', 240000)) * 1.2,
                baseline_ms=120000.0,
                max_clamp_ms=480000.0
            ),
            BudgetFamily.VISION_INFERENCE: BudgetConfig(
                soft_budget_ms=15000.0,
                hard_deadline_ms=25000.0,
                baseline_ms=8000.0,
                max_clamp_ms=40000.0
            ),
            BudgetFamily.TEXT_INFERENCE: BudgetConfig(
                soft_budget_ms=10000.0,
                hard_deadline_ms=20000.0,
                baseline_ms=5000.0,
                max_clamp_ms=30000.0
            ),
        }
        
        # Metrics tracking
        self.metrics: Dict[BudgetFamily, BudgetMetrics] = {
            family: BudgetMetrics() for family in BudgetFamily
        }
        
        # Active executions
        self.active_executions: Dict[str, BudgetExecution] = {}
        
        logger.info("⏱️ BudgetManager initialized with adaptive timeout management")
    
    def _calculate_adaptive_soft_budget(self, family: BudgetFamily) -> float:
        """Calculate adaptive soft budget based on p95 performance. [PA]"""
        config = self.budget_configs[family]
        metrics = self.metrics[family]
        
        if not config.adaptive_enabled or metrics.p95_latency_ms == 0:
            return config.soft_budget_ms
        
        # Adaptive budget: max(baseline, 1.2×p95) within clamps
        adaptive_budget = max(
            config.baseline_ms,
            min(config.max_clamp_ms, metrics.p95_latency_ms * 1.2)
        )
        
        return adaptive_budget
    
    @asynccontextmanager
    async def execute_with_budget(
        self,
        family: BudgetFamily,
        operation_id: str,
        check_interval_ms: float = 100.0
    ):
        """Context manager for budget-controlled execution. [CA][REH]"""
        # Calculate current budgets
        soft_budget_ms = self._calculate_adaptive_soft_budget(family)
        hard_deadline_ms = self.budget_configs[family].hard_deadline_ms
        
        # Create execution tracking
        execution = BudgetExecution(
            family=family,
            operation_id=operation_id,
            started_at=time.time(),
            soft_budget_ms=soft_budget_ms,
            hard_deadline_ms=hard_deadline_ms
        )
        
        self.active_executions[operation_id] = execution
        self.metrics[family].total_operations += 1
        
        # Start budget monitoring task
        monitor_task = asyncio.create_task(
            self._monitor_budget(execution, check_interval_ms)
        )
        
        try:
            yield execution
            
            # Mark as completed
            execution.completed_at = time.time()
            self.metrics[family].completed_operations += 1
            
            # Update metrics
            latency_ms = execution.elapsed_ms
            self.metrics[family].add_latency_sample(latency_ms)
            self.metrics[family].update_success_rate()
            
            logger.debug(
                f"✅ {family.value} completed in {latency_ms:.1f}ms "
                f"(budget: {soft_budget_ms:.1f}ms, deadline: {hard_deadline_ms:.1f}ms)",
                extra={
                    "event": "budget.completed",
                    "detail": {
                        "family": family.value,
                        "operation_id": operation_id,
                        "latency_ms": latency_ms,
                        "soft_budget_ms": soft_budget_ms,
                        "hard_deadline_ms": hard_deadline_ms
                    }
                }
            )
            
        except SoftBudgetExceeded as e:
            execution.route_switched = True
            self.metrics[family].route_switches += 1
            self.metrics[family].soft_budget_violations += 1
            
            logger.warning(
                f"⚠️ {family.value} soft budget exceeded, route switching: {e}",
                extra={
                    "event": "budget.soft_exceeded",
                    "detail": {
                        "family": family.value,
                        "operation_id": operation_id,
                        "elapsed_ms": e.elapsed_ms,
                        "budget_ms": e.budget_ms
                    }
                }
            )
            raise
            
        except HardDeadlineExceeded as e:
            execution.cancelled = True
            self.metrics[family].hard_cancellations += 1
            self.metrics[family].hard_deadline_violations += 1
            
            logger.error(
                f"❌ {family.value} hard deadline exceeded, cancelling: {e}",
                extra={
                    "event": "budget.hard_exceeded",
                    "detail": {
                        "family": family.value,
                        "operation_id": operation_id,
                        "elapsed_ms": e.elapsed_ms,
                        "deadline_ms": e.deadline_ms
                    }
                }
            )
            raise
            
        except Exception as e:
            execution.error = e
            logger.error(f"❌ {family.value} failed: {e}")
            raise
            
        finally:
            # Clean up
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            if operation_id in self.active_executions:
                del self.active_executions[operation_id]
    
    async def _monitor_budget(self, execution: BudgetExecution, check_interval_ms: float) -> None:
        """Monitor execution against budget and deadline. [REH]"""
        check_interval_s = check_interval_ms / 1000.0
        
        try:
            while True:
                await asyncio.sleep(check_interval_s)
                
                # Check if already completed
                if execution.completed_at is not None:
                    break
                
                # Check hard deadline first (more critical)
                if execution.is_hard_exceeded:
                    raise HardDeadlineExceeded(
                        execution.family,
                        execution.elapsed_ms,
                        execution.hard_deadline_ms
                    )
                
                # Check soft budget (route switch opportunity)
                if execution.is_soft_exceeded and not execution.route_switched:
                    raise SoftBudgetExceeded(
                        execution.family,
                        execution.elapsed_ms,
                        execution.soft_budget_ms
                    )
                    
        except asyncio.CancelledError:
            # Monitor task was cancelled (normal cleanup)
            pass
        except (SoftBudgetExceeded, HardDeadlineExceeded):
            # Re-raise budget/deadline exceptions
            raise
    
    async def run_with_budget(
        self,
        family: BudgetFamily,
        operation_id: str,
        coro: Awaitable[T],
        on_soft_exceeded: Optional[Callable[[], Awaitable[T]]] = None
    ) -> T:
        """Run coroutine with budget control and optional route switching. [PA][REH]"""
        async with self.execute_with_budget(family, operation_id) as execution:
            try:
                # Run the main operation
                result = await coro
                execution.result = result
                return result
                
            except SoftBudgetExceeded:
                # Try route switching if handler provided
                if on_soft_exceeded is not None:
                    logger.info(f"🔄 {family.value} route switching after soft budget exceeded")
                    try:
                        result = await on_soft_exceeded()
                        execution.result = result
                        return result
                    except Exception as fallback_error:
                        logger.error(f"❌ {family.value} route switch also failed: {fallback_error}")
                        raise fallback_error
                else:
                    # No route switching available, re-raise
                    raise
    
    def get_metrics(self, family: Optional[BudgetFamily] = None) -> Dict[BudgetFamily, BudgetMetrics]:
        """Get metrics for specific family or all families. [PA]"""
        if family is not None:
            return {family: self.metrics[family]}
        return self.metrics.copy()
    
    def get_active_executions(self) -> Dict[str, BudgetExecution]:
        """Get currently active executions. [PA]"""
        return self.active_executions.copy()
    
    def update_budget_config(self, family: BudgetFamily, config: BudgetConfig) -> None:
        """Update budget configuration for a family. [CMV]"""
        self.budget_configs[family] = config
        logger.info(f"🔧 Updated budget config for {family.value}")

# Global singleton instance
_budget_manager_instance: Optional[BudgetManager] = None

def get_budget_manager(config: Optional[Dict[str, Any]] = None) -> BudgetManager:
    """Get or create the global budget manager instance. [CA]"""
    global _budget_manager_instance
    
    if _budget_manager_instance is None:
        _budget_manager_instance = BudgetManager(config)
    
    return _budget_manager_instance

def cleanup_budget_manager() -> None:
    """Clean up the global budget manager instance."""
    global _budget_manager_instance
    _budget_manager_instance = None
