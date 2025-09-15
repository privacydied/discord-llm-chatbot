"""Startup Orchestrator for parallelized component initialization with dependency management.

Implements dependency-aware concurrent startup with timeout, fallback, and timing support
to achieve 3-5 second startup performance improvement target.

[RAT: CA, PA] - Clean Architecture, Performance Awareness
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from bot.metrics import (
    metrics,
    METRIC_STARTUP_TOTAL_DURATION,
    METRIC_STARTUP_COMPONENT_DURATION,
    METRIC_STARTUP_PARALLEL_GROUPS,
    METRIC_COMPONENT_INIT_SUCCESS,
    METRIC_COMPONENT_INIT_FAILURE,
    METRIC_COMPONENT_LAST_INIT_TIMESTAMP,
    METRIC_DEGRADED_MODE,
)


class ComponentStatus(Enum):
    """Status of component initialization."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    DEGRADED = "degraded"  # Success with fallback
    FAILED = "failed"


class ErrorClass(Enum):
    """Classification of initialization errors for handling strategy."""

    CONFIG_ERROR = "config_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    EXTERNAL_TIMEOUT = "external_timeout"
    PROGRAMMING_ERROR = "programming_error"


@dataclass
class ComponentResult:
    """Result of component initialization."""

    name: str
    status: ComponentStatus
    duration_ms: float
    error_class: Optional[ErrorClass] = None
    error_message: Optional[str] = None
    fallback_used: Optional[str] = None
    attempt_count: int = 1


@dataclass
class ComponentSpec:
    """Specification for a component initialization."""

    name: str
    initializer: Union[Callable[[], Any], Callable[[], asyncio.Task]]
    dependencies: Set[str] = field(default_factory=set)
    timeout_seconds: float = 30.0
    fallback: Optional[Callable[[], Any]] = None
    is_fatal: bool = False
    expected_exceptions: Tuple[type, ...] = field(default_factory=tuple)
    retry_count: int = 0  # Number of retries on failure

    def __post_init__(self):
        """Validate component specification."""
        if self.timeout_seconds <= 0:
            raise ValueError(f"Component '{self.name}' timeout must be positive")
        if self.retry_count < 0:
            raise ValueError(
                f"Component '{self.name}' retry_count must be non-negative"
            )


class StartupOrchestrator:
    """Dependency-aware parallel startup orchestrator with timing and fallbacks.

    Features:
    - Dependency graph modeling with parallel execution of independent groups
    - Per-component timeout and retry logic
    - Graceful degradation with fallback activation
    - Comprehensive timing and metrics collection
    - Fatal vs recoverable error classification

    [RAT: CA, PA, REH] - Clean Architecture, Performance Awareness, Robust Error Handling
    """

    def __init__(self, max_concurrent: int = 4):
        """Initialize orchestrator.

        Args:
            max_concurrent: Maximum concurrent initializations to prevent thrash
        """
        self.logger = logging.getLogger(__name__)
        self.components: Dict[str, ComponentSpec] = {}
        self.results: Dict[str, ComponentResult] = {}
        self.max_concurrent = max_concurrent
        self.degraded_mode = False
        self.degraded_components: Set[str] = set()

    def add_component(self, spec: ComponentSpec) -> None:
        """Add a component specification to the orchestration plan.

        Args:
            spec: Component specification with dependencies and configuration
        """
        # Validate dependencies exist
        for dep in spec.dependencies:
            if dep not in self.components:
                raise ValueError(
                    f"Component '{spec.name}' depends on undefined component '{dep}'"
                )

        self.components[spec.name] = spec
        self.logger.debug(
            f"🔧 Added component: {spec.name} (deps: {spec.dependencies})",
            extra={"subsys": "startup"},
        )

    def _build_dependency_groups(self) -> List[List[str]]:
        """Build ordered groups of components that can run in parallel.

        Returns:
            List of component groups, where each group can run concurrently
        """
        # Topological sort with grouping
        remaining = set(self.components.keys())
        completed = set()
        groups = []

        while remaining:
            # Find components with all dependencies satisfied
            ready = {
                name
                for name in remaining
                if self.components[name].dependencies.issubset(completed)
            }

            if not ready:
                # Circular dependency detected
                raise ValueError(f"Circular dependencies detected in: {remaining}")

            groups.append(list(ready))
            completed.update(ready)
            remaining -= ready

        return groups

    async def _initialize_component(self, spec: ComponentSpec) -> ComponentResult:
        """Initialize a single component with timeout, retry, and fallback logic.

        Args:
            spec: Component specification

        Returns:
            ComponentResult with timing and status information
        """
        start_time = time.time()
        last_error = None

        for attempt in range(spec.retry_count + 1):
            try:
                self.logger.info(
                    f"🔧 Initializing {spec.name} (attempt {attempt + 1}/{spec.retry_count + 1})",
                    extra={"subsys": "startup"},
                )

                # Run initializer with timeout
                await asyncio.wait_for(
                    self._run_initializer(spec.initializer),
                    timeout=spec.timeout_seconds,
                )

                duration_ms = (time.time() - start_time) * 1000

                # Success path
                metrics.increment(
                    METRIC_COMPONENT_INIT_SUCCESS, {"component": spec.name}
                )
                metrics.gauge(
                    METRIC_COMPONENT_LAST_INIT_TIMESTAMP,
                    time.time(),
                    {"component": spec.name},
                )

                self.logger.info(
                    f"✅ {spec.name} initialized successfully ({duration_ms:.1f}ms)",
                    extra={"subsys": "startup"},
                )

                return ComponentResult(
                    name=spec.name,
                    status=ComponentStatus.SUCCESS,
                    duration_ms=duration_ms,
                    attempt_count=attempt + 1,
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {spec.timeout_seconds}s"
                error_class = ErrorClass.EXTERNAL_TIMEOUT
                self.logger.warning(
                    f"⏱️ {spec.name} initialization timeout (attempt {attempt + 1})",
                    extra={"subsys": "startup"},
                )

            except spec.expected_exceptions as e:
                last_error = str(e)
                error_class = self._classify_error(e)
                self.logger.warning(
                    f"⚠️ {spec.name} initialization failed: {e} (attempt {attempt + 1})",
                    extra={"subsys": "startup"},
                )

            except Exception as e:
                last_error = str(e)
                error_class = ErrorClass.PROGRAMMING_ERROR
                self.logger.error(
                    f"❌ {spec.name} unexpected error: {e} (attempt {attempt + 1})",
                    exc_info=True,
                    extra={"subsys": "startup"},
                )

            # Don't retry on fatal errors or programming errors
            if (
                error_class in (ErrorClass.PROGRAMMING_ERROR, ErrorClass.CONFIG_ERROR)
                or spec.is_fatal
            ):
                break

        duration_ms = (time.time() - start_time) * 1000

        # All attempts failed - try fallback
        if spec.fallback:
            try:
                self.logger.info(
                    f"🔄 Attempting fallback for {spec.name}",
                    extra={"subsys": "startup"},
                )

                await asyncio.wait_for(
                    self._run_initializer(spec.fallback),
                    timeout=spec.timeout_seconds / 2,  # Shorter timeout for fallbacks
                )

                self.degraded_mode = True
                self.degraded_components.add(spec.name)
                metrics.gauge(METRIC_DEGRADED_MODE, 1)

                self.logger.warning(
                    f"🔄 {spec.name} using fallback after {spec.retry_count + 1} attempts",
                    extra={"subsys": "startup"},
                )

                return ComponentResult(
                    name=spec.name,
                    status=ComponentStatus.DEGRADED,
                    duration_ms=duration_ms,
                    error_class=error_class,
                    error_message=last_error,
                    fallback_used="enabled",
                    attempt_count=spec.retry_count + 1,
                )

            except Exception as fallback_error:
                self.logger.error(
                    f"❌ {spec.name} fallback also failed: {fallback_error}",
                    exc_info=True,
                    extra={"subsys": "startup"},
                )

        # Complete failure
        metrics.increment(METRIC_COMPONENT_INIT_FAILURE, {"component": spec.name})

        return ComponentResult(
            name=spec.name,
            status=ComponentStatus.FAILED,
            duration_ms=duration_ms,
            error_class=error_class,
            error_message=last_error,
            attempt_count=spec.retry_count + 1,
        )

    async def _run_initializer(self, initializer: Callable) -> Any:
        """Run component initializer, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(initializer):
            return await initializer()
        else:
            # Run blocking initializers in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, initializer)

    def _classify_error(self, error: Exception) -> ErrorClass:
        """Classify error for handling strategy determination."""
        type(error).__name__.lower()
        error_msg = str(error).lower()

        if any(
            keyword in error_msg for keyword in ["config", "configuration", "setting"]
        ):
            return ErrorClass.CONFIG_ERROR
        elif any(
            keyword in error_msg for keyword in ["timeout", "connection", "network"]
        ):
            return ErrorClass.EXTERNAL_TIMEOUT
        elif any(
            keyword in error_msg for keyword in ["not found", "unavailable", "missing"]
        ):
            return ErrorClass.DEPENDENCY_UNAVAILABLE
        else:
            return ErrorClass.PROGRAMMING_ERROR

    async def execute(self) -> Dict[str, ComponentResult]:
        """Execute the startup orchestration with parallel groups.

        Returns:
            Dictionary of component results with timing and status information
        """
        if not self.components:
            self.logger.warning(
                "No components registered for startup", extra={"subsys": "startup"}
            )
            return {}

        startup_start = time.time()

        try:
            # Build dependency-ordered groups
            groups = self._build_dependency_groups()
            metrics.gauge(METRIC_STARTUP_PARALLEL_GROUPS, len(groups))

            self.logger.info(
                f"🚀 Starting parallel initialization: {len(groups)} groups, {len(self.components)} components",
                extra={"subsys": "startup"},
            )

            # Execute groups sequentially, components within groups concurrently
            for group_idx, group in enumerate(groups):
                group_start = time.time()

                self.logger.info(
                    f"🔧 Initializing group {group_idx + 1}/{len(groups)}: {group}",
                    extra={"subsys": "startup"},
                )

                # Limit concurrency within each group
                semaphore = asyncio.Semaphore(min(self.max_concurrent, len(group)))

                async def init_with_semaphore(component_name: str):
                    async with semaphore:
                        spec = self.components[component_name]
                        return await self._initialize_component(spec)

                # Run group components concurrently
                tasks = [init_with_semaphore(name) for name in group]
                group_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for result in group_results:
                    if isinstance(result, ComponentResult):
                        self.results[result.name] = result
                        # Record timing metric
                        metrics.observe(
                            METRIC_STARTUP_COMPONENT_DURATION,
                            result.duration_ms / 1000,
                            {"component": result.name, "status": result.status.value},
                        )
                    else:
                        self.logger.error(
                            f"Unexpected result type: {result}",
                            extra={"subsys": "startup"},
                        )

                group_duration = (time.time() - group_start) * 1000
                self.logger.info(
                    f"✅ Group {group_idx + 1} completed in {group_duration:.1f}ms",
                    extra={"subsys": "startup"},
                )

                # Check for fatal failures before continuing
                fatal_failures = [
                    r
                    for r in group_results
                    if isinstance(r, ComponentResult)
                    and r.status == ComponentStatus.FAILED
                    and self.components[r.name].is_fatal
                ]

                if fatal_failures:
                    fatal_names = [f.name for f in fatal_failures]
                    raise RuntimeError(f"Fatal component failures: {fatal_names}")

            total_duration = time.time() - startup_start
            metrics.observe(METRIC_STARTUP_TOTAL_DURATION, total_duration)

            # Summary statistics
            success_count = sum(
                1 for r in self.results.values() if r.status == ComponentStatus.SUCCESS
            )
            degraded_count = sum(
                1 for r in self.results.values() if r.status == ComponentStatus.DEGRADED
            )
            failed_count = sum(
                1 for r in self.results.values() if r.status == ComponentStatus.FAILED
            )

            self.logger.info(
                f"🎉 Startup completed in {total_duration:.2f}s: "
                f"{success_count} success, {degraded_count} degraded, {failed_count} failed",
                extra={"subsys": "startup"},
            )

            return self.results

        except Exception as e:
            total_duration = time.time() - startup_start
            self.logger.error(
                f"💥 Startup orchestration failed after {total_duration:.2f}s: {e}",
                exc_info=True,
                extra={"subsys": "startup"},
            )
            raise

    def get_startup_summary(self) -> Dict[str, Any]:
        """Get structured startup summary for logging and health reporting.

        Returns:
            Dictionary with startup timing and status information
        """
        if not self.results:
            return {"status": "not_started", "components": []}

        components = []
        for name, result in self.results.items():
            components.append(
                {
                    "name": name,
                    "duration_ms": result.duration_ms,
                    "status": result.status.value,
                    "fallback_used": result.fallback_used,
                    "error_class": result.error_class.value
                    if result.error_class
                    else None,
                    "error_message": result.error_message,
                    "attempt_count": result.attempt_count,
                }
            )

        success_count = sum(
            1 for r in self.results.values() if r.status == ComponentStatus.SUCCESS
        )
        total_count = len(self.results)

        return {
            "status": "degraded" if self.degraded_mode else "ready",
            "total_components": total_count,
            "success_count": success_count,
            "degraded_count": len(self.degraded_components),
            "failed_count": total_count - success_count - len(self.degraded_components),
            "degraded_mode": self.degraded_mode,
            "components": components,
        }
