"""Configuration validation and health monitoring system.

Implements schema validation for environment/config with fail-fast diagnostics,
liveness/readiness checks, and comprehensive health reporting with degraded mode support.

[RAT: IV, SFT] - Input Validation, Security-First Thinking
"""

from __future__ import annotations

import os
import time
import asyncio
import psutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path

from bot.util.logging import get_logger


class HealthStatus(Enum):
    """Health status enumeration for components and overall system."""

    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


class ConfigErrorType(Enum):
    """Configuration error classification."""

    MISSING_REQUIRED = "missing_required"
    INVALID_VALUE = "invalid_value"
    INVALID_TYPE = "invalid_type"
    CROSS_FIELD_CONSTRAINT = "cross_field_constraint"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class ConfigConstraint:
    """Configuration field constraint specification."""

    key: str
    required: bool = False
    value_type: Optional[type] = None
    allowed_values: Optional[Set[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    file_must_exist: bool = False
    validator: Optional[Callable[[str], bool]] = None
    error_message: Optional[str] = None


@dataclass
class ComponentHealth:
    """Health information for a single component."""

    name: str
    status: HealthStatus
    last_init_timestamp: Optional[float] = None
    last_error: Optional[str] = None
    check_count: int = 0
    consecutive_failures: int = 0


@dataclass
class SystemHealth:
    """Overall system health with component details."""

    status: HealthStatus
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    prometheus_enabled: bool = False
    degraded_mode: bool = False
    degraded_reasons: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0
    rss_mb: float = 0.0
    event_loop_lag_ms: float = 0.0
    last_health_check: float = field(default_factory=time.time)


class ConfigValidator:
    """Schema validation for environment and configuration with fail-fast diagnostics.

    Features:
    - Required key validation
    - Type checking and value constraints
    - Cross-field constraint validation
    - File existence verification
    - Custom validator functions
    - Crisp diagnostic messages

    [RAT: IV, SFT] - Input Validation, Security-First Thinking
    """

    def __init__(self):
        """Initialize configuration validator."""
        self.logger = get_logger(__name__)
        self.constraints: List[ConfigConstraint] = []
        self._define_constraints()

    def _define_constraints(self) -> None:
        """Define configuration constraints for the Discord bot."""
        # Core Discord configuration
        self.constraints.extend(
            [
                ConfigConstraint(
                    key="DISCORD_TOKEN",
                    required=True,
                    error_message="Discord bot token is required. Get one from https://discord.com/developers/applications",
                ),
                ConfigConstraint(
                    key="DISCORD_INTENTS",
                    required=False,
                    allowed_values={"default", "all", "messages", "guild_messages"},
                    error_message="DISCORD_INTENTS must be one of: default, all, messages, guild_messages",
                ),
            ]
        )

        # AI/LLM configuration
        self.constraints.extend(
            [
                ConfigConstraint(
                    key="OPENAI_API_KEY",
                    required=False,  # May not be required if other providers configured
                    error_message="OpenAI API key required for GPT models",
                ),
                ConfigConstraint(
                    key="ANTHROPIC_API_KEY",
                    required=False,
                    error_message="Anthropic API key required for Claude models",
                ),
            ]
        )

        # Observability configuration
        self.constraints.extend(
            [
                ConfigConstraint(
                    key="OBS_ENABLE_PROMETHEUS",
                    required=False,
                    allowed_values={"true", "false"},
                    error_message="OBS_ENABLE_PROMETHEUS must be 'true' or 'false'",
                ),
                ConfigConstraint(
                    key="OBS_PARALLEL_STARTUP",
                    required=False,
                    allowed_values={"true", "false"},
                    error_message="OBS_PARALLEL_STARTUP must be 'true' or 'false'",
                ),
                ConfigConstraint(
                    key="OBS_ENABLE_HEALTHCHECKS",
                    required=False,
                    allowed_values={"true", "false"},
                    error_message="OBS_ENABLE_HEALTHCHECKS must be 'true' or 'false'",
                ),
                ConfigConstraint(
                    key="OBS_ENABLE_RESOURCE_METRICS",
                    required=False,
                    allowed_values={"true", "false"},
                    error_message="OBS_ENABLE_RESOURCE_METRICS must be 'true' or 'false'",
                ),
                ConfigConstraint(
                    key="PROMETHEUS_PORT",
                    required=False,
                    value_type=int,
                    min_value=1024,
                    max_value=65535,
                    error_message="PROMETHEUS_PORT must be between 1024-65535",
                ),
            ]
        )

        # Logging configuration
        self.constraints.extend(
            [
                ConfigConstraint(
                    key="LOG_LEVEL",
                    required=False,
                    allowed_values={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
                    error_message="LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
                ),
                ConfigConstraint(
                    key="LOG_JSONL_PATH",
                    required=False,
                    file_must_exist=False,  # Will be created if doesn't exist
                    error_message="LOG_JSONL_PATH must be a valid file path",
                ),
            ]
        )

        # RAG system configuration
        self.constraints.extend(
            [
                ConfigConstraint(
                    key="RAG_EAGER_VECTOR_LOAD",
                    required=False,
                    allowed_values={"true", "false"},
                    error_message="RAG_EAGER_VECTOR_LOAD must be 'true' or 'false'",
                ),
            ]
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration against all defined constraints.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            for constraint in self.constraints:
                error = self._validate_constraint(constraint)
                if error:
                    errors.append(error)

            # Cross-field constraint validation
            cross_field_errors = self._validate_cross_field_constraints()
            errors.extend(cross_field_errors)

        except Exception as e:
            errors.append(f"Configuration validation failed unexpectedly: {e}")

        return len(errors) == 0, errors

    def _validate_constraint(self, constraint: ConfigConstraint) -> Optional[str]:
        """Validate a single configuration constraint."""
        value = os.getenv(constraint.key)

        # Required field validation
        if constraint.required and not value:
            return (
                constraint.error_message
                or f"Required configuration '{constraint.key}' is missing"
            )

        # Skip further validation if field is not set and not required
        if not value:
            return None

        # Type validation
        if constraint.value_type:
            try:
                if constraint.value_type is int:
                    typed_value = int(value)
                elif constraint.value_type is float:
                    typed_value = float(value)
                elif constraint.value_type is bool:
                    typed_value = value.lower() in ("true", "1", "yes", "on")
                else:
                    typed_value = constraint.value_type(value)
            except (ValueError, TypeError):
                return (
                    f"Configuration '{constraint.key}' must be of type "
                    f"{constraint.value_type.__name__}"
                )
        else:
            typed_value = value

        # Allowed values validation
        if constraint.allowed_values and value not in constraint.allowed_values:
            allowed_list = ", ".join(sorted(constraint.allowed_values))
            return f"Configuration '{constraint.key}' value '{value}' not allowed. Must be one of: {allowed_list}"

        # Numeric range validation
        if constraint.min_value is not None or constraint.max_value is not None:
            try:
                numeric_value = (
                    float(typed_value)
                    if constraint.value_type is not int
                    else int(typed_value)
                )
                if (
                    constraint.min_value is not None
                    and numeric_value < constraint.min_value
                ):
                    return f"Configuration '{constraint.key}' value {numeric_value} is below minimum {constraint.min_value}"
                if (
                    constraint.max_value is not None
                    and numeric_value > constraint.max_value
                ):
                    return f"Configuration '{constraint.key}' value {numeric_value} is above maximum {constraint.max_value}"
            except (ValueError, TypeError):
                return f"Configuration '{constraint.key}' must be numeric for range validation"

        # File existence validation
        if constraint.file_must_exist:
            path = Path(value)
            if not path.exists():
                return f"Configuration '{constraint.key}' file '{value}' does not exist"
            if not path.is_file():
                return f"Configuration '{constraint.key}' path '{value}' is not a file"

        # Custom validator
        if constraint.validator:
            try:
                if not constraint.validator(value):
                    return (
                        constraint.error_message
                        or f"Configuration '{constraint.key}' failed custom validation"
                    )
            except Exception as e:
                return f"Configuration '{constraint.key}' custom validator error: {e}"

        return None

    def _validate_cross_field_constraints(self) -> List[str]:
        """Validate constraints that depend on multiple configuration fields."""
        errors = []

        # Ensure at least one AI provider is configured
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        ollama_host = os.getenv("OLLAMA_HOST")

        if not any([openai_key, anthropic_key, ollama_host]):
            errors.append(
                "At least one AI provider must be configured: OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_HOST"
            )

        # Prometheus configuration consistency
        prometheus_enabled = (
            os.getenv("OBS_ENABLE_PROMETHEUS", "false").lower() == "true"
        )
        prometheus_port = os.getenv("PROMETHEUS_PORT")
        prometheus_http = os.getenv("PROMETHEUS_HTTP_SERVER", "true").lower() == "true"

        if prometheus_enabled and prometheus_http and not prometheus_port:
            errors.append(
                "PROMETHEUS_PORT must be set when Prometheus HTTP server is enabled"
            )

        return errors


class HealthMonitor:
    """Liveness and readiness health monitoring with component tracking.

    Features:
    - Liveness checks (process up)
    - Readiness checks (all required components ready or fallbacks active)
    - Component health tracking with last init timestamp and error history
    - Resource monitoring (RSS, event loop lag)
    - Degraded mode detection and reporting

    [RAT: SFT, PA] - Security-First Thinking, Performance Awareness
    """

    def __init__(self):
        """Initialize health monitor."""
        self.logger = get_logger(__name__)
        self.start_time = time.time()
        self.components: Dict[str, ComponentHealth] = {}
        self._last_event_loop_check = time.time()

    def register_component(
        self, name: str, status: HealthStatus = HealthStatus.NOT_READY
    ) -> None:
        """Register a component for health monitoring."""
        self.components[name] = ComponentHealth(name=name, status=status)
        self.logger.debug(
            f"📊 Registered component for health monitoring: {name}",
            extra={"subsys": "health"},
        )

    def update_component_health(
        self, name: str, status: HealthStatus, error: Optional[str] = None
    ) -> None:
        """Update the health status of a component."""
        if name not in self.components:
            self.register_component(name)

        component = self.components[name]
        previous_status = component.status

        component.status = status
        component.check_count += 1
        component.last_init_timestamp = time.time()

        if status != HealthStatus.READY:
            component.consecutive_failures += 1
            component.last_error = error
        else:
            component.consecutive_failures = 0
            component.last_error = None

        # Log status changes
        if previous_status != status:
            status_icon = {"ready": "✅", "degraded": "⚠️", "not_ready": "❌"}
            self.logger.info(
                f"{status_icon.get(status.value, '?')} Component '{name}' health: {previous_status.value} → {status.value}",
                extra={"subsys": "health", "component": name},
            )

    async def check_liveness(self) -> bool:
        """Check if the process is alive and responsive.

        Returns:
            True if process is alive and event loop is responsive
        """
        try:
            # Basic process aliveness
            process = psutil.Process()
            if not process.is_running():
                return False

            # Event loop responsiveness check
            start_time = time.time()
            await asyncio.sleep(0.001)  # Small async operation
            response_time = (time.time() - start_time) * 1000

            # If this takes more than 100ms, event loop is severely blocked
            return response_time < 100.0

        except Exception as e:
            self.logger.error(f"Liveness check failed: {e}", extra={"subsys": "health"})
            return False

    async def check_readiness(self) -> Tuple[bool, List[str]]:
        """Check if all required components are ready or have fallbacks active.

        Returns:
            Tuple of (is_ready, reasons_if_not_ready)
        """
        not_ready_reasons = []

        try:
            for name, component in self.components.items():
                if component.status == HealthStatus.NOT_READY:
                    not_ready_reasons.append(
                        f"Component '{name}' not ready: {component.last_error or 'unknown error'}"
                    )

            # Check for critical system resources
            try:
                process = psutil.Process()
                memory_percent = process.memory_percent()
                if memory_percent > 90.0:  # 90% memory usage threshold
                    not_ready_reasons.append(
                        f"High memory usage: {memory_percent:.1f}%"
                    )
            except Exception:
                pass  # Non-critical check

            return len(not_ready_reasons) == 0, not_ready_reasons

        except Exception as e:
            self.logger.error(
                f"Readiness check failed: {e}", extra={"subsys": "health"}
            )
            return False, [f"Readiness check error: {e}"]

    async def measure_event_loop_lag(self) -> float:
        """Measure current event loop lag in milliseconds."""
        try:
            expected_sleep = 0.01  # 10ms
            start = time.time()
            await asyncio.sleep(expected_sleep)
            actual_duration = time.time() - start

            lag_ms = max(0, (actual_duration - expected_sleep) * 1000)
            self._last_event_loop_check = time.time()

            return lag_ms

        except Exception:
            return 0.0  # Fallback on error

    async def get_health_status(self) -> SystemHealth:
        """Get comprehensive system health status.

        Returns:
            SystemHealth object with all component and system metrics
        """
        try:
            # Perform health checks
            is_alive = await self.check_liveness()
            is_ready, ready_reasons = await self.check_readiness()

            # Determine overall status
            if not is_alive:
                overall_status = HealthStatus.NOT_READY
            elif not is_ready:
                overall_status = HealthStatus.NOT_READY
            elif any(
                c.status == HealthStatus.DEGRADED for c in self.components.values()
            ):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.READY

            # Collect system metrics
            try:
                process = psutil.Process()
                rss_mb = process.memory_info().rss / 1024 / 1024
            except Exception:
                rss_mb = 0.0

            event_loop_lag_ms = await self.measure_event_loop_lag()

            # Check for degraded mode from metrics system
            from bot.metrics import is_degraded_mode, get_degraded_reasons

            degraded_mode = is_degraded_mode()
            degraded_reasons = get_degraded_reasons() if degraded_mode else []

            # Check Prometheus enablement
            prometheus_enabled = (
                os.getenv("OBS_ENABLE_PROMETHEUS", "false").lower() == "true"
            )

            return SystemHealth(
                status=overall_status,
                components=self.components.copy(),
                prometheus_enabled=prometheus_enabled,
                degraded_mode=degraded_mode,
                degraded_reasons=degraded_reasons,
                uptime_seconds=time.time() - self.start_time,
                rss_mb=rss_mb,
                event_loop_lag_ms=event_loop_lag_ms,
                last_health_check=time.time(),
            )

        except Exception as e:
            self.logger.error(
                f"Failed to get health status: {e}",
                exc_info=True,
                extra={"subsys": "health"},
            )
            # Return minimal failed health status
            return SystemHealth(
                status=HealthStatus.NOT_READY,
                components={},
                last_health_check=time.time(),
            )

    def get_health_summary_json(self, health: SystemHealth) -> Dict[str, Any]:
        """Convert SystemHealth to JSON-serializable summary for logging/API."""
        components_data = []
        for name, component in health.components.items():
            components_data.append(
                {
                    "name": name,
                    "state": component.status.value,
                    "last_init_ts": component.last_init_timestamp,
                    "last_error": component.last_error,
                    "check_count": component.check_count,
                    "consecutive_failures": component.consecutive_failures,
                }
            )

        return {
            "status": health.status.value,
            "components": components_data,
            "prometheus_enabled": health.prometheus_enabled,
            "degraded_mode": health.degraded_mode,
            "degraded_reasons": health.degraded_reasons,
            "uptime_s": round(health.uptime_seconds, 2),
            "rss_mb": round(health.rss_mb, 1),
            "event_loop_lag_ms": round(health.event_loop_lag_ms, 2),
            "last_health_check": health.last_health_check,
        }


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def validate_config_or_exit() -> None:
    """Validate configuration and exit with crisp diagnostic on failure.

    [RAT: IV, SFT] - Input Validation, Security-First Thinking
    """
    logger = get_logger(__name__)

    try:
        validator = ConfigValidator()
        is_valid, errors = validator.validate()

        if not is_valid:
            # Single crisp diagnostic with all errors
            error_summary = (
                f"Configuration validation failed with {len(errors)} error(s):"
            )
            for i, error in enumerate(errors, 1):
                error_summary += f"\n  {i}. {error}"

            error_summary += "\n\nFix configuration errors and restart."

            logger.critical(error_summary, extra={"subsys": "config"})

            # Also emit to stderr for visibility
            import sys

            sys.stderr.write(f"CRITICAL: {error_summary}\n")
            sys.stderr.flush()

            # Exit with error code
            sys.exit(1)
        else:
            logger.info(
                "✅ Configuration validation passed", extra={"subsys": "config"}
            )

    except Exception as e:
        logger.critical(
            f"Configuration validation system failed: {e}",
            exc_info=True,
            extra={"subsys": "config"},
        )
        import sys

        sys.exit(2)
