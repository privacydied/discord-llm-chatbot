"""Comprehensive tests for observability integration system.

Tests all observability components including startup orchestrator, config validation,
health monitoring, background task monitoring, resource monitoring, and integration.

[RAT: CDiP, REH] - Continuous Documentation in Progress, Robust Error Handling
"""

import asyncio
import os
import pytest
import tempfile
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import observability components
from bot.core.observability_integration import ObservabilityManager, get_observability_manager
from bot.core.startup_orchestrator import StartupOrchestrator, ComponentSpec, ComponentStatus
from bot.core.config_validation import ConfigValidator, HealthMonitor, HealthStatus
from bot.core.background_task_monitor import BackgroundTaskMonitor, TaskConfig, RestartPolicy
from bot.core.resource_monitor import ResourceMonitor
from bot.metrics import metrics, is_degraded_mode


class TestObservabilityIntegration:
    """Test observability integration system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.obs_manager = ObservabilityManager()
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        # Reset any global state
        pass

    def test_observability_manager_initialization(self):
        """Test observability manager initializes correctly."""
        assert self.obs_manager.startup_orchestrator is not None
        assert self.obs_manager.health_monitor is not None
        assert self.obs_manager.task_monitor is not None
        assert self.obs_manager.resource_monitor is not None
        assert self.obs_manager.initialized is False
        assert self.obs_manager.start_time > 0

    @pytest.mark.asyncio
    async def test_initialize_observability_stack_success(self):
        """Test successful observability stack initialization."""
        with patch('bot.core.config_validation.validate_config_or_exit') as mock_validate:
            mock_validate.return_value = None
            
            with patch.object(self.obs_manager.resource_monitor, 'start_monitoring') as mock_start:
                mock_start.return_value = None
                
                with patch.object(self.obs_manager.task_monitor, 'start_watchdog') as mock_watchdog:
                    mock_watchdog.return_value = None
                    
                    with patch.object(self.obs_manager.task_monitor, 'start_task') as mock_start_task:
                        mock_start_task.return_value = None
                        
                        result = await self.obs_manager.initialize_observability_stack()
                        
                        assert result is True
                        assert self.obs_manager.initialized is True
                        mock_validate.assert_called_once()
                        mock_start.assert_called_once()
                        mock_watchdog.assert_called_once()

    @pytest.mark.asyncio  
    async def test_initialize_observability_stack_failure(self):
        """Test observability stack initialization failure handling."""
        with patch('bot.core.config_validation.validate_config_or_exit') as mock_validate:
            mock_validate.side_effect = Exception("Config validation failed")
            
            result = await self.obs_manager.initialize_observability_stack()
            
            assert result is False
            assert self.obs_manager.initialized is False

    def test_register_health_components(self):
        """Test health component registration."""
        with patch.object(self.obs_manager.health_monitor, 'register_component') as mock_register:
            self.obs_manager._register_health_components()
            
            # Should register at least core components
            assert mock_register.call_count >= 6
            
            # Verify core components were registered
            registered_components = [call[0][0] for call in mock_register.call_args_list]
            assert "discord_connection" in registered_components
            assert "message_router" in registered_components
            assert "metrics_system" in registered_components

    @pytest.mark.asyncio
    async def test_setup_background_task_monitoring(self):
        """Test background task monitoring setup."""
        with patch.object(self.obs_manager.task_monitor, 'register_task') as mock_register:
            with patch.object(self.obs_manager.task_monitor, 'start_watchdog') as mock_watchdog:
                with patch.object(self.obs_manager.task_monitor, 'start_task') as mock_start:
                    mock_watchdog.return_value = None
                    mock_start.return_value = None
                    
                    await self.obs_manager._setup_background_task_monitoring()
                    
                    # Should register at least 3 tasks
                    assert mock_register.call_count >= 3
                    mock_watchdog.assert_called_once()
                    mock_start.assert_called_once()

    @patch.dict(os.environ, {"OBS_PARALLEL_STARTUP": "true"})
    def test_configure_parallel_startup_enabled(self):
        """Test parallel startup configuration when enabled."""
        with patch.object(self.obs_manager.startup_orchestrator, 'add_component') as mock_add:
            self.obs_manager._configure_parallel_startup()
            
            # Should add multiple components
            assert mock_add.call_count > 10
            
            # Verify specific components were added
            added_components = [call[0][0].name for call in mock_add.call_args_list]
            assert "logging_system" in added_components
            assert "metrics_system" in added_components
            assert "discord_connection" in added_components

    @patch.dict(os.environ, {"OBS_PARALLEL_STARTUP": "false"})
    @pytest.mark.asyncio
    async def test_execute_startup_orchestration_disabled(self):
        """Test startup orchestration when parallel startup is disabled."""
        result = await self.obs_manager.execute_startup_orchestration()
        
        assert result["status"] == "sequential"
        assert result["parallel_enabled"] is False

    @patch.dict(os.environ, {"OBS_PARALLEL_STARTUP": "true"})
    @pytest.mark.asyncio
    async def test_execute_startup_orchestration_enabled(self):
        """Test startup orchestration when parallel startup is enabled."""
        # Mock the orchestrator results
        mock_results = {
            "logging_system": Mock(status=ComponentStatus.SUCCESS),
            "metrics_system": Mock(status=ComponentStatus.SUCCESS),
        }
        mock_summary = {
            "status": "success",
            "total_components": 2,
            "success_count": 2,
            "degraded_count": 0,
            "failed_count": 0,
            "components": [
                {
                    "name": "logging_system", 
                    "status": "success",
                    "duration_ms": 100.0,
                    "attempt_count": 1,
                    "fallback_used": False
                }
            ]
        }
        
        with patch.object(self.obs_manager.startup_orchestrator, 'execute') as mock_execute:
            with patch.object(self.obs_manager.startup_orchestrator, 'get_startup_summary') as mock_summary_method:
                with patch.object(self.obs_manager.health_monitor, 'update_component_health') as mock_health:
                    mock_execute.return_value = mock_results
                    mock_summary_method.return_value = mock_summary
                    
                    # Configure the orchestrator first
                    self.obs_manager._configure_parallel_startup()
                    
                    result = await self.obs_manager.execute_startup_orchestration()
                    
                    assert result["status"] == "success"
                    assert result["total_components"] == 2
                    mock_execute.assert_called_once()
                    mock_summary_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_comprehensive_health_status(self):
        """Test comprehensive health status retrieval."""
        mock_system_health = Mock(status=HealthStatus.READY)
        mock_task_statuses = {"test_task": {"status": "running"}}
        mock_resource_stats = {"memory_mb": 256, "cpu_percent": 15.5}
        
        with patch.object(self.obs_manager.health_monitor, 'get_health_status') as mock_health:
            with patch.object(self.obs_manager.task_monitor, 'get_all_task_statuses') as mock_tasks:
                with patch.object(self.obs_manager.resource_monitor, 'get_resource_stats') as mock_resources:
                    with patch.object(self.obs_manager.health_monitor, 'get_health_summary_json') as mock_summary:
                        mock_health.return_value = mock_system_health
                        mock_tasks.return_value = mock_task_statuses
                        mock_resources.return_value = mock_resource_stats
                        mock_summary.return_value = {"status": "ready", "components": {}}
                        
                        self.obs_manager.initialized = True
                        
                        result = await self.obs_manager.get_comprehensive_health_status()
                        
                        assert "observability_initialized" in result
                        assert result["observability_initialized"] is True
                        assert "background_tasks" in result
                        assert "resource_monitoring" in result
                        assert "metrics_degraded" in result
                        assert "uptime_seconds" in result

    @pytest.mark.asyncio
    async def test_shutdown_observability_stack(self):
        """Test observability stack shutdown."""
        with patch.object(self.obs_manager.resource_monitor, 'stop_monitoring') as mock_stop_resources:
            with patch.object(self.obs_manager.task_monitor, 'stop_all_tasks') as mock_stop_tasks:
                mock_stop_resources.return_value = None
                mock_stop_tasks.return_value = None
                
                await self.obs_manager.shutdown_observability_stack()
                
                mock_stop_resources.assert_called_once()
                mock_stop_tasks.assert_called_once()

    def test_get_observability_manager_singleton(self):
        """Test observability manager singleton pattern."""
        manager1 = get_observability_manager()
        manager2 = get_observability_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, ObservabilityManager)


class TestStartupOrchestratorIntegration:
    """Test startup orchestrator integration."""

    def test_component_spec_creation(self):
        """Test component specification creation."""
        spec = ComponentSpec(
            name="test_component",
            initializer=AsyncMock(),
            timeout_seconds=10.0,
            dependencies={"dependency1"},
            is_fatal=True
        )
        
        assert spec.name == "test_component"
        assert spec.timeout_seconds == 10.0
        assert "dependency1" in spec.dependencies
        assert spec.is_fatal is True

    @pytest.mark.asyncio
    async def test_orchestrator_with_dependencies(self):
        """Test orchestrator respects dependencies."""
        orchestrator = StartupOrchestrator()
        
        # Create mock initializers
        init1 = AsyncMock()
        init2 = AsyncMock()
        
        # Add components with dependencies
        spec1 = ComponentSpec(name="component1", initializer=init1, timeout_seconds=5.0)
        spec2 = ComponentSpec(
            name="component2", 
            initializer=init2, 
            dependencies={"component1"},
            timeout_seconds=5.0
        )
        
        orchestrator.add_component(spec1)
        orchestrator.add_component(spec2)
        
        results = await orchestrator.execute()
        
        # Both should succeed
        assert results["component1"].status == ComponentStatus.SUCCESS
        assert results["component2"].status == ComponentStatus.SUCCESS
        
        # Verify initialization order
        init1.assert_called_once()
        init2.assert_called_once()


class TestBackgroundTaskIntegration:
    """Test background task monitoring integration."""

    @pytest.mark.asyncio
    async def test_task_heartbeat_wrapper(self):
        """Test task heartbeat wrapper functionality."""
        monitor = BackgroundTaskMonitor()
        
        config = TaskConfig(
            name="test_task",
            task_func=AsyncMock(),
            heartbeat_interval=1.0,
            staleness_threshold=3.0,
            restart_policy=RestartPolicy.ON_FAILURE
        )
        
        monitor.register_task(config)
        wrapper = monitor.get_heartbeat_wrapper("test_task")
        
        assert wrapper is not None
        
        # Test heartbeat
        async with wrapper:
            wrapper.heartbeat("Test message")
            status = monitor.get_task_status("test_task")
            assert status is not None
            assert status["last_heartbeat_message"] == "Test message"

    @pytest.mark.asyncio
    async def test_task_monitoring_lifecycle(self):
        """Test complete task monitoring lifecycle."""
        monitor = BackgroundTaskMonitor()
        
        # Mock task that runs briefly
        async def mock_task():
            wrapper = monitor.get_heartbeat_wrapper("lifecycle_test")
            async with wrapper:
                wrapper.heartbeat("Task started")
                await asyncio.sleep(0.1)
                wrapper.heartbeat("Task completed")
        
        config = TaskConfig(
            name="lifecycle_test",
            task_func=mock_task,
            heartbeat_interval=0.5,
            staleness_threshold=1.0,
            restart_policy=RestartPolicy.ON_FAILURE
        )
        
        monitor.register_task(config)
        
        # Start the task
        await monitor.start_task("lifecycle_test")
        await asyncio.sleep(0.2)  # Let it run briefly
        
        # Check status
        status = monitor.get_task_status("lifecycle_test")
        assert status is not None
        assert status["is_running"] is True


class TestResourceMonitoringIntegration:
    """Test resource monitoring integration."""

    @pytest.mark.asyncio
    async def test_resource_monitor_startup(self):
        """Test resource monitor startup."""
        monitor = ResourceMonitor()
        
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
            mock_process.return_value.cpu_percent.return_value = 25.0
            
            await monitor.start_monitoring()
            await asyncio.sleep(0.1)  # Let monitoring run briefly
            
            stats = monitor.get_resource_stats()
            assert "memory_mb" in stats
            assert "cpu_percent" in stats
            
            await monitor.stop_monitoring()

    @pytest.mark.asyncio 
    async def test_event_loop_lag_measurement(self):
        """Test event loop lag measurement."""
        monitor = ResourceMonitor()
        
        # Simulate some load
        start_time = time.time()
        await asyncio.sleep(0.01)  # Small delay
        
        lag = monitor._measure_event_loop_lag()
        assert lag >= 0.0
        assert lag < 1.0  # Should be small for this test


class TestHealthMonitoringIntegration:
    """Test health monitoring integration."""

    @pytest.mark.asyncio
    async def test_health_status_comprehensive(self):
        """Test comprehensive health status reporting."""
        monitor = HealthMonitor()
        
        # Register some components
        monitor.register_component("test_component1", HealthStatus.READY)
        monitor.register_component("test_component2", HealthStatus.DEGRADED, "Minor issue")
        
        health = await monitor.get_health_status()
        
        assert health.status in [HealthStatus.READY, HealthStatus.DEGRADED]
        assert len(health.components) == 2
        
        # Test JSON serialization
        json_summary = monitor.get_health_summary_json(health)
        assert "status" in json_summary
        assert "components" in json_summary
        assert "liveness" in json_summary
        assert "readiness" in json_summary


class TestConfigValidationIntegration:
    """Test configuration validation integration."""

    def test_config_validator_basic_validation(self):
        """Test basic configuration validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("TEST_KEY=test_value\n")
            f.write("TEST_NUMBER=42\n")
            temp_file = f.name
        
        try:
            validator = ConfigValidator()
            
            # Define validation rules
            rules = {
                "TEST_KEY": {"required": True, "type": str},
                "TEST_NUMBER": {"required": True, "type": int, "min": 1, "max": 100}
            }
            
            # Mock environment
            with patch.dict(os.environ, {"TEST_KEY": "test_value", "TEST_NUMBER": "42"}):
                errors = validator.validate_config(rules)
                assert len(errors) == 0
                
        finally:
            os.unlink(temp_file)

    def test_config_validator_validation_failure(self):
        """Test configuration validation failure handling."""
        validator = ConfigValidator()
        
        rules = {
            "REQUIRED_KEY": {"required": True, "type": str},
            "NUMERIC_KEY": {"required": True, "type": int, "min": 10}
        }
        
        # Test missing required key
        with patch.dict(os.environ, {"NUMERIC_KEY": "5"}, clear=True):
            errors = validator.validate_config(rules)
            assert len(errors) > 0
            assert any("REQUIRED_KEY" in error for error in errors)
            assert any("NUMERIC_KEY" in error and "minimum" in error.lower() for error in errors)


# Integration test fixtures
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "DISCORD_TOKEN": "test_token",
        "OPENAI_API_KEY": "test_key",
        "LOG_LEVEL": "INFO",
        "PROMETHEUS_ENABLED": "true",
        "OBS_PARALLEL_STARTUP": "false"
    }


@pytest.fixture
async def initialized_observability_manager():
    """Initialized observability manager for testing."""
    manager = ObservabilityManager()
    
    with patch('bot.core.config_validation.validate_config_or_exit'):
        with patch.object(manager.resource_monitor, 'start_monitoring'):
            with patch.object(manager.task_monitor, 'start_watchdog'):
                with patch.object(manager.task_monitor, 'start_task'):
                    await manager.initialize_observability_stack()
    
    yield manager
    
    # Cleanup
    await manager.shutdown_observability_stack()


# End-to-end integration tests
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_observability_lifecycle(self, initialized_observability_manager):
        """Test complete observability lifecycle."""
        manager = initialized_observability_manager
        
        # Verify initialization
        assert manager.initialized is True
        
        # Test health status
        health = await manager.get_comprehensive_health_status()
        assert health["observability_initialized"] is True
        
        # Test graceful shutdown
        await manager.shutdown_observability_stack()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OBS_PARALLEL_STARTUP": "true"})
    async def test_parallel_startup_integration(self):
        """Test parallel startup integration."""
        manager = ObservabilityManager()
        
        with patch('bot.core.config_validation.validate_config_or_exit'):
            await manager.initialize_observability_stack()
            
            # Configure parallel startup
            manager._configure_parallel_startup()
            
            # Mock component initializers to succeed quickly
            with patch.multiple(
                manager,
                _init_logging_validated=AsyncMock(),
                _init_metrics_system=AsyncMock(),
                _init_config_validation=AsyncMock(),
                _init_system_prompts=AsyncMock(),
                _init_memory_system=AsyncMock(),
                _init_tts_system=AsyncMock(),
                _init_rag_system=AsyncMock(),
                _init_ai_backends=AsyncMock(),
                _init_discord_connection=AsyncMock()
            ):
                results = await manager.execute_startup_orchestration()
                
                # Should complete successfully
                assert "status" in results
                assert results.get("total_components", 0) > 0
        
        await manager.shutdown_observability_stack()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
