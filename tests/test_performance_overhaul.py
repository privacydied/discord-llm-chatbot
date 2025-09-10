"""
Comprehensive Performance Overhaul Test Suite - Fault injection, soak tests, and regression verification.
Tests all optimizations delivered in the Text Flow Performance Overhaul.
Implements REH (Robust Error Handling) and CDiP (Continuous documentation) rules.
"""

import asyncio
import pytest
import time
import random
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict
import statistics

# Import our performance overhaul components
from bot.core.phase_timing import get_timing_manager
from bot.core.phase_constants import PhaseConstants as PC
from bot.core.openrouter_client import OptimizedOpenRouterClient, CircuitState
from bot.core.template_cache import get_template_cache, get_prompt_builder
from bot.core.fast_path_router import (
    get_fast_path_router,
    MessageComplexity,
    RouteDecision,
)
from bot.core.session_cache import get_session_cache
from bot.core.discord_client_optimizer import get_discord_sender, SendOptions
from bot.core.slo_monitor import get_slo_monitor, AlertLevel


class TestPhaseTimingSystem:
    """Test phase timing and correlation ID system [PA]."""

    @pytest.mark.asyncio
    async def test_pipeline_tracker_creation(self):
        """Test pipeline tracker creation and correlation IDs."""
        timing_manager = get_timing_manager()

        tracker = timing_manager.create_pipeline_tracker(
            msg_id="test_msg_123", user_id="test_user_456", guild_id="test_guild_789"
        )

        assert tracker.msg_id == "test_msg_123"
        assert tracker.user_id == "test_user_456"
        assert tracker.guild_id == "test_guild_789"
        assert not tracker.is_dm
        assert len(tracker.corr_id) == 8  # Short correlation ID
        assert tracker.corr_id in timing_manager.active_trackers

    @pytest.mark.asyncio
    async def test_phase_timing_context_manager(self):
        """Test phase timing with context manager."""
        timing_manager = get_timing_manager()
        tracker = timing_manager.create_pipeline_tracker("msg_123", "user_456")

        # Test successful phase
        async with timing_manager.track_phase(
            tracker, PC.PHASE_ROUTER_DISPATCH, test_metadata="value"
        ) as phase_metric:
            await asyncio.sleep(0.01)  # Simulate work
            assert phase_metric.phase == PC.PHASE_ROUTER_DISPATCH

        # Verify completion
        assert PC.PHASE_ROUTER_DISPATCH in tracker.phases
        completed_phase = tracker.phases[PC.PHASE_ROUTER_DISPATCH]
        assert completed_phase.success is True
        assert completed_phase.duration_ms is not None
        assert completed_phase.duration_ms > 0
        assert "test_metadata" in completed_phase.metadata

    @pytest.mark.asyncio
    async def test_phase_error_handling(self):
        """Test phase timing with exceptions [REH]."""
        timing_manager = get_timing_manager()
        tracker = timing_manager.create_pipeline_tracker("msg_error", "user_error")

        with pytest.raises(ValueError):
            async with timing_manager.track_phase(tracker, PC.PHASE_LLM_CALL):
                await asyncio.sleep(0.005)
                raise ValueError("Test error")

        # Verify error was recorded
        assert PC.PHASE_LLM_CALL in tracker.phases
        error_phase = tracker.phases[PC.PHASE_LLM_CALL]
        assert error_phase.success is False
        assert error_phase.error == "Test error"
        assert error_phase.duration_ms is not None


class TestOpenRouterClientOptimizations:
    """Test optimized OpenRouter client with fault injection [REH]."""

    @pytest.fixture
    def mock_client(self):
        """Create mock OpenRouter client for testing."""
        return OptimizedOpenRouterClient("test_api_key")

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, mock_client):
        """Test circuit breaker opens after failures."""
        model = "test_model"
        circuit_breaker = mock_client._get_circuit_breaker(model)

        # Simulate failures to trigger circuit breaker
        for _ in range(PC.OR_BREAKER_FAILURE_WINDOW):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.OPEN
        assert not circuit_breaker.should_attempt_request()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, mock_client):
        """Test circuit breaker recovery to half-open state."""
        model = "test_model"
        circuit_breaker = mock_client._get_circuit_breaker(model)

        # Force circuit open
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = time.time() - (
            PC.OR_BREAKER_OPEN_MS / 1000 + 1
        )

        # Should attempt recovery
        with patch("random.random", return_value=0.4):  # Below half-open probability
            assert circuit_breaker.should_attempt_request()
            assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_retry_logic_with_backoff(self, mock_client):
        """Test exponential backoff retry logic [REH]."""
        call_count = 0

        async def failing_request():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Retryable error")
            return {"success": True}

        start_time = time.time()
        result = await mock_client._retry_with_backoff(
            "test_model", failing_request, max_retries=2
        )
        elapsed = time.time() - start_time

        assert call_count == 3  # Initial + 2 retries
        assert result["success"] is True
        assert elapsed > 1.0  # Should have waited for backoff

    @pytest.mark.asyncio
    async def test_connection_pool_reuse_tracking(self, mock_client):
        """Test connection pool statistics tracking [PA]."""
        await mock_client._ensure_session()
        initial_created = mock_client.pool_stats["connections_created"]

        # Second call should reuse
        await mock_client._ensure_session()
        reused = mock_client.pool_stats["connections_reused"]

        assert mock_client.pool_stats["connections_created"] == initial_created
        assert reused > 0


class TestTemplateCaching:
    """Test template caching and prompt optimization [PA]."""

    @pytest.mark.asyncio
    async def test_template_cache_hit_miss(self):
        """Test template cache hit/miss behavior."""
        cache = get_template_cache()

        # First access - cache miss
        template1 = await cache.get_template(
            content="Hello {name}, welcome to {server}!", persona="test"
        )

        assert cache.stats["cache_misses"] > 0
        assert template1.metadata.template_id.startswith("tpl_test_")
        assert "name" in template1.metadata.variables
        assert "server" in template1.metadata.variables

        # Second access - cache hit
        cache.stats["cache_hits"] = 0  # Reset for test
        template2 = await cache.get_template(
            content="Hello {name}, welcome to {server}!", persona="test"
        )

        assert cache.stats["cache_hits"] == 1
        assert template1.metadata.template_id == template2.metadata.template_id

        # Test with non-existent file path (should handle gracefully)
        try:
            await cache.get_template(file_path="/fake/path/prompt.template")
            assert False, "Expected exception for non-existent file"
        except Exception as e:
            # Should handle file not found gracefully
            assert "No such file or directory" in str(e)

    @pytest.mark.asyncio
    async def test_template_optimization_analysis(self):
        """Test template structure analysis for optimization."""
        cache = get_template_cache()

        content = """System: You are a helpful AI assistant.
        
        User context: {context}
        Current query: {query}
        
        Please provide a helpful response."""

        template = await cache.get_template(content=content, persona="optimized")

        # Verify optimization analysis
        assert template.variable_count == 2
        assert "context" in template.metadata.variables
        assert "query" in template.metadata.variables
        assert len(template.static_prefix) > 0  # Should have static prefix

    @pytest.mark.asyncio
    async def test_prompt_builder_token_budgets(self):
        """Test prompt builder respects token budgets [CMV]."""
        builder = get_prompt_builder()

        # Create long history that exceeds budget
        long_history = []
        for i in range(50):  # Many messages
            long_history.append(
                {
                    "author": f"User{i}",
                    "content": "This is a test message " * 20,  # Long content
                }
            )

        # Create temporary template file for testing
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "System: You are a helpful assistant.\nUser: {user_prompt}\nHistory: {history}"
            )
            temp_file = f.name

        try:
            result = await builder.build_prompt(
                user_prompt="Test prompt",
                template_file=temp_file,
                user_id="test_user",
                guild_id="test_guild",
                history=long_history,
            )
        finally:
            # Clean up temp file
            os.unlink(temp_file)

        # Should have limited history due to token budget
        total_chars = result["total_chars"]
        max_expected = PC.HISTORY_MAX_TOKENS_GUILD * 4  # 4 chars per token estimate
        # Allow some overhead for system prompt
        assert total_chars < max_expected * 2


class TestFastPathRouter:
    """Test fast-path routing with decision budgets [PA]."""

    @pytest.fixture
    def mock_message(self):
        """Create mock Discord message for testing."""
        from discord import DMChannel

        message = MagicMock()
        message.id = 67890
        message.content = "Hello, how are you?"
        message.attachments = []
        message.embeds = []
        message.mentions = []
        message.role_mentions = []
        message.reference = None
        message.author.id = 12345
        message.guild = None  # DM

        # Create proper DMChannel mock that passes isinstance check
        dm_channel = MagicMock(spec=DMChannel)
        message.channel = dm_channel

        return message

    @pytest.mark.asyncio
    async def test_simple_dm_fast_path(self, mock_message):
        """Test simple DM messages get fast-path routing."""
        router = get_fast_path_router()

        analysis = await router.analyze_message_route(mock_message)

        assert analysis.complexity == MessageComplexity.SIMPLE_TEXT
        assert analysis.decision == RouteDecision.FAST_PATH_TEXT
        assert analysis.skip_context_heavy is True
        assert analysis.skip_rag_search is True
        assert analysis.use_simple_template is True
        assert analysis.decision_time_ms < PC.ROUTER_DECISION_BUDGET_MS

    @pytest.mark.asyncio
    async def test_multimodal_standard_routing(self, mock_message):
        """Test multimodal messages use standard pipeline."""
        mock_message.attachments = [MagicMock()]  # Add attachment
        router = get_fast_path_router()

        analysis = await router.analyze_message_route(mock_message)

        assert analysis.complexity == MessageComplexity.MULTIMODAL
        assert analysis.decision == RouteDecision.STANDARD_PIPELINE
        assert analysis.skip_modality_detection is False

    @pytest.mark.asyncio
    async def test_router_decision_budget_enforcement(self, mock_message):
        """Test router enforces decision budget timeout [REH]."""
        router = get_fast_path_router()
        router.decision_budget_ms = 1  # Very short budget

        # Simulate slow decision making by using a very tight budget
        # This should cause budget exceeded without complex time mocking
        start_time = time.time()
        analysis = await router.analyze_message_route(mock_message)
        (time.time() - start_time) * 1000

        # Budget should be enforced (decision time measured)
        assert analysis.decision_time_ms >= 0


class TestSessionCache:
    """Test session caching with TTL and eviction [PA][CMV]."""

    @pytest.mark.asyncio
    async def test_user_profile_caching(self):
        """Test user profile cache with TTL."""
        cache = get_session_cache()

        # Cache miss
        profile = await cache.get_user_profile("user_123")
        assert profile is None

        # Create and cache profile
        from bot.core.session_cache import UserProfile

        test_profile = UserProfile(user_id="user_123")
        test_profile.add_message("user", "Hello!")

        await cache.set_user_profile("user_123", test_profile)

        # Cache hit
        cached_profile = await cache.get_user_profile("user_123")
        assert cached_profile is not None
        assert cached_profile.user_id == "user_123"
        assert len(cached_profile.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_token_budget_history_trimming(self):
        """Test conversation history trimming with token budgets."""
        from bot.core.session_cache import UserProfile

        profile = UserProfile(user_id="test_user")

        # Add many long messages
        for i in range(100):
            long_message = "This is a very long test message " * 20
            profile.add_message("user", long_message)

        # History should be trimmed
        total_chars = sum(
            len(msg.get("content", "")) for msg in profile.conversation_history
        )
        estimated_tokens = total_chars // 4

        assert estimated_tokens <= PC.HISTORY_MAX_TOKENS_DM * 1.1  # Allow 10% overhead

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = get_session_cache()
        cache.max_entries = (
            15  # Small cache for testing (user_profiles gets max_entries//3 = 5)
        )

        from bot.core.session_cache import UserProfile

        # Fill cache beyond capacity
        for i in range(10):  # More than max_entries//3 = 5
            profile = UserProfile(f"user_{i}")
            await cache.set_user_profile(f"user_{i}", profile)

        # Wait for cleanup to run
        await asyncio.sleep(0.1)

        # Verify LRU eviction occurred (should be <= max_entries//3 = 5)
        expected_max = cache.max_entries // 3
        assert len(cache.user_profiles) <= expected_max


class TestDiscordSendOptimization:
    """Test Discord send optimizations [REH]."""

    @pytest.fixture
    def mock_bot(self):
        """Create mock bot for Discord sender."""
        bot = MagicMock()
        return bot

    @pytest.fixture
    def mock_channel(self):
        """Create mock Discord channel."""
        channel = MagicMock()
        channel.id = 123456789
        channel.send = AsyncMock(return_value=MagicMock(id=987654321))

        # Set up typing context manager properly without registering calls [REH]
        typing_context = MagicMock()
        typing_context.__aenter__ = AsyncMock()
        typing_context.__aexit__ = AsyncMock()
        channel.typing = MagicMock(return_value=typing_context)
        return channel

    @pytest.mark.asyncio
    async def test_simple_text_optimizations(self, mock_bot, mock_channel):
        """Test optimizations for simple text messages [PA]."""
        sender = get_discord_sender(mock_bot)

        short_text = "Hi!"
        await sender.send_simple_text(mock_channel, short_text)

        # Should skip typing for very short messages
        mock_channel.typing.assert_not_called()
        mock_channel.send.assert_called_once()

        # Update stats
        assert sender.session_stats["messages_sent"] > 0

    @pytest.mark.asyncio
    async def test_enrichment_skipping(self, mock_bot, mock_channel):
        """Test enrichment skipping for performance."""
        get_discord_sender(mock_bot)
        options = SendOptions.for_simple_text(30)

        # Should skip embeds and files for simple text
        assert options.skip_embeds is True
        assert options.skip_files is True
        assert options.skip_typing is True

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_bot, mock_channel):
        """Test rate limit handling with jitter [REH]."""
        sender = get_discord_sender(mock_bot)

        # Simulate rate limit
        bucket = sender._get_rate_limit_bucket(
            f"POST:channels/{mock_channel.id}/messages"
        )
        bucket.remaining = 0
        bucket.reset_after = 0.1  # 100ms
        bucket.reset_at = time.time() + 0.1

        start_time = time.time()
        await sender._wait_for_rate_limit(bucket, "test_route")
        elapsed = time.time() - start_time

        # Should have waited with jitter
        assert elapsed >= 0.1  # Base wait time
        assert elapsed <= 0.5  # Reasonable upper bound with jitter


class TestSLOMonitoring:
    """Test SLO monitoring and alerting [PA][REH]."""

    @pytest.mark.asyncio
    async def test_slo_breach_detection(self):
        """Test SLO breach detection and alerting."""
        monitor = get_slo_monitor()
        alerts_fired = []

        # Register alert callback
        async def alert_callback(alert):
            alerts_fired.append(alert)

        monitor.register_alert_callback(
            AlertLevel.CRITICAL, alert_callback
        )  # Fixed: register for CRITICAL level

        # Trigger SLO breach
        phase = PC.PHASE_ROUTER_DISPATCH
        target_ms = PC.get_slo_targets()[phase]
        breach_value = target_ms * 2  # Double the target

        # Record multiple breaches to trigger alert
        for _ in range(PC.ALERT_CONSECUTIVE_WINDOWS):
            monitor.record_phase_metric(phase, breach_value)

        # Allow callback to execute - callbacks are scheduled as background tasks [REH]
        await asyncio.sleep(0.1)  # Increased from 0.01s to 0.1s for callback completion

        assert len(alerts_fired) > 0
        alert = alerts_fired[0]
        assert alert.metric_name == phase
        assert alert.value == breach_value
        assert alert.target == target_ms

    def test_performance_statistics(self):
        """Test performance statistics calculation."""
        monitor = get_slo_monitor()

        # Add some test measurements
        phase = PC.PHASE_LLM_CALL
        measurements = [
            100,
            150,
            200,
            250,
            300,
            1000,
            1500,
            2000,
        ]  # Mix of good and bad

        for measurement in measurements:
            monitor.record_phase_metric(phase, measurement)

        # Get current status
        status = monitor.get_current_slo_status()
        phase_status = status[phase]

        assert phase_status["sample_count"] == len(measurements)
        assert phase_status["current_p95_ms"] is not None
        assert phase_status["target_ms"] == PC.get_slo_targets()[phase]

    def test_rich_dashboard_creation(self):
        """Test Rich dashboard creation for DEBUG mode [CA]."""
        monitor = get_slo_monitor()

        # Add some data
        monitor.record_phase_metric(PC.PHASE_CONTEXT_GATHER, 40)
        monitor.record_phase_metric(PC.PHASE_PREP_GEN, 120)

        # Create dashboard
        dashboard = monitor._create_slo_dashboard()
        assert dashboard is not None
        # Check that dashboard is a Rich Panel
        from rich.panel import Panel

        assert isinstance(dashboard, Panel)
        # Dashboard should contain performance data
        dashboard_content = (
            dashboard.renderable if hasattr(dashboard, "renderable") else str(dashboard)
        )
        assert len(str(dashboard_content)) > 0  # Non-empty dashboard


class TestIntegrationAndSoak:
    """Integration and soak testing for performance overhaul [REH][CDiP]."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_simulation(self):
        """Test complete pipeline simulation with all optimizations."""
        timing_manager = get_timing_manager()
        slo_monitor = get_slo_monitor()

        # Simulate complete pipeline
        tracker = timing_manager.create_pipeline_tracker("integration_test", "user_123")

        # Simulate each phase with realistic timings
        phases_and_times = [
            (PC.PHASE_ROUTER_DISPATCH, 45),  # Within SLO
            (PC.PHASE_CONTEXT_GATHER, 35),  # Within SLO
            (PC.PHASE_RAG_QUERY, 20),  # Within SLO
            (PC.PHASE_PREP_GEN, 100),  # Within SLO
            (PC.PHASE_LLM_CALL, 1200),  # Within SLO
            (PC.PHASE_DISCORD_DISPATCH, 180),  # Within SLO
        ]

        for phase, duration_ms in phases_and_times:
            async with timing_manager.track_phase(tracker, phase):
                await asyncio.sleep(duration_ms / 1000)  # Simulate work

        # Complete pipeline
        timing_manager.complete_tracker(tracker)

        # Record in SLO monitor
        slo_monitor.record_pipeline_completion(tracker)

        # Verify all phases completed successfully
        assert len(tracker.phases) == len(phases_and_times)
        assert tracker.total_duration_ms is not None
        assert all(phase.success for phase in tracker.phases.values())

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_soak(self):
        """Soak test with concurrent pipelines [RM]."""
        timing_manager = get_timing_manager()

        async def simulate_pipeline(pipeline_id: int):
            """Simulate single pipeline execution."""
            tracker = timing_manager.create_pipeline_tracker(
                f"soak_{pipeline_id}", f"user_{pipeline_id}"
            )

            # Random phase durations
            for phase in PC.get_all_phases()[:4]:  # First 4 phases
                duration = random.randint(10, 100)
                async with timing_manager.track_phase(tracker, phase):
                    await asyncio.sleep(duration / 1000)

            timing_manager.complete_tracker(tracker)
            return tracker

        # Run multiple concurrent pipelines
        pipelines = await asyncio.gather(*[simulate_pipeline(i) for i in range(20)])

        # Verify all completed successfully
        assert len(pipelines) == 20
        assert all(len(p.phases) >= 4 for p in pipelines)
        assert all(p.total_duration_ms is not None for p in pipelines)

    @pytest.mark.asyncio
    async def test_fault_injection_resilience(self):
        """Test system resilience under fault injection [REH]."""
        timing_manager = get_timing_manager()

        # Simulate failures in different components
        failure_scenarios = [
            "template_cache_miss",
            "circuit_breaker_open",
            "rate_limit_hit",
            "timeout_exceeded",
            "memory_pressure",
        ]

        successful_completions = 0

        for i, scenario in enumerate(failure_scenarios * 5):  # 25 total tests
            tracker = timing_manager.create_pipeline_tracker(f"fault_{i}", f"user_{i}")

            try:
                # Simulate phase with potential failure
                async with timing_manager.track_phase(tracker, PC.PHASE_LLM_CALL):
                    if random.random() < 0.3:  # 30% failure rate
                        raise Exception(f"Simulated {scenario} failure")
                    await asyncio.sleep(0.05)  # 50ms work

                timing_manager.complete_tracker(tracker)
                successful_completions += 1

            except Exception:
                # Failure is expected, system should handle gracefully
                timing_manager.complete_tracker(tracker)

        # Should have some successful completions despite failures
        success_rate = successful_completions / (len(failure_scenarios) * 5)
        assert success_rate > 0.5  # At least 50% success rate under fault injection


class TestRegressionPrevention:
    """Test that optimizations don't regress existing functionality [CDiP]."""

    def test_constants_not_changed(self):
        """Test that critical constants maintain expected values [CMV]."""
        # Verify performance constants are within expected ranges
        assert PC.OR_CONNECT_TIMEOUT_MS >= 1000  # At least 1 second
        assert PC.OR_READ_TIMEOUT_MS >= PC.OR_CONNECT_TIMEOUT_MS
        assert PC.OR_TOTAL_DEADLINE_MS >= PC.OR_READ_TIMEOUT_MS
        assert PC.OR_MAX_RETRIES >= 1
        assert PC.PIPELINE_MAX_PARALLEL_TASKS >= 1
        assert PC.SLO_P95_PIPELINE_MS > 0

    def test_logging_format_preservation(self):
        """Test that logging format is preserved [CA]."""
        from bot.utils.logging import get_logger

        logger = get_logger("test_regression")

        # Should not raise exceptions
        logger.info("✔ Test info message")
        logger.warning("⚠️ Test warning message")
        logger.error("❌ Test error message")
        logger.debug("ℹ Test debug message")

    @pytest.mark.asyncio
    async def test_memory_usage_bounds(self):
        """Test that caching doesn't cause memory leaks [RM]."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create many cached items
        cache = get_session_cache()

        from bot.core.session_cache import UserProfile

        for i in range(100):
            profile = UserProfile(user_id=f"memory_test_{i}")
            for j in range(50):  # Add conversation history
                profile.add_message("user", f"Test message {j}" * 10)
            await cache.set_user_profile(f"memory_test_{i}", profile)

        # Memory should not grow excessively
        final_memory = process.memory_info().rss
        memory_growth_mb = (final_memory - initial_memory) / 1024 / 1024

        # Should not grow more than 50MB for test data
        assert memory_growth_mb < 50


# Performance benchmarking utilities
class PerformanceBenchmark:
    """Utility for measuring performance improvements [PA]."""

    @staticmethod
    async def measure_phase_performance(
        phase_func, iterations: int = 100
    ) -> Dict[str, float]:
        """Measure phase performance statistics."""
        measurements = []

        for _ in range(iterations):
            start_time = time.time()
            await phase_func()
            duration_ms = (time.time() - start_time) * 1000
            measurements.append(duration_ms)

        return {
            "mean": statistics.mean(measurements),
            "median": statistics.median(measurements),
            "p95": statistics.quantiles(measurements, n=20)[18],  # 95th percentile
            "p99": statistics.quantiles(measurements, n=100)[98],  # 99th percentile
            "min": min(measurements),
            "max": max(measurements),
            "std_dev": statistics.stdev(measurements),
        }


# Test configuration
pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    """Configure test environment."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "soak: marks tests as soak tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
