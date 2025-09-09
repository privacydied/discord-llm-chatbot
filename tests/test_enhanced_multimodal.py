"""
Comprehensive test suite for enhanced multimodal processing with provider fallback,
circuit breaker, per-item budget enforcement, and spam prevention.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch

# Import the modules we're testing
from bot.router import Router
from bot.enhanced_retry import EnhancedRetryManager, ProviderConfig
from bot.metrics.null_metrics import NoopMetrics
from bot.rag.embedding_interface import create_embedding_model, is_rag_legacy_mode


@pytest.fixture
def mock_bot():
    """Create a mock Discord bot."""
    bot = Mock()
    bot.config = Mock()
    bot.tts_manager = Mock()
    bot.loop = asyncio.get_event_loop()
    bot.user = Mock()
    bot.user.id = 12345
    
    # Use NoopMetrics to test metrics interface
    bot.metrics = NoopMetrics()
    
    return bot


@pytest.fixture
def router(mock_bot):
    """Create a router instance for testing."""
    return Router(mock_bot)


@pytest.fixture
def retry_manager():
    """Create a retry manager instance for testing."""
    return EnhancedRetryManager()


class TestEnhancedRetrySystem:
    """Test the enhanced retry system with provider fallback and circuit breaker."""
    
    @pytest.mark.asyncio
    async def test_provider_fallback_success(self, retry_manager):
        """Test successful fallback to secondary provider."""
        call_count = 0
        
        def create_coro_factory(provider_config: ProviderConfig):
            async def coro():
                nonlocal call_count
                call_count += 1
                
                # First provider fails twice, second succeeds
                if provider_config.name == "openrouter" and provider_config.model == "moonshotai/kimi-vl-a3b-thinking:free":
                    raise Exception("502 Bad Gateway - Provider returned error")
                elif provider_config.name == "openrouter" and provider_config.model == "openai/gpt-4o-mini":
                    return "Success with fallback provider"
                else:
                    raise Exception("Unexpected provider")
            return coro
        
        result = await retry_manager.run_with_fallback(
            modality="vision",
            coro_factory=create_coro_factory,
            per_item_budget=30.0
        )
        
        assert result.success is True
        assert result.fallback_occurred is True
        assert "Success with fallback provider" in result.result
        assert result.provider_used == "openrouter:openai/gpt-4o-mini"
        assert call_count >= 2  # At least one failure + one success
    
    @pytest.mark.asyncio
    async def test_per_item_budget_enforcement(self, retry_manager):
        """Test that per-item budget is enforced."""
        start_time = time.time()
        
        def create_coro_factory(provider_config: ProviderConfig):
            async def coro():
                # Simulate slow operation that exceeds budget
                await asyncio.sleep(2.0)
                return "Should not reach here"
            return coro
        
        result = await retry_manager.run_with_fallback(
            modality="vision",
            coro_factory=create_coro_factory,
            per_item_budget=1.0  # Very tight budget
        )
        
        elapsed = time.time() - start_time
        assert result.success is False
        assert elapsed < 3.0  # Should abort early due to budget
        assert "budget" in str(result.error).lower() or result.attempts == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, retry_manager):
        """Test that circuit breaker opens after repeated failures."""
        failure_count = 0
        
        def create_coro_factory(provider_config: ProviderConfig):
            async def coro():
                nonlocal failure_count
                failure_count += 1
                raise Exception("502 Bad Gateway")
            return coro
        
        # First call should exhaust all providers
        result1 = await retry_manager.run_with_fallback(
            modality="vision",
            coro_factory=create_coro_factory,
            per_item_budget=30.0
        )
        
        assert result1.success is False
        initial_failures = failure_count
        
        # Second call should skip circuit-broken providers
        result2 = await retry_manager.run_with_fallback(
            modality="vision",
            coro_factory=create_coro_factory,
            per_item_budget=30.0
        )
        
        assert result2.success is False
        # Should have fewer attempts due to circuit breaker
        assert failure_count < initial_failures * 2
    
    @pytest.mark.asyncio
    async def test_non_retryable_error_fast_fail(self, retry_manager):
        """Test that non-retryable errors fail fast."""
        call_count = 0
        
        def create_coro_factory(provider_config: ProviderConfig):
            async def coro():
                nonlocal call_count
                call_count += 1
                raise Exception("401 Unauthorized")  # Non-retryable
            return coro
        
        result = await retry_manager.run_with_fallback(
            modality="vision",
            coro_factory=create_coro_factory,
            per_item_budget=30.0
        )
        
        assert result.success is False
        # Should fail fast with minimal attempts
        assert call_count <= len(retry_manager.provider_configs["vision"])


class TestSequentialProcessing:
    """Test sequential multimodal processing with no parallelism."""
    
    @pytest.mark.asyncio
    async def test_two_images_sequential(self, router):
        """Test that two images are processed sequentially, never in parallel."""
        processing_times = []
        processing_active = []
        
        async def mock_see_infer(*args, **kwargs):
            processing_active.append(len(processing_active))  # Track concurrent calls
            processing_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate processing time
            processing_active.pop()
            
            mock_response = Mock()
            mock_response.content = f"Image analysis at {time.time()}"
            mock_response.error = None
            return mock_response
        
        with patch('bot.router.see_infer', side_effect=mock_see_infer):
            # Create mock message with two image attachments
            message = Mock()
            message.id = 12345
            message.content = "Analyze these images"
            message.attachments = [
                Mock(filename="image1.jpg", content_type="image/jpeg", url="http://example.com/1.jpg"),
                Mock(filename="image2.jpg", content_type="image/jpeg", url="http://example.com/2.jpg")
            ]
            message.embeds = []
            
            # Mock attachment save method
            for attachment in message.attachments:
                attachment.save = AsyncMock()
            
            # Mock text flow to capture final result
            router._invoke_text_flow = AsyncMock()
            
            await router._process_multimodal_message_internal(message, "test context")
            
            # Verify sequential processing (no parallelism)
            assert max(processing_active) == 1, "Images were processed in parallel!"
            assert len(processing_times) == 2, "Expected exactly 2 image processing calls"
            
            # Verify text flow was called once with aggregated result
            router._invoke_text_flow.assert_called_once()
            aggregated_content = router._invoke_text_flow.call_args[0][0]
            assert "1/2" in aggregated_content and "2/2" in aggregated_content
    
    @pytest.mark.asyncio
    async def test_mixed_modalities_order(self, router):
        """Test processing order matches message order for mixed modalities."""
        processing_order = []
        
        async def mock_handlers(*args, **kwargs):
            # Track which handler was called
            if 'image' in str(args) or any('jpg' in str(arg) for arg in args):
                processing_order.append('image')
            elif 'pdf' in str(args) or any('pdf' in str(arg) for arg in args):
                processing_order.append('pdf')
            elif 'url' in str(args):
                processing_order.append('url')
            
            return "Mock processing result"
        
        # Mock all relevant processing methods
        router._process_image_from_attachment = AsyncMock(side_effect=mock_handlers)
        router._process_pdf_from_attachment = AsyncMock(side_effect=mock_handlers)
        router._handle_general_url = AsyncMock(side_effect=mock_handlers)
        router._invoke_text_flow = AsyncMock()
        
        # Create message with mixed modalities in specific order
        message = Mock()
        message.id = 12345
        message.content = "Process these in order"
        message.attachments = [
            Mock(filename="doc.pdf", content_type="application/pdf", url="http://example.com/doc.pdf"),
            Mock(filename="photo.jpg", content_type="image/jpeg", url="http://example.com/photo.jpg")
        ]
        message.embeds = [
            Mock(url="https://example.com/webpage", type="rich", image=None, thumbnail=None)
        ]
        
        # Mock attachment save methods
        for attachment in message.attachments:
            attachment.save = AsyncMock()
        
        await router._process_multimodal_message_internal(message, "test context")
        
        # Verify processing order matches message order: PDF, Image, URL
        expected_order = ['pdf', 'image', 'url']
        assert processing_order == expected_order, f"Expected {expected_order}, got {processing_order}"
    
    @pytest.mark.asyncio
    async def test_never_drop_items_on_failure(self, router):
        """Test that failed items are still included in aggregated result."""
        call_count = 0
        
        async def mock_image_handler(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First image processing failed")
            return "Second image processed successfully"
        
        router._process_image_from_attachment = AsyncMock(side_effect=mock_image_handler)
        router._invoke_text_flow = AsyncMock()
        
        # Create message with two images
        message = Mock()
        message.id = 12345
        message.content = "Process both images"
        message.attachments = [
            Mock(filename="fail.jpg", content_type="image/jpeg", url="http://example.com/fail.jpg"),
            Mock(filename="success.jpg", content_type="image/jpeg", url="http://example.com/success.jpg")
        ]
        message.embeds = []
        
        for attachment in message.attachments:
            attachment.save = AsyncMock()
        
        await router._process_multimodal_message_internal(message, "test context")
        
        # Verify both items appear in final result
        router._invoke_text_flow.assert_called_once()
        aggregated_content = router._invoke_text_flow.call_args[0][0]
        
        assert "1/2" in aggregated_content, "First item missing from aggregation"
        assert "2/2" in aggregated_content, "Second item missing from aggregation"
        assert "failed" in aggregated_content.lower(), "Failure not recorded"
        assert "success" in aggregated_content.lower(), "Success not recorded"


class TestMetricsInterface:
    """Test that metrics interface never throws exceptions."""
    
    def test_noop_metrics_all_methods(self):
        """Test that NoopMetrics implements all expected methods safely."""
        metrics = NoopMetrics()
        
        # Test all methods that should never throw
        metrics.define_counter("test_counter", "Test counter")
        metrics.define_histogram("test_histogram", "Test histogram")
        metrics.inc("test_counter", value=1, labels={"key": "value"})
        metrics.increment("test_counter", labels={"key": "value"}, value=1)
        metrics.observe("test_histogram", value=1.5, labels={"key": "value"})
        metrics.gauge("test_gauge", value=2.0, labels={"key": "value"})
        
        # Test timer context manager
        with metrics.timer("test_timer", labels={"key": "value"}):
            pass
        
        # All operations should complete without exceptions
        assert True
    
    @pytest.mark.asyncio
    async def test_router_metrics_never_throw(self, router):
        """Test that router metrics calls never throw exceptions."""
        # Test with NoopMetrics (should never throw)
        router._metric_inc("test_metric", {"key": "value"})
        
        # Test with broken metrics object
        router.bot.metrics = Mock()
        router.bot.metrics.increment = Mock(side_effect=Exception("Metrics broken"))
        router.bot.metrics.inc = Mock(side_effect=Exception("Metrics broken"))
        
        # Should not raise exception
        router._metric_inc("test_metric", {"key": "value"})
        
        # Test with metrics object missing methods
        router.bot.metrics = Mock(spec=[])  # Empty spec = no methods
        router._metric_inc("test_metric", {"key": "value"})
        
        assert True  # Should reach here without exceptions


class TestRAGMisconfigSuppression:
    """Test that RAG misconfig warnings are suppressed after first occurrence."""
    
    def test_rag_misconfig_warn_once(self):
        """Test that RAG misconfig warning appears only once."""
        # Reset global state
        import bot.rag.embedding_interface as embedding_module
        embedding_module._rag_misconfig_warned = False
        embedding_module._rag_legacy_mode = False
        
        with patch('bot.rag.embedding_interface.logger') as mock_logger:
            # First call should warn
            result1 = create_embedding_model("unknown_model_type")
            assert result1 is None
            assert is_rag_legacy_mode() is True
            mock_logger.warning.assert_called_once()
            
            # Second call should not warn again
            mock_logger.reset_mock()
            result2 = create_embedding_model("unknown_model_type")
            assert result2 is None
            assert is_rag_legacy_mode() is True
            mock_logger.warning.assert_not_called()  # No additional warning
    
    def test_rag_legacy_mode_persistence(self):
        """Test that RAG legacy mode persists across calls."""
        # Reset global state
        import bot.rag.embedding_interface as embedding_module
        embedding_module._rag_misconfig_warned = False
        embedding_module._rag_legacy_mode = False
        
        # Trigger misconfig
        create_embedding_model("invalid_type")
        assert is_rag_legacy_mode() is True
        
        # Should remain in legacy mode
        assert is_rag_legacy_mode() is True
        
        # Valid model types should still work
        with patch('bot.rag.embedding_interface.SentenceTransformerEmbedding'):
            result = create_embedding_model("sentence-transformers")
            assert result is not None


class TestProviderConfiguration:
    """Test provider configuration and fallback ladder."""
    
    def test_provider_config_loading(self, retry_manager):
        """Test that provider configurations are loaded correctly."""
        vision_providers = retry_manager.provider_configs.get("vision", [])
        text_providers = retry_manager.provider_configs.get("text", [])
        
        assert len(vision_providers) > 0, "No vision providers configured"
        assert len(text_providers) > 0, "No text providers configured"
        
        # Check that providers have required fields
        for provider in vision_providers:
            assert hasattr(provider, 'name')
            assert hasattr(provider, 'model')
            assert hasattr(provider, 'timeout')
            assert provider.timeout > 0
    
    def test_circuit_breaker_state_management(self, retry_manager):
        """Test circuit breaker state management."""
        provider_key = "test_provider:test_model"
        
        # Initially healthy
        assert retry_manager._is_provider_available(provider_key) is True
        
        # Record failures
        for _ in range(3):
            retry_manager._record_failure(provider_key)
        
        # Should be circuit open
        assert retry_manager._is_provider_available(provider_key) is False
        
        # Record success should reset
        retry_manager._record_success(provider_key)
        assert retry_manager._is_provider_available(provider_key) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
