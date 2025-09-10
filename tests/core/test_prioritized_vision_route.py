import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.router import Router
from bot.action import BotAction


@pytest.fixture
def mock_bot():
    """Create a lightweight mock bot with configurable settings."""
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.tts_manager = MagicMock()
    bot.loop = asyncio.get_event_loop()
    # Use a plain dict so Router reads feature flags directly
    bot.config = {}
    return bot


@pytest.fixture
def router(mock_bot):
    # Instantiate with VISION disabled to avoid heavy init; tests will toggle flags as needed
    r = Router(mock_bot)
    return r


@pytest.fixture
def mock_message():
    msg = MagicMock()
    msg.id = 111
    msg.content = ""
    msg.author = MagicMock()
    msg.author.id = 999
    msg.guild = MagicMock()
    msg.guild.id = 222
    msg.channel = MagicMock()
    msg.channel.send = AsyncMock()
    return msg


class TestPrioritizedVisionRoute:
    @pytest.mark.asyncio
    async def test_cfg_disabled_returns_none_and_increments_skipped(
        self, router, mock_message
    ):
        mock_message.content = "hello"
        router.config["VISION_ENABLED"] = False

        with patch.object(router, "_metric_inc") as metric_inc:
            result = await router._prioritized_vision_route(
                mock_message, context_str=""
            )

        assert result is None
        metric_inc.assert_any_call("vision.route.skipped", {"reason": "cfg_disabled"})

    @pytest.mark.asyncio
    async def test_direct_trigger_dry_run_returns_action_and_metrics(
        self, router, mock_message
    ):
        mock_message.content = "generate an image of a red cat"
        router.config["VISION_ENABLED"] = True
        router.config["VISION_DRY_RUN_MODE"] = True

        with (
            patch.object(router, "_metric_inc") as metric_inc,
            patch.object(
                router, "_handle_vision_generation", new_callable=AsyncMock
            ) as handle_vision,
        ):
            result = await router._prioritized_vision_route(
                mock_message, context_str=""
            )

        assert isinstance(result, BotAction)
        assert result.content.startswith("[DRY RUN]")
        handle_vision.assert_not_called()
        # direct precheck metric and dry-run metric
        metric_inc.assert_any_call("vision.route.direct", {"stage": "precheck"})
        metric_inc.assert_any_call("vision.route.dry_run", {"path": "direct"})

    @pytest.mark.asyncio
    async def test_direct_trigger_blocked_without_orchestrator(
        self, router, mock_message
    ):
        mock_message.content = "draw a dragon breathing fire"
        router.config["VISION_ENABLED"] = True
        router.config["VISION_DRY_RUN_MODE"] = False
        router._vision_orchestrator = None

        with patch.object(router, "_metric_inc") as metric_inc:
            result = await router._prioritized_vision_route(
                mock_message, context_str="ctx"
            )

        assert isinstance(result, BotAction)
        assert (
            result.content
            == "🚫 Vision generation is not available right now. Please try again later."
        )
        metric_inc.assert_any_call("vision.route.direct", {"stage": "precheck"})
        metric_inc.assert_any_call(
            "vision.route.blocked",
            {"reason": "orchestrator_unavailable", "path": "direct"},
        )

    @pytest.mark.asyncio
    async def test_direct_trigger_calls_handler_when_ready(self, router, mock_message):
        mock_message.content = "create a picture of a beach at sunset"
        router.config["VISION_ENABLED"] = True
        router.config["VISION_DRY_RUN_MODE"] = False
        router._vision_orchestrator = MagicMock()  # mark as available

        expected = BotAction(content="ok")
        with patch.object(
            router, "_handle_vision_generation", new=AsyncMock(return_value=expected)
        ) as handle_vision:
            result = await router._prioritized_vision_route(
                mock_message, context_str="ctx"
            )

        assert result is expected
        handle_vision.assert_called_once()

    @pytest.mark.asyncio
    async def test_intent_dry_run_returns_action_and_metrics(
        self, router, mock_message
    ):
        mock_message.content = "this is a normal request"
        router.config["VISION_ENABLED"] = True
        router.config["VISION_DRY_RUN_MODE"] = True
        router._vision_orchestrator = None  # shouldn't matter due to dry-run

        # Provide a fake intent result with use_vision=True
        class Decision:
            use_vision = True

        class IntentResult:
            def __init__(self):
                self.decision = Decision()
                self.confidence = 0.72
                self.extracted_params = MagicMock()

        router._vision_intent_router = MagicMock()
        router._vision_intent_router.determine_intent = AsyncMock(
            return_value=IntentResult()
        )

        with (
            patch.object(router, "_metric_inc") as metric_inc,
            patch.object(
                router, "_handle_vision_generation", new_callable=AsyncMock
            ) as handle_vision,
        ):
            result = await router._prioritized_vision_route(
                mock_message, context_str="ctx"
            )

        assert isinstance(result, BotAction)
        assert result.content.startswith("[DRY RUN]")
        handle_vision.assert_not_called()
        metric_inc.assert_any_call("vision.route.intent", {"stage": "precheck"})
        metric_inc.assert_any_call("vision.route.dry_run", {"path": "intent"})

    @pytest.mark.asyncio
    async def test_intent_blocked_without_orchestrator(self, router, mock_message):
        mock_message.content = "a regular sentence"
        router.config["VISION_ENABLED"] = True
        router.config["VISION_DRY_RUN_MODE"] = False
        router._vision_orchestrator = None

        # Fake intent result that requests vision
        class Decision:
            use_vision = True

        class IntentResult:
            def __init__(self):
                self.decision = Decision()
                self.confidence = 0.55
                self.extracted_params = MagicMock()

        router._vision_intent_router = MagicMock()
        router._vision_intent_router.determine_intent = AsyncMock(
            return_value=IntentResult()
        )

        with patch.object(router, "_metric_inc") as metric_inc:
            result = await router._prioritized_vision_route(
                mock_message, context_str="ctx"
            )

        assert isinstance(result, BotAction)
        assert (
            result.content
            == "🚫 Vision generation is not available right now. Please try again later."
        )
        metric_inc.assert_any_call("vision.route.intent", {"stage": "precheck"})
        metric_inc.assert_any_call(
            "vision.route.blocked",
            {"reason": "orchestrator_unavailable", "path": "intent"},
        )

    @pytest.mark.asyncio
    async def test_intent_calls_handler_when_ready(self, router, mock_message):
        mock_message.content = "non trigger text"
        router.config["VISION_ENABLED"] = True
        router.config["VISION_DRY_RUN_MODE"] = False
        router._vision_orchestrator = MagicMock()  # orchestrator available

        # Intent says to use vision
        class Decision:
            use_vision = True

        class IntentResult:
            def __init__(self):
                self.decision = Decision()
                self.confidence = 0.61
                self.extracted_params = MagicMock()

        router._vision_intent_router = MagicMock()
        router._vision_intent_router.determine_intent = AsyncMock(
            return_value=IntentResult()
        )

        expected = BotAction(content="ok")
        with patch.object(
            router, "_handle_vision_generation", new=AsyncMock(return_value=expected)
        ) as handle_vision:
            result = await router._prioritized_vision_route(
                mock_message, context_str="ctx"
            )

        assert result is expected
        handle_vision.assert_called_once()

    @pytest.mark.asyncio
    async def test_intent_error_returns_none_and_increments_metric(
        self, router, mock_message
    ):
        mock_message.content = "plain message"
        router.config["VISION_ENABLED"] = True
        router.config["VISION_DRY_RUN_MODE"] = False
        router._vision_orchestrator = MagicMock()
        router._vision_intent_router = MagicMock()
        router._vision_intent_router.determine_intent = AsyncMock(
            side_effect=Exception("boom")
        )

        with patch.object(router, "_metric_inc") as metric_inc:
            result = await router._prioritized_vision_route(
                mock_message, context_str="ctx"
            )

        assert result is None
        metric_inc.assert_any_call("vision.intent.error", None)


class TestDirectTriggerDetection:
    def test_detect_direct_trigger_extracts_prompt(self, router):
        content = "generate an image of a fluffy dog playing fetch"
        out = router._detect_direct_vision_triggers(content)
        assert out is not None
        assert out["task"] == "text_to_image"
        # Ensure leading filler words are removed (e.g., 'of a/an')
        assert out["prompt"].lower().startswith("fluffy dog")
        assert out["confidence"] >= 0.9

    def test_no_direct_trigger_returns_none(self, router):
        content = "let's just chat about something"
        out = router._detect_direct_vision_triggers(content)
        assert out is None


class TestMetricInc:
    def test_metric_inc_uses_increment_method(self, router):
        metrics = MagicMock()
        router.bot.metrics = metrics

        router._metric_inc("test.metric", {"a": "b"})
        metrics.increment.assert_called_once_with("test.metric", {"a": "b"})

    def test_metric_inc_falls_back_to_inc(self, router):
        class M:
            def __init__(self):
                self.increment = None  # simulate absence
                self._called = False

            def inc(self, metric_name, labels=None):
                self._called = True
                assert metric_name == "test.metric"
                assert labels == {"x": "y"}

        m = M()
        router.bot.metrics = m

        router._metric_inc("test.metric", {"x": "y"})
        assert m._called is True

    def test_metric_inc_swallow_exceptions(self, router):
        class M:
            def increment(self, *args, **kwargs):
                raise RuntimeError("fail")

        router.bot.metrics = M()

        # Should not raise
        router._metric_inc("test.metric.err", {"k": "v"})
