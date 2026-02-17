from bot.router import Router


class DummyBot:
    def __init__(self):
        self.config = {}
        self.tts_manager = None
        self.loop = None


class CaptureLogger:
    def __init__(self):
        self.info_lines = []

    def info(self, message, *args, **kwargs):
        if args:
            message = message % args
        self.info_lines.append(str(message))

    def debug(self, *args, **kwargs):
        return None


def test_build_visual_anchored_system_prompt_returns_none_without_visual_facts() -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()

    out = router._build_visual_anchored_system_prompt("plain text only")

    assert out is None
    assert router.logger.info_lines == []


def test_build_visual_anchored_system_prompt_builds_anchor_and_logs_primary() -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()

    out = router._build_visual_anchored_system_prompt("VISUAL_FACTS:\nimage notes")

    assert out is not None
    assert "[VISUAL-ANALYSIS-ANCHOR]" in out
    assert "Do not claim there is no image" in out
    assert "text.anchor | visual_facts_detected=true" in router.logger.info_lines


def test_build_visual_anchored_system_prompt_logs_fallback_variant() -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()

    out = router._build_visual_anchored_system_prompt(
        "vl prompt output:\ndetails", fallback=True
    )

    assert out is not None
    assert "[VISUAL-ANALYSIS-ANCHOR]" in out
    assert (
        "text.anchor | visual_facts_detected=true (fallback)"
        in router.logger.info_lines
    )
