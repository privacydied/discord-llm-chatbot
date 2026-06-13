from types import SimpleNamespace

from bot.stt_pipeline.spec_select import select_initial_model_spec


class _Logger:
    def __init__(self) -> None:
        self.calls = []

    def info(self, msg, *args) -> None:
        self.calls.append((msg, args))


def test_select_initial_model_spec_keeps_default_for_short_audio() -> None:
    base = SimpleNamespace(size="base")
    manager = SimpleNamespace(
        default_spec=base,
        downgrade_spec=lambda _spec: SimpleNamespace(size="tiny"),
    )
    logger = _Logger()

    result = select_initial_model_spec(
        manager=manager,
        duration_in_s=30.0,
        downgrade_threshold_s=120.0,
        logger=logger,
    )

    assert result is base
    assert logger.calls == []


def test_select_initial_model_spec_downgrades_and_logs_for_long_audio() -> None:
    base = SimpleNamespace(size="base")
    tiny = SimpleNamespace(size="tiny")
    manager = SimpleNamespace(
        default_spec=base,
        downgrade_spec=lambda _spec: tiny,
    )
    logger = _Logger()

    result = select_initial_model_spec(
        manager=manager,
        duration_in_s=121.0,
        downgrade_threshold_s=120.0,
        logger=logger,
    )

    assert result is tiny
    assert len(logger.calls) == 1
    msg, args = logger.calls[0]
    assert msg == "whisper.model_downgrade from=%s to=%s reason=long_audio"
    assert args == ("base", "tiny")
