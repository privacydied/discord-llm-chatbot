from types import SimpleNamespace

from bot.stt_pipeline.stitch import run_stitch_stage


class _Spans:
    def __init__(self) -> None:
        self.calls = []

    def start(self, stage: str) -> None:
        self.calls.append(("start", stage, {}))

    def end(self, stage: str, **kwargs) -> None:
        self.calls.append(("end", stage, kwargs))


def test_run_stitch_stage_with_summary_callback() -> None:
    spans = _Spans()
    pre = SimpleNamespace()
    transcript = SimpleNamespace(cache_hit=True)
    summary_calls = []

    def _build() -> str:
        return "result text"

    def _summary(spans_arg, pre_arg, transcript_arg, cache_hit=None) -> None:
        summary_calls.append((spans_arg, pre_arg, transcript_arg, cache_hit))

    result = run_stitch_stage(
        spans=spans,
        pre=pre,
        transcript=transcript,
        build_result=_build,
        log_summary=_summary,
    )

    assert result == "result text"
    assert spans.calls == [
        ("start", "stitch", {}),
        ("end", "stitch", {"ok": True}),
    ]
    assert summary_calls == [(spans, pre, transcript, True)]


def test_run_stitch_stage_without_summary_callback() -> None:
    spans = _Spans()
    transcript = SimpleNamespace(cache_hit=False)

    result = run_stitch_stage(
        spans=spans,
        pre=SimpleNamespace(),
        transcript=transcript,
        build_result=lambda: {"ok": True},
    )

    assert result == {"ok": True}
    assert spans.calls == [
        ("start", "stitch", {}),
        ("end", "stitch", {"ok": True}),
    ]
