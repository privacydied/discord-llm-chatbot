from types import SimpleNamespace

from bot.stt_pipeline.ffmpeg_runtime import (
    ffmpeg_bin_has_aac,
    ffmpeg_candidates_from_env,
    ffmpeg_supports_aac_decoder,
    reset_ffmpeg_runtime_cache,
    resolve_ffmpeg_bin,
)


def test_ffmpeg_candidates_from_env_order_and_dedupe(monkeypatch) -> None:
    monkeypatch.setenv("STT_FFMPEG_BIN", "/usr/local/bin/ffmpeg7")
    monkeypatch.setenv("FFMPEG_BIN", "ffmpeg")
    monkeypatch.setenv("FFMPEG_BINARY", "ffmpeg")

    candidates = ffmpeg_candidates_from_env()

    assert candidates[0] == "/usr/local/bin/ffmpeg7"
    assert "ffmpeg7" in candidates
    assert "ffmpeg" in candidates
    assert len(candidates) == len(set(candidates))


def test_ffmpeg_supports_aac_decoder_true(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=" A..... aac\n A..... aac_fixed\n")

    monkeypatch.setattr("bot.stt_pipeline.ffmpeg_runtime.subprocess.run", fake_run)
    assert ffmpeg_supports_aac_decoder("ffmpeg") is True


def test_ffmpeg_supports_aac_decoder_false(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=" V..... h264\n")

    monkeypatch.setattr("bot.stt_pipeline.ffmpeg_runtime.subprocess.run", fake_run)
    assert ffmpeg_supports_aac_decoder("ffmpeg") is False


def test_resolve_ffmpeg_bin_caches_and_tracks_aac(monkeypatch) -> None:
    reset_ffmpeg_runtime_cache()
    monkeypatch.delenv("STT_FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)

    monkeypatch.setattr(
        "bot.stt_pipeline.ffmpeg_runtime.shutil.which",
        lambda name: "/usr/bin/ffmpeg7" if name == "ffmpeg7" else None,
    )
    monkeypatch.setattr(
        "bot.stt_pipeline.ffmpeg_runtime.ffmpeg_supports_aac_decoder",
        lambda _bin: True,
    )

    first = resolve_ffmpeg_bin()
    assert first == "/usr/bin/ffmpeg7"
    assert ffmpeg_bin_has_aac() is True

    # Cache hit: should still return cached value even if which() no longer resolves.
    monkeypatch.setattr(
        "bot.stt_pipeline.ffmpeg_runtime.shutil.which",
        lambda _name: None,
    )
    second = resolve_ffmpeg_bin()
    assert second == "/usr/bin/ffmpeg7"

    reset_ffmpeg_runtime_cache()


def test_resolve_ffmpeg_bin_raises_when_missing(monkeypatch) -> None:
    reset_ffmpeg_runtime_cache()
    monkeypatch.delenv("STT_FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setattr(
        "bot.stt_pipeline.ffmpeg_runtime.shutil.which",
        lambda _name: None,
    )

    try:
        resolve_ffmpeg_bin()
    except RuntimeError as exc:
        assert "ffmpeg executable not found" in str(exc)
    else:
        msg = "Expected resolve_ffmpeg_bin to raise RuntimeError"
        raise AssertionError(msg)
