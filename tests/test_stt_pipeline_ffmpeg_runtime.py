from types import SimpleNamespace

from bot.stt_pipeline.ffmpeg_runtime import (
    ffmpeg_candidates_from_env,
    ffmpeg_supports_aac_decoder,
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
        return SimpleNamespace(
            returncode=0, stdout=" A..... aac\n A..... aac_fixed\n"
        )

    monkeypatch.setattr("bot.stt_pipeline.ffmpeg_runtime.subprocess.run", fake_run)
    assert ffmpeg_supports_aac_decoder("ffmpeg") is True


def test_ffmpeg_supports_aac_decoder_false(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=" V..... h264\n")

    monkeypatch.setattr("bot.stt_pipeline.ffmpeg_runtime.subprocess.run", fake_run)
    assert ffmpeg_supports_aac_decoder("ffmpeg") is False
