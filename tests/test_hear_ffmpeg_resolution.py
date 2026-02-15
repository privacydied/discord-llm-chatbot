import pytest

from bot.exceptions import InferenceError
import bot.hear as hear


def _reset_ffmpeg_cache(monkeypatch) -> None:
    monkeypatch.setattr(hear, "_FFMPEG_BIN_CACHE", None)
    monkeypatch.setattr(hear, "_FFMPEG_BIN_HAS_AAC", None)


def test_resolve_ffmpeg_prefers_ffmpeg7(monkeypatch) -> None:
    _reset_ffmpeg_cache(monkeypatch)
    monkeypatch.delenv("STT_FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)

    mapping = {
        "ffmpeg7": "/usr/local/bin/ffmpeg7",
        "ffmpeg": "/bin/ffmpeg",
    }
    monkeypatch.setattr(hear.shutil, "which", lambda name: mapping.get(name))
    monkeypatch.setattr(
        hear,
        "_ffmpeg_supports_aac_decoder",
        lambda path: path.endswith("ffmpeg7"),
    )

    resolved = hear._resolve_ffmpeg_bin()

    assert resolved == "/usr/local/bin/ffmpeg7"
    assert hear._FFMPEG_BIN_HAS_AAC is True


def test_resolve_ffmpeg_honors_env_override(monkeypatch) -> None:
    _reset_ffmpeg_cache(monkeypatch)
    monkeypatch.setenv("STT_FFMPEG_BIN", "custom-ffmpeg")
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)

    mapping = {
        "custom-ffmpeg": "/opt/custom/ffmpeg",
        "ffmpeg7": "/usr/local/bin/ffmpeg7",
    }
    monkeypatch.setattr(hear.shutil, "which", lambda name: mapping.get(name))
    monkeypatch.setattr(hear, "_ffmpeg_supports_aac_decoder", lambda _path: True)

    resolved = hear._resolve_ffmpeg_bin()

    assert resolved == "/opt/custom/ffmpeg"


def test_resolve_ffmpeg_raises_when_missing(monkeypatch) -> None:
    _reset_ffmpeg_cache(monkeypatch)
    monkeypatch.delenv("STT_FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setattr(hear.shutil, "which", lambda _name: None)

    with pytest.raises(InferenceError, match="ffmpeg executable not found"):
        hear._resolve_ffmpeg_bin()
