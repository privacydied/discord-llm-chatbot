from bot.stt_pipeline.runtime import load_stt_runtime_compat, parse_stt_max_ram_mb


def test_parse_stt_max_ram_mb_valid(monkeypatch) -> None:
    monkeypatch.setenv("STT_MAX_RAM_MB", "512")
    assert parse_stt_max_ram_mb() == 512


def test_parse_stt_max_ram_mb_invalid_or_non_positive(monkeypatch) -> None:
    monkeypatch.setenv("STT_MAX_RAM_MB", "0")
    assert parse_stt_max_ram_mb() is None

    monkeypatch.setenv("STT_MAX_RAM_MB", "-1")
    assert parse_stt_max_ram_mb() is None

    monkeypatch.setenv("STT_MAX_RAM_MB", "abc")
    assert parse_stt_max_ram_mb() is None


def test_load_stt_runtime_compat(monkeypatch) -> None:
    monkeypatch.setenv("YOUTUBE_TRANSCRIPT_FIRST", "0")
    monkeypatch.setenv("STT_MAX_RAM_MB", "256")
    cfg = load_stt_runtime_compat()
    assert cfg.youtube_transcript_first is False
    assert cfg.max_ram_mb == 256
