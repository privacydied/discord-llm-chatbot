def test_normalize_ipa_rewrites_to_official_symbols(monkeypatch):
    from bot.tts import ipa_vocab_loader
    from bot.tts import ipa_vocab_kokoro_v1

    sample_vocab = {
        "ɡ": 0,
        "ɚ": 1,
        "ʤ": 2,
        "ʧ": 3,
        "θ": 4,
    }

    monkeypatch.setattr(ipa_vocab_kokoro_v1, "PHONEME_TO_ID", sample_vocab, raising=False)
    monkeypatch.setattr(ipa_vocab_loader, "PHONEME_TO_ID", sample_vocab, raising=False)
    monkeypatch.setattr(ipa_vocab_loader, "IS_PLACEHOLDER", False, raising=False)

    cleaned = ipa_vocab_loader.normalize_ipa("g ɝ dʒ tʃ θ")
    assert cleaned == "ɡ ɚ ʤ ʧ θ"


def test_load_vocab_raises_when_vocab_missing(monkeypatch):
    from bot.tts import ipa_vocab_loader

    monkeypatch.setattr(ipa_vocab_loader, "PHONEME_TO_ID", {}, raising=False)
    monkeypatch.setattr(ipa_vocab_loader, "IS_PLACEHOLDER", True, raising=False)

    try:
        ipa_vocab_loader.load_vocab(session=None)
    except RuntimeError as exc:
        assert "vocabulary" in str(exc).lower()
    else:  # pragma: no cover - sanity guard
        raise AssertionError("Expected RuntimeError when vocabulary missing")
