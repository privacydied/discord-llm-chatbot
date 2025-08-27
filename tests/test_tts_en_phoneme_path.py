def test_en_uses_phoneme_only(monkeypatch):
    from bot.tts.engines.kokoro import KokoroEngine

    created = {}
    class KDStub:
        def __init__(self, *a, **k): created["use_tokenizer"] = k.get("use_tokenizer", True)
        def create(self, **k): created.update(k); return "/tmp/f.wav"

    eng = KokoroEngine()
    eng._get_kokoro_direct = lambda **k: KDStub(**k)
    eng._wav_to_bytes = lambda p: b"WAV"

    out = eng.synthesize("one two three", language="en")
    assert out == b"WAV"
    assert created["use_tokenizer"] is False
    assert created.get("phonemes"), "phonemes must be supplied"
    assert created.get("disable_autodiscovery") is True

def test_no_error_when_no_tokenizer_if_phonemes(monkeypatch, caplog):
    from bot.kokoro_direct_fixed import KokoroDirect
    kd = KokoroDirect("/model.onnx", "/voices.bin", use_tokenizer=False)
    # create must not log 'No tokenizer available' nor invoke grapheme path
    caplog.clear()
    kd._synthesize_from_ipa = lambda ipa, **k: "/tmp/ok.wav"  # monkeypatch the final call
    kd.create(phonemes="t ɹ iː", voice="af_heart", lang="en", speed=1.0, disable_autodiscovery=True)
    assert not any("No tokenizer available" in r.message for r in caplog.records)
