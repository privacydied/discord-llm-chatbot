def test_en_skips_tokenizer(monkeypatch):
    from bot.tts.engines.kokoro import KokoroEngine

    created = {}

    class KDStub:
        def __init__(self, *a, **k):
            created["use_tokenizer"] = k.get("use_tokenizer", True)

        def create(self, **k):
            created.update(k)
            return "/tmp/fake.wav"

    def kd_factory(*a, **k):
        return KDStub(*a, **k)

    eng = KokoroEngine()
    eng._get_kokoro_direct = kd_factory  # patch
    eng._wav_to_bytes = lambda p: b"WAV"

    out = eng.synthesize("three things", language="en")
    assert out == b"WAV"
    assert created["use_tokenizer"] is False
    assert "phonemes" in created
    assert created["disable_autodiscovery"] is True
