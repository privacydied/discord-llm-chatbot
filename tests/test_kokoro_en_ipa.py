
def test_en_forces_ipa(monkeypatch):
    from bot.tts.engines.kokoro import KokoroEngine
    eng = KokoroEngine()
    # enforce flag
    eng.force_ipa_en = True

    created = {}
    def fake_create(**kw):
        created.update(kw)
        class R: pass
        return R()
    eng.kd = type("KD", (), {"create": staticmethod(fake_create)})

    eng.synthesize("three things", language="en")

    assert created.get("disable_autodiscovery") is True
    assert "phonemes" in created
    assert isinstance(created["phonemes"], str)
    assert created.get("lang") == "en"
