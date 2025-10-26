import sys
import types

import bot.tts.eng_g2p_local as g2p


def _reset_official(monkeypatch):
    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER_STATE", "uninitialized", raising=False)
    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER", None, raising=False)


def test_text_to_ipa_prefers_official_tokenizer(monkeypatch):
    _reset_official(monkeypatch)

    class FakeTokenizer:
        def phonemize(self, text, lang="en-us", norm=True):
            assert lang == "en-us"
            assert norm is True
            assert text == "testing"
            return "tˈɛstɪŋ"

    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER_STATE", "ready", raising=False)
    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER", FakeTokenizer(), raising=False)

    # Remove cmudict to ensure fallback is not used when official path succeeds
    monkeypatch.setitem(sys.modules, "cmudict", None)

    ipa = g2p.text_to_ipa("testing")
    assert ipa == "tˈɛstɪŋ"


def test_text_to_ipa_falls_back_to_cmudict(monkeypatch):
    _reset_official(monkeypatch)

    fake_cmudict = types.SimpleNamespace(
        dict=lambda: {"testing": [["T", "EH1", "S", "T", "IH0", "NG"]]}
    )

    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER_STATE", "failed", raising=False)
    monkeypatch.setattr(g2p, "cmudict", fake_cmudict, raising=False)
    monkeypatch.setitem(sys.modules, "cmudict", fake_cmudict)

    ipa = g2p.text_to_ipa("testing")
    assert ipa == "t ɛ s t ɪ ŋ"


def test_text_to_ipa_maps_er_to_r_colored(monkeypatch):
    _reset_official(monkeypatch)

    fake_cmudict = types.SimpleNamespace(
        dict=lambda: {"bird": [["B", "ER1", "D"]]}
    )

    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER_STATE", "failed", raising=False)
    monkeypatch.setattr(g2p, "cmudict", fake_cmudict, raising=False)
    monkeypatch.setitem(sys.modules, "cmudict", fake_cmudict)

    ipa = g2p.text_to_ipa("bird")
    assert ipa == "b ɚ d"
