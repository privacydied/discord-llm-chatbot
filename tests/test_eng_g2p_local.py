import sys
import types

import bot.tts.eng_g2p_local as g2p


def _reset_official(monkeypatch):
    monkeypatch.setattr(
        g2p, "_OFFICIAL_TOKENIZER_STATE", "uninitialized", raising=False
    )
    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER", None, raising=False)
    monkeypatch.setattr(g2p, "_ESPEAK_TMPDIR_CONFIGURED", False, raising=False)
    monkeypatch.setattr(g2p, "_ESPEAK_TMPDIR_PATH", None, raising=False)


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

    fake_cmudict = types.SimpleNamespace(dict=lambda: {"bird": [["B", "ER1", "D"]]})

    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER_STATE", "failed", raising=False)
    monkeypatch.setattr(g2p, "cmudict", fake_cmudict, raising=False)
    monkeypatch.setitem(sys.modules, "cmudict", fake_cmudict)

    ipa = g2p.text_to_ipa("bird")
    assert ipa == "b ɚ d"


def test_text_to_ipa_retries_tempdir_failure(monkeypatch, tmp_path):
    _reset_official(monkeypatch)

    class RetryTokenizer:
        def __init__(self):
            self.calls = 0

        def phonemize(self, text, lang="en-us", norm=True):
            self.calls += 1
            if self.calls == 1:
                raise OSError("failed to map segment from shared object")
            return "tˈɛstɪŋ"

    retry_tokenizer = RetryTokenizer()
    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER_STATE", "ready", raising=False)
    monkeypatch.setattr(g2p, "_OFFICIAL_TOKENIZER", retry_tokenizer, raising=False)
    monkeypatch.setattr(g2p, "cmudict", None, raising=False)
    monkeypatch.setitem(sys.modules, "cmudict", None)

    configured = []

    def fake_configure():
        configured.append(True)
        return tmp_path

    monkeypatch.setattr(
        g2p, "_configure_official_tokenizer_tmpdir", fake_configure, raising=False
    )

    ipa = g2p.text_to_ipa("testing")

    assert ipa == "tˈɛstɪŋ"
    assert retry_tokenizer.calls == 2
    assert configured
