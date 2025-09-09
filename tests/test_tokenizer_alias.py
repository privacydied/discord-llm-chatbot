from importlib import reload


def test_tts_tokenizer_env_alias(monkeypatch):
    # Ensure alias is picked when canonical is unset
    monkeypatch.delenv('TTS_TOKENISER', raising=False)
    monkeypatch.setenv('TTS_TOKENIZER', 'espeak')

    import bot.tts.validation as val
    reload(val)

    # Pretend 'espeak' is available
    val.AVAILABLE_TOKENIZERS.clear()
    val.AVAILABLE_TOKENIZERS.update({'espeak'})

    tok = val.select_tokenizer_for_language('en', available_tokenizers={'espeak': True, 'grapheme': True})
    # For English, select_tokenizer_for_language enforces phonetic; env override should return the alias value
    assert tok == 'espeak'

