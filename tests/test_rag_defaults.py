from importlib import reload


def test_rag_background_indexing_default_true(monkeypatch):
    # Ensure the environment is clean for RAG flags
    for key in [
        "RAG_BACKGROUND_INDEXING",
        "RAG_EAGER_VECTOR_LOAD",
    ]:
        monkeypatch.delenv(key, raising=False)

    # Import after environment cleanup
    import bot.rag.config as rag_config

    reload(rag_config)

    cfg = rag_config.reload_rag_config()
    assert cfg.background_indexing is True
