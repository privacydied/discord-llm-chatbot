"""Idle-TTL unload for fastembed ONNX sessions. [PA][REH]"""

from __future__ import annotations

import time

import pytest

from bot.rag.fastembed_embedding import FastEmbedEmbedding, unload_idle_models


@pytest.fixture
def embedder():
    inst = FastEmbedEmbedding()
    inst.model = object()  # simulate a loaded session without the real model
    inst.embedding_dim = 384
    return inst


class TestUnloadIdleModels:
    def test_unloads_after_ttl(self, embedder) -> None:
        embedder._last_used = time.monotonic() - 1000
        assert unload_idle_models(900) >= 1
        assert embedder.model is None

    def test_keeps_recently_used(self, embedder) -> None:
        embedder._last_used = time.monotonic()
        unload_idle_models(900)
        assert embedder.model is not None

    def test_disabled_ttl_is_noop(self, embedder) -> None:
        embedder._last_used = time.monotonic() - 1000
        assert unload_idle_models(0) == 0
        assert embedder.model is not None

    async def test_encode_reloads_after_unload(self, embedder, monkeypatch) -> None:
        """After an idle unload, the next encode lazily reloads. [REH]"""
        import numpy as np

        embedder._last_used = time.monotonic() - 1000
        unload_idle_models(900)
        assert embedder.model is None

        class FakeModel:
            def embed(self, texts):
                return [np.ones(384, dtype=np.float32) for _ in texts]

        def fake_load():
            embedder.model = FakeModel()
            embedder.embedding_dim = 384

        monkeypatch.setattr(embedder, "_load_sync", fake_load)

        result = await embedder.encode(["hello"])
        assert result.shape == (1, 384)
        assert embedder.model is not None
