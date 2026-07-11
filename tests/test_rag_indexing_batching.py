"""Tests for RAG batched write paths [PA].

Covers two write-amplification fixes from the disk-I/O audit:
- ChromaRAGBackend.add_documents_batch() commits multiple documents in a single
  ChromaDB write transaction instead of one per document.
- IndexingQueue's worker loop drains up to `batch_size` queued documents and
  hands them to the backend together, instead of one commit per document.
"""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from bot.rag.chroma_backend import ChromaRAGBackend
from bot.rag.indexing_queue import IndexingQueue, IndexingTask, IndexingTaskStatus


@pytest.mark.asyncio
async def test_add_documents_batch_single_commit() -> None:
    """Two documents added via add_documents_batch should hit ChromaDB exactly once."""
    with tempfile.TemporaryDirectory() as temp_dir, patch("chromadb.PersistentClient") as mock_client_class:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        mock_embedding = AsyncMock()
        mock_embedding.get_embedding_dimension.return_value = 384
        # Each call to encode() may be asked for a different number of chunks;
        # return however many rows are requested based on input length.
        mock_embedding.encode.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)

        backend = ChromaRAGBackend(db_path=temp_dir, embedding_model=mock_embedding)
        await backend.initialize()

        results = await backend.add_documents_batch(
            [
                {
                    "source_id": "doc_a",
                    "text": "Alpha document with enough content to be chunked into at least one usable piece of text.",
                    "metadata": {},
                },
                {
                    "source_id": "doc_b",
                    "text": "Bravo document with enough content to be chunked into at least one usable piece of text.",
                    "metadata": {},
                },
            ],
        )

        assert set(results.keys()) == {"doc_a", "doc_b"}
        assert all(len(docs) > 0 for docs in results.values())
        # One combined write for both documents, not two separate ones.
        assert mock_collection.upsert.call_count == 1
        assert mock_collection.add.call_count == 0


@pytest.mark.asyncio
async def test_add_documents_batch_skips_empty_without_failing_others() -> None:
    """An empty/invalid item in the batch shouldn't block the rest of the batch."""
    with tempfile.TemporaryDirectory() as temp_dir, patch("chromadb.PersistentClient") as mock_client_class:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        mock_embedding = AsyncMock()
        mock_embedding.get_embedding_dimension.return_value = 384
        mock_embedding.encode.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)

        backend = ChromaRAGBackend(db_path=temp_dir, embedding_model=mock_embedding)
        await backend.initialize()

        results = await backend.add_documents_batch(
            [
                {"source_id": "empty_doc", "text": "   ", "metadata": {}},
                {
                    "source_id": "good_doc",
                    "text": "Perfectly good document with plenty of real content to survive chunk filtering.",
                    "metadata": {},
                },
            ],
        )

        assert results["empty_doc"] == []
        assert len(results["good_doc"]) > 0
        assert mock_collection.upsert.call_count == 1


@pytest.mark.asyncio
async def test_indexing_queue_batches_worker_commits() -> None:
    """Enqueuing several documents back-to-back should result in one backend
    add_documents_batch() call instead of N separate add_document() calls, as
    long as the backend supports batching.
    """
    fake_backend = MagicMock()
    fake_backend.add_documents_batch = AsyncMock(
        return_value={"doc_1": ["chunk"], "doc_2": ["chunk"], "doc_3": ["chunk"]},
    )
    fake_backend.add_document = AsyncMock(side_effect=AssertionError("should not be called for a multi-item batch"))

    queue = IndexingQueue(fake_backend, max_queue_size=100, num_workers=1, batch_size=10, enabled=True)

    for i in range(1, 4):
        task = IndexingTask(source_id=f"doc_{i}", text=f"content {i}")
        await queue.enqueue_task(task)

    await queue.start_workers()
    try:
        # Give the single worker a moment to drain the queue in one batch.
        for _ in range(50):
            if fake_backend.add_documents_batch.await_count >= 1:
                break
            await asyncio.sleep(0.05)
    finally:
        await queue.shutdown(timeout=2.0)

    assert fake_backend.add_documents_batch.await_count == 1
    call_args = fake_backend.add_documents_batch.await_args.args[0]
    assert {item["source_id"] for item in call_args} == {"doc_1", "doc_2", "doc_3"}


@pytest.mark.asyncio
async def test_indexing_queue_falls_back_to_per_task_without_batch_support() -> None:
    """Backends that only implement add_document (no add_documents_batch) should
    keep working exactly as before -- one call per task.
    """
    fake_backend = MagicMock(spec=["add_document"])
    fake_backend.add_document = AsyncMock(return_value=True)

    queue = IndexingQueue(fake_backend, max_queue_size=100, num_workers=1, batch_size=10, enabled=True)

    task = IndexingTask(source_id="solo_doc", text="solo content")
    await queue.enqueue_task(task)

    await queue.start_workers()
    try:
        for _ in range(50):
            if fake_backend.add_document.await_count >= 1:
                break
            await asyncio.sleep(0.05)
    finally:
        await queue.shutdown(timeout=2.0)

    assert fake_backend.add_document.await_count == 1
    assert task.status == IndexingTaskStatus.COMPLETED
