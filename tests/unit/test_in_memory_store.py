"""Tests for InMemoryTraceStore — validates it matches TraceStore interface."""

from __future__ import annotations

from unittest.mock import MagicMock

from behavioral_memory.core.schemas import ExecutionTrace, ToolCall
from behavioral_memory.memory.in_memory_store import InMemoryTraceStore


def _make_trace(desc: str) -> ExecutionTrace:
    return ExecutionTrace(
        task_description=desc,
        tool_chain=[ToolCall(step_id="s1", tool_name="test_tool", parameters={"key": "val"})],
    )


def _make_mock_embeddings(dim: int = 4):
    """Create a mock embeddings model that returns deterministic vectors."""
    emb = MagicMock()

    def embed_query(text: str) -> list[float]:
        h = hash(text) % 10000
        return [float(h % (i + 2)) / 10.0 for i in range(dim)]

    def embed_documents(texts: list[str]) -> list[list[float]]:
        return [embed_query(t) for t in texts]

    emb.embed_query = embed_query
    emb.embed_documents = embed_documents
    return emb


class TestInMemoryTraceStore:
    def test_empty_search(self):
        store = InMemoryTraceStore(embeddings=_make_mock_embeddings())
        results = store.search("anything")
        assert results == []

    def test_add_and_count(self):
        store = InMemoryTraceStore(embeddings=_make_mock_embeddings())
        store.add(_make_trace("test task"))
        assert store.count() == 1

    def test_bulk_add(self):
        store = InMemoryTraceStore(embeddings=_make_mock_embeddings())
        traces = [_make_trace(f"task {i}") for i in range(5)]
        added = store.add_bulk(traces)
        assert added == 5
        assert store.count() == 5

    def test_search_returns_results(self):
        store = InMemoryTraceStore(embeddings=_make_mock_embeddings())
        store.add(_make_trace("build a data pipeline"))
        store.add(_make_trace("deploy a web application"))
        store.add(_make_trace("analyze revenue data"))

        results = store.search("data pipeline", k=2)
        assert len(results) == 2
        for trace, score in results:
            assert isinstance(trace, ExecutionTrace)
            assert isinstance(score, float)

    def test_search_respects_k(self):
        store = InMemoryTraceStore(embeddings=_make_mock_embeddings())
        for i in range(10):
            store.add(_make_trace(f"task number {i}"))

        results = store.search("query", k=3)
        assert len(results) == 3

    def test_similarity_score_empty(self):
        store = InMemoryTraceStore(embeddings=_make_mock_embeddings())
        assert store.similarity_score("anything") == 0.0

    def test_similarity_score_with_data(self):
        store = InMemoryTraceStore(embeddings=_make_mock_embeddings())
        store.add(_make_trace("test task"))
        score = store.similarity_score("test task")
        assert isinstance(score, float)
        assert score >= -1.0

    def test_interface_matches_trace_store(self):
        """Verify InMemoryTraceStore has the same public methods as TraceStore."""
        required_methods = ["search", "add", "add_bulk", "similarity_score", "count"]
        for method in required_methods:
            assert hasattr(InMemoryTraceStore, method), f"Missing method: {method}"
