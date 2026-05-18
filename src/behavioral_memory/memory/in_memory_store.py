"""In-memory trace store — no PostgreSQL required.

Drop-in replacement for TraceStore that keeps embeddings in memory
using numpy cosine similarity. Perfect for:
  - Running the benchmark without database setup
  - Local development and demos
  - CI/CD testing with a real LLM

Implements the same public interface as TraceStore so PlanEngine,
Deduplicator, and the full pipeline work identically.
"""

from __future__ import annotations

import logging

import numpy as np
from langchain_core.embeddings import Embeddings

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import ExecutionTrace

logger = logging.getLogger(__name__)


class InMemoryTraceStore:
    """Vector store backed by in-memory numpy arrays.

    Same interface as TraceStore but needs zero infrastructure.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        settings: Settings | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._settings = settings or Settings()
        self._traces: list[ExecutionTrace] = []
        self._vectors: list[list[float]] = []

    def search(
        self, query: str, k: int | None = None
    ) -> list[tuple[ExecutionTrace, float]]:
        k = k or self._settings.few_shot_k
        if not self._traces:
            return []

        query_vec = self._embeddings.embed_query(query)
        scores = self._cosine_similarities(query_vec, self._vectors)

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            results.append((self._traces[idx], float(scores[idx])))
        return results

    def add(self, trace: ExecutionTrace) -> None:
        vec = self._embeddings.embed_query(trace.task_description)
        self._traces.append(trace)
        self._vectors.append(vec)
        logger.info("Stored trace (in-memory): %s", trace.task_description[:80])

    def add_bulk(self, traces: list[ExecutionTrace]) -> int:
        texts = [t.task_description for t in traces]
        vecs = self._embeddings.embed_documents(texts)
        self._traces.extend(traces)
        self._vectors.extend(vecs)
        logger.info("Bulk-added %d traces (in-memory)", len(traces))
        return len(traces)

    def similarity_score(self, query: str) -> float:
        results = self.search(query, k=1)
        if not results:
            return 0.0
        return results[0][1]

    def count(self) -> int:
        return len(self._traces)

    @staticmethod
    def _cosine_similarities(
        query_vec: list[float], doc_vecs: list[list[float]]
    ) -> np.ndarray:
        q = np.array(query_vec)
        d = np.array(doc_vecs)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        d_norms = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-10)
        return d_norms @ q_norm
