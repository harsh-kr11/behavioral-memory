"""Semantic deduplication — check-before-add gate.

Before adding a new trace to memory, we check whether a near-identical
trace already exists. This keeps the store lean and prevents wasting
context-window tokens on redundant traces. The paper uses a cosine
similarity threshold of 0.95 (Section III.E.3).
"""

from __future__ import annotations

import logging
from typing import Any

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import ExecutionTrace

logger = logging.getLogger(__name__)


class Deduplicator:
    """Gate that prevents semantically redundant traces from entering memory."""

    def __init__(
        self,
        store: Any,
        threshold: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._store = store
        self._settings = settings or Settings()
        self.threshold = threshold or self._settings.similarity_dedup_threshold

    def is_duplicate(self, trace: ExecutionTrace) -> tuple[bool, float]:
        """Check if a trace is too similar to an existing one.

        Returns (is_duplicate, similarity_score).
        Both TraceStore and InMemoryTraceStore return cosine similarity
        (0-1, higher = more similar) from similarity_score().
        """
        score = self._store.similarity_score(trace.task_description)
        is_dup = score >= self.threshold
        if is_dup:
            logger.info(
                "Duplicate detected (%.3f >= %.3f): '%s'",
                score,
                self.threshold,
                trace.task_description[:60],
            )
        return is_dup, float(score)

    def add_if_unique(self, trace: ExecutionTrace) -> bool:
        """Add trace only if it passes the deduplication gate.

        Returns True if the trace was added, False if rejected as duplicate.
        """
        is_dup, score = self.is_duplicate(trace)
        if is_dup:
            logger.info("Skipping duplicate trace (score=%.3f)", score)
            return False
        self._store.add(trace)
        return True
