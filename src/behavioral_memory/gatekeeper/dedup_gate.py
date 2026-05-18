"""Deduplication gate — wraps the memory deduplicator for the gatekeeper pipeline.

Implements gate 3 of the gatekeeper pipeline (Section III.E.3):
rejects traces with cosine similarity > 0.95 to existing entries.
"""

from __future__ import annotations

from behavioral_memory.core.schemas import ExecutionTrace
from behavioral_memory.memory.dedup import Deduplicator


class DeduplicationGate:
    """Thin wrapper around Deduplicator for use in the gatekeeper pipeline."""

    def __init__(self, deduplicator: Deduplicator) -> None:
        self._dedup = deduplicator

    def check(self, trace: ExecutionTrace) -> tuple[bool, float]:
        """Check if a trace would be rejected as a duplicate.

        Returns (is_unique, similarity_score).
        """
        is_dup, score = self._dedup.is_duplicate(trace)
        return not is_dup, score
